/* ============================================================
 * mediasplit.js – Video-/Audiodatei in Video- und Audio-Teil trennen
 *
 *   const media = await splitMedia(file, { onProgress, videoElement });
 *   media.video  -> { url, element, duration, width, height }  (für videoanalyse: <video src=url>)
 *   media.audio  -> { samples: Float32Array (mono), sampleRate, duration, method }
 *                   (direkt für worker.js / runWorker)
 *   media.release() -> Object-URL freigeben
 *
 * Audio-Extraktion (effizient -> robust):
 *   1. MP4/MOV/M4V/3GP (AAC/MP3): Container wird selbst gelesen, NUR die Audio-Samples
 *      werden per File.slice geladen und mit WebCodecs (AudioDecoder) dekodiert.
 *      -> kein Laden der kompletten Videodatei, native Samplerate, Abbruch nach MAX_DURATION.
 *   2. Alle anderen Formate (WebM, MKV, OGV, fragmentiertes MP4, ...): decodeAudioData
 *      auf der gesamten Datei (Browser demuxt selbst), native Samplerate aus Header gelesen.
 * Das Video selbst wird NICHT zerlegt: <video> dekodiert nur die Videospur, die
 * Videoanalyse arbeitet per Seek direkt darauf.
 * ============================================================ */
'use strict';

(function () {
  const AUDIO_ONLY_EXT = /\.(mp3|wav|flac|oga|opus|m4a|aac|weba)$/i;
  const MP4_BRANDS = ['ftyp', 'moov', 'mdat', 'free', 'wide', 'skip'];
  const MAX_WHOLE_FILE = 2 * 1024 * 1024 * 1024 - 1; // ArrayBuffer-Grenze

  const fourcc = (u8, o) => String.fromCharCode(u8[o], u8[o + 1], u8[o + 2], u8[o + 3]);
  const sleep0 = () => new Promise((r) => setTimeout(r, 0));

  // ============================================================
  // Samplerate aus Header (damit decodeAudioData nicht resampled)
  // ============================================================
  function sniffWebmRate(b) {
    const end = Math.min(b.length, 1 << 20);
    const MASTER = new Set([0x18538067, 0x1654ae6b, 0xae, 0xe1]); // Segment, Tracks, TrackEntry, Audio
    let rate = null, opus = false, done = false;

    const readVint = (p, keep) => {
      const first = b[p];
      let len = 1, mask = 0x80;
      while (len <= 8 && !(first & mask)) { len++; mask >>= 1; }
      if (len > 8 || p + len > end) return null;
      let v = keep ? first : first & (mask - 1);
      let allOnes = (first & (mask - 1)) === mask - 1;
      for (let i = 1; i < len; i++) { v = v * 256 + b[p + i]; if (b[p + i] !== 0xff) allOnes = false; }
      return { v, len, unknown: !keep && allOnes };
    };

    const walk = (p, e) => {
      while (p < e && !done) {
        const id = readVint(p, true); if (!id) return;
        const sz = readVint(p + id.len, false); if (!sz) return;
        const ds = p + id.len + sz.len;
        const de = sz.unknown ? e : Math.min(ds + sz.v, e);
        if (id.v === 0x1f43b675) { done = true; return; } // Cluster -> Header-Teil vorbei
        if (MASTER.has(id.v)) walk(ds, de);
        else if (id.v === 0x86) {
          if (fourcc(b, ds) === 'A_OP') opus = true;
        } else if (id.v === 0xb5 && (sz.v === 4 || sz.v === 8) && rate === null) {
          const dv = new DataView(b.buffer, b.byteOffset + ds, sz.v);
          rate = sz.v === 4 ? dv.getFloat32(0) : dv.getFloat64(0);
        }
        p = de;
      }
    };
    walk(0, end);
    if (opus) return 48000;
    return rate && rate >= 8000 ? Math.round(rate) : null;
  }

  function sniffSampleRate(buf) {
    const b = new Uint8Array(buf);
    const dv = new DataView(buf);
    const str = (o, n) => String.fromCharCode(...b.subarray(o, o + n));

    if (str(0, 4) === 'RIFF' && str(8, 4) === 'WAVE') {
      let o = 12;
      while (o + 8 <= b.length) {
        const id = str(o, 4), size = dv.getUint32(o + 4, true);
        if (id === 'fmt ') return dv.getUint32(o + 12, true);
        o += 8 + size + (size & 1);
      }
    }
    if (str(0, 4) === 'fLaC' && b.length > 21) return (b[18] << 12) | (b[19] << 4) | (b[20] >> 4);
    if (str(0, 4) === 'OggS') {
      const head = b.subarray(0, Math.min(b.length, 4096));
      const find = (s) => {
        outer: for (let i = 0; i < head.length - s.length; i++) {
          for (let j = 0; j < s.length; j++) if (head[i + j] !== s.charCodeAt(j)) continue outer;
          return i;
        }
        return -1;
      };
      if (find('OpusHead') >= 0) return 48000;
      const v = find('vorbis');
      if (v >= 1) return dv.getUint32(v + 11, true);
    }
    if (b[0] === 0x1a && b[1] === 0x45 && b[2] === 0xdf && b[3] === 0xa3) return sniffWebmRate(b);

    let o = 0;
    if (str(0, 3) === 'ID3') o = 10 + (((b[6] & 0x7f) << 21) | ((b[7] & 0x7f) << 14) | ((b[8] & 0x7f) << 7) | (b[9] & 0x7f));
    for (let i = o; i < Math.min(b.length - 4, o + 65536); i++) {
      if (b[i] === 0xff && (b[i + 1] & 0xe0) === 0xe0) {
        const ver = (b[i + 1] >> 3) & 3, layer = (b[i + 1] >> 1) & 3, idx = (b[i + 2] >> 2) & 3;
        if (ver === 1 || layer === 0 || idx === 3) continue;
        const table = { 3: [44100, 48000, 32000], 2: [22050, 24000, 16000], 0: [11025, 12000, 8000] };
        return table[ver][idx];
      }
    }
    return null;
  }

  // ============================================================
  // Fallback: komplette Datei per decodeAudioData
  // ============================================================
  async function decodeWholeFile(file, hintRate, maxDuration) {
    if (file.size > MAX_WHOLE_FILE) {
      throw new Error('Datei zu groß (>2 GB) für die Audio-Extraktion in diesem Format. MP4/MOV (AAC) wird auch bei großen Dateien unterstützt.');
    }
    const buf = await file.arrayBuffer();
    const sniffed = sniffSampleRate(buf) || hintRate || null;
    const Ctx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
    let ctx;
    try { ctx = new Ctx(1, 1, sniffed || 44100); } catch (e) { ctx = new Ctx(1, 1, 44100); }

    let audioBuffer;
    try {
      audioBuffer = await ctx.decodeAudioData(buf);
    } catch (e) {
      throw new Error('Keine dekodierbare Audiospur gefunden (Codec nicht unterstützt oder kein Ton).');
    }
    const sr = audioBuffer.sampleRate;
    const len = Math.min(audioBuffer.length, Math.trunc(sr * maxDuration));
    const ch = audioBuffer.numberOfChannels;
    const samples = new Float32Array(len);
    for (let c = 0; c < ch; c++) {
      const d = audioBuffer.getChannelData(c);
      for (let i = 0; i < len; i++) samples[i] += d[i];
    }
    if (ch > 1) for (let i = 0; i < len; i++) samples[i] /= ch;
    return { samples, sampleRate: sr, verified: !!sniffed && sniffed === sr, method: 'decodeAudioData' };
  }

  // ============================================================
  // MP4 / MOV: Box-Parser (nur Audiospur)
  // ============================================================
  function* boxes(u8, start = 0, end = u8.length) {
    const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
    let p = start;
    while (p + 8 <= end) {
      let size = dv.getUint32(p), hdr = 8;
      const type = fourcc(u8, p + 4);
      if (size === 1) { if (p + 16 > end) return; size = Number(dv.getBigUint64(p + 8)); hdr = 16; }
      else if (size === 0) size = end - p;
      if (size < hdr || p + size > end) return;
      yield { type, start: p + hdr, end: p + size };
      p += size;
    }
  }
  const child = (u8, box, type) => { for (const b of boxes(u8, box.start, box.end)) if (b.type === type) return b; return null; };

  async function findMoov(file) {
    let pos = 0;
    const size = file.size;
    while (pos + 8 <= size) {
      const h = new Uint8Array(await file.slice(pos, pos + 16).arrayBuffer());
      const dv = new DataView(h.buffer);
      let boxSize = dv.getUint32(0), hdr = 8;
      const type = fourcc(h, 4);
      if (boxSize === 1) { if (h.length < 16) return null; boxSize = Number(dv.getBigUint64(8)); hdr = 16; }
      else if (boxSize === 0) boxSize = size - pos;
      if (boxSize < hdr) return null;
      if (type === 'moov') return new Uint8Array(await file.slice(pos + hdr, pos + boxSize).arrayBuffer());
      pos += boxSize;
    }
    return null;
  }

  function readDesc(u8, p) {
    const tag = u8[p++];
    let len = 0;
    for (let i = 0; i < 4; i++) { const b = u8[p++]; len = (len << 7) | (b & 0x7f); if (!(b & 0x80)) break; }
    return { tag, len, p };
  }
  function parseEsds(u8, s, e) {
    let d = readDesc(u8, s + 4);
    if (d.tag !== 3) return null;
    let p = d.p + 2;
    const fl = u8[p++];
    if (fl & 0x80) p += 2;
    if (fl & 0x40) p += 1 + u8[p];
    if (fl & 0x20) p += 2;
    d = readDesc(u8, p);
    if (d.tag !== 4) return null;
    const oti = u8[d.p];
    p = d.p + 13;
    let asc = null;
    if (p < e) { d = readDesc(u8, p); if (d.tag === 5) asc = u8.slice(d.p, d.p + d.len); }
    return { oti, asc };
  }

  /** -> { status: 'ok'|'noaudio'|'unsupported'|'fragmented', track? } */
  function parseMp4Audio(moov) {
    const dv = new DataView(moov.buffer, moov.byteOffset, moov.byteLength);
    let fragmented = false, sawAudio = false, firstUnsupported = false;

    for (const tb of boxes(moov)) {
      if (tb.type === 'mvex') fragmented = true;
      if (tb.type !== 'trak') continue;
      const mdia = child(moov, tb, 'mdia'); if (!mdia) continue;
      const hdlr = child(moov, mdia, 'hdlr');
      if (!hdlr || fourcc(moov, hdlr.start + 8) !== 'soun') continue;
      sawAudio = true;

      const mdhd = child(moov, mdia, 'mdhd');
      const timescale = dv.getUint32(mdhd.start + (moov[mdhd.start] === 1 ? 20 : 12));
      const stbl = child(moov, child(moov, mdia, 'minf'), 'stbl');
      const stsd = child(moov, stbl, 'stsd');

      // Sample-Entry
      const e = stsd.start + 8;
      const format = fourcc(moov, e + 4);
      const qtVer = dv.getUint16(e + 16);
      const channels = dv.getUint16(e + 24);
      const rate = dv.getUint32(e + 32) >>> 16;
      if (qtVer > 1) { firstUnsupported = true; continue; }
      const entryEnd = e + dv.getUint32(e);
      const kids = e + 36 + (qtVer === 1 ? 16 : 0);

      let codec = null, description = null;
      if (format === 'mp4a') {
        for (const k of boxes(moov, kids, entryEnd)) {
          if (k.type !== 'esds') continue;
          const es = parseEsds(moov, k.start, k.end);
          if (!es) break;
          if (es.oti === 0x40 && es.asc && es.asc.length >= 2) {
            codec = 'mp4a.40.' + (es.asc[0] >> 3);
            description = es.asc;
          } else if (es.oti === 0x69 || es.oti === 0x6b) codec = 'mp3';
        }
      } else if (format === '.mp3' || format === 'mp3 ') codec = 'mp3';
      if (!codec) { firstUnsupported = true; continue; }

      // Sample-Tabellen
      const sttsB = child(moov, stbl, 'stts'), stszB = child(moov, stbl, 'stsz');
      const stscB = child(moov, stbl, 'stsc');
      const stcoB = child(moov, stbl, 'stco') || child(moov, stbl, 'co64');
      if (!sttsB || !stszB || !stscB || !stcoB) { firstUnsupported = true; continue; }
      const is64 = stcoB.type === 'co64';

      const nSamples = dv.getUint32(stszB.start + 8);
      const constSize = dv.getUint32(stszB.start + 4);
      const sizes = new Uint32Array(nSamples);
      for (let i = 0; i < nSamples; i++) sizes[i] = constSize || dv.getUint32(stszB.start + 12 + i * 4);

      const ts = new Float64Array(nSamples + 1), dur = new Float64Array(nSamples);
      {
        const cnt = dv.getUint32(sttsB.start + 4);
        let s = 0, t = 0;
        for (let i = 0; i < cnt && s < nSamples; i++) {
          const c = dv.getUint32(sttsB.start + 8 + i * 8), d = dv.getUint32(sttsB.start + 12 + i * 8);
          for (let j = 0; j < c && s < nSamples; j++, s++) { ts[s] = t; dur[s] = d; t += d; }
        }
        ts[nSamples] = t;
      }

      const nChunks = dv.getUint32(stcoB.start + 4);
      const chunkOffset = (i) => is64
        ? Number(dv.getBigUint64(stcoB.start + 8 + i * 8))
        : dv.getUint32(stcoB.start + 8 + i * 4);
      const stscCnt = dv.getUint32(stscB.start + 4);
      const stsc = [];
      for (let i = 0; i < stscCnt; i++) {
        stsc.push([dv.getUint32(stscB.start + 8 + i * 12), dv.getUint32(stscB.start + 12 + i * 12)]);
      }

      // Chunks -> zusammenhängende "Runs" (weniger Dateizugriffe)
      const runs = [];
      let sample = 0, si = 0;
      for (let c = 0; c < nChunks && sample < nSamples; c++) {
        while (si + 1 < stsc.length && stsc[si + 1][0] <= c + 1) si++;
        const per = stsc[si][1];
        const off = chunkOffset(c);
        let bytes = 0;
        const cnt = Math.min(per, nSamples - sample);
        for (let k = 0; k < cnt; k++) bytes += sizes[sample + k];
        const last = runs[runs.length - 1];
        if (last && last.offset + last.size === off) { last.size += bytes; last.count += cnt; }
        else runs.push({ offset: off, size: bytes, firstSample: sample, count: cnt });
        sample += cnt;
      }

      return {
        status: nSamples > 0 ? 'ok' : 'unsupported',
        track: { codec, description, sampleRate: rate, channels: channels || 2, timescale, sizes, ts, dur, runs, nSamples }
      };
    }
    if (!sawAudio) return { status: fragmented ? 'fragmented' : 'noaudio' };
    return { status: fragmented ? 'fragmented' : 'unsupported' };
  }

  async function decodeMp4Audio(file, t, maxDuration, onProgress) {
    const cfg = { codec: t.codec, sampleRate: t.sampleRate, numberOfChannels: t.channels };
    if (t.description) cfg.description = t.description;
    const sup = await AudioDecoder.isConfigSupported(cfg);
    if (!sup.supported) throw new Error('Audio-Codec ' + t.codec + ' wird von WebCodecs nicht unterstützt');

    const parts = [];
    let total = 0, outRate = 0, error = null;
    const decoder = new AudioDecoder({
      output: (ad) => {
        const n = ad.numberOfFrames, ch = ad.numberOfChannels;
        outRate = ad.sampleRate;
        const mono = new Float32Array(n), tmp = new Float32Array(n);
        for (let c = 0; c < ch; c++) {
          ad.copyTo(tmp, { planeIndex: c, format: 'f32-planar' });
          for (let i = 0; i < n; i++) mono[i] += tmp[i];
        }
        if (ch > 1) for (let i = 0; i < n; i++) mono[i] /= ch;
        ad.close();
        parts.push(mono);
        total += n;
      },
      error: (e) => { error = e; }
    });
    decoder.configure(cfg);

    const limitTicks = maxDuration * t.timescale;
    const BATCH = 24;
    const runs = t.runs;
    for (let r = 0; r < runs.length && !error; r += BATCH) {
      if (t.ts[runs[r].firstSample] > limitTicks) break;
      const group = runs.slice(r, r + BATCH);
      const bufs = await Promise.all(group.map(async (run) =>
        new Uint8Array(await file.slice(run.offset, run.offset + run.size).arrayBuffer())));
      for (let g = 0; g < group.length; g++) {
        const run = group[g], buf = bufs[g];
        let o = 0;
        for (let k = 0; k < run.count; k++) {
          const s = run.firstSample + k, sz = t.sizes[s];
          if (o + sz > buf.length) break;
          decoder.decode(new EncodedAudioChunk({
            type: 'key',
            timestamp: Math.round((t.ts[s] * 1e6) / t.timescale),
            duration: Math.round((t.dur[s] * 1e6) / t.timescale),
            data: buf.subarray(o, o + sz)
          }));
          o += sz;
        }
        while (decoder.decodeQueueSize > 64 && !error) await sleep0();
      }
      onProgress(Math.min(1, t.ts[group[group.length - 1].firstSample] / limitTicks));
    }
    if (!error) await decoder.flush();
    if (decoder.state !== 'closed') decoder.close();
    if (error) throw new Error('AudioDecoder: ' + (error.message || error));
    if (!total) throw new Error('Keine Audiodaten dekodiert');

    const len = Math.min(total, Math.trunc(outRate * maxDuration));
    const samples = new Float32Array(len);
    let pos = 0;
    for (const p of parts) {
      if (pos >= len) break;
      const n = Math.min(p.length, len - pos);
      samples.set(n === p.length ? p : p.subarray(0, n), pos);
      pos += n;
    }
    return { samples, sampleRate: outRate, verified: true, method: 'webcodecs-mp4' };
  }

  // ============================================================
  // Audio-Extraktion (Strategie wählen)
  // ============================================================
  async function extractAudio(file, maxDuration, onProgress, warnings) {
    const head = new Uint8Array(await file.slice(0, 12).arrayBuffer());
    const isMp4 = head.length >= 8 && MP4_BRANDS.includes(fourcc(head, 4));
    let hint = null;

    if (isMp4) {
      try {
        const moov = await findMoov(file);
        if (moov) {
          const res = parseMp4Audio(moov);
          if (res.status === 'noaudio') return null;               // Video ohne Tonspur
          if (res.status === 'ok') {
            hint = res.track.sampleRate;
            if (typeof AudioDecoder !== 'undefined' && typeof EncodedAudioChunk !== 'undefined') {
              try { return await decodeMp4Audio(file, res.track, maxDuration, onProgress); }
              catch (e) { warnings.push('WebCodecs-Pfad fehlgeschlagen (' + e.message + '), nutze Fallback.'); }
            } else warnings.push('WebCodecs nicht verfügbar, nutze Fallback (lädt die ganze Datei).');
          }
        }
      } catch (e) {
        warnings.push('MP4-Parser fehlgeschlagen (' + e.message + '), nutze Fallback.');
      }
    }
    onProgress(0);
    return await decodeWholeFile(file, hint, maxDuration);
  }

  // ============================================================
  // Video-Teil
  // ============================================================
  function openVideo(file, videoElement) {
    return new Promise((resolve) => {
      const url = URL.createObjectURL(file);
      const el = videoElement || document.createElement('video');
      el.preload = 'metadata';
      el.muted = true;
      el.playsInline = true;
      let settled = false;
      const finish = (ok, why) => {
        if (settled) return;
        settled = true;
        clearTimeout(tid);
        el.removeEventListener('loadedmetadata', onMeta);
        el.removeEventListener('error', onErr);
        if (ok && el.videoWidth > 0) {
          resolve({ ok: true, video: { url, element: el, duration: el.duration, width: el.videoWidth, height: el.videoHeight } });
        } else {
          if (!videoElement) el.removeAttribute('src');
          URL.revokeObjectURL(url);
          resolve({ ok: false, why: why || 'Keine Videospur' });
        }
      };
      const onMeta = () => finish(true);
      const onErr = () => finish(false, 'Browser kann das Video nicht abspielen (Codec/Container nicht unterstützt)');
      const tid = setTimeout(() => finish(false, 'Zeitüberschreitung beim Laden der Video-Metadaten'), 15000);
      el.addEventListener('loadedmetadata', onMeta);
      el.addEventListener('error', onErr);
      el.src = url;
    });
  }

  // ============================================================
  // Öffentliche API
  // ============================================================
  async function splitMedia(file, options = {}) {
    const o = Object.assign({ maxAudioDuration: 1500, onProgress: () => {}, videoElement: null }, options);
    const warnings = [];
    const audioOnly = (file.type || '').startsWith('audio/') || (!file.type && AUDIO_ONLY_EXT.test(file.name));

    const videoP = audioOnly
      ? Promise.resolve({ ok: false, why: 'Audiodatei' })
      : openVideo(file, o.videoElement);
    const audioP = extractAudio(file, o.maxAudioDuration, (v) => o.onProgress(v, 'Audio extrahieren'), warnings)
      .catch((e) => { warnings.push(e.message); return null; });

    const [v, a] = await Promise.all([videoP, audioP]);
    if (!v.ok && !audioOnly) warnings.push('Video: ' + v.why);
    if (!v.ok && !a) throw new Error('Datei konnte weder als Video noch als Audio gelesen werden. ' + warnings.join(' '));

    const audio = a && {
      samples: a.samples,
      sampleRate: a.sampleRate,
      duration: a.samples.length / a.sampleRate,
      method: a.method,
      nativeSampleRateVerified: a.verified
    };
    if (!audio && !audioOnly) warnings.push('Keine Audiospur gefunden.');

    const video = v.ok ? v.video : null;
    return {
      kind: video ? 'video' : 'audio',
      file,
      video,
      audio,
      warnings,
      release() { if (video) { URL.revokeObjectURL(video.url); } }
    };
  }

  window.splitMedia = splitMedia;
})();

/* ============================================================
 * Videoanalyse (Port aus videoanalyseindex.html) – ohne DOM-Abhängigkeit
 *   const r = await analyzeVideoElement(videoEl, { config, signal }, onProgress(0..1))
 * ============================================================ */
(function () {
  'use strict';

  const DEFAULTS = {
    ANALYSIS_WIDTH: 160, ANALYSIS_HEIGHT: 90, SAMPLE_FPS: 10,
    GRID_COLS: 16, GRID_ROWS: 9,
    MOTION_MIN_DELTA_TIME: 1 / 120, SEEK_TIMEOUT_MS: 3000, STUCK_SAMPLES_LIMIT: 2,
    BLANK_LUMINANCE_THRESHOLD: 10, BLANK_DIFF_THRESHOLD: 2, BLANK_STREAK_LIMIT: 50,
    CUT_THRESHOLD: 36, CUT_MIN_DISTANCE: 0.35,
    // Scores (aus videoanalyse.html)
    SCORE_COLOR_MAX: 40,          // Farbkontrast-Score = roh / 40 * 100
    SCORE_MOTION_MAX: 110,        // Bewegungs-Score    = roh / 110 * 100
    SCORE_CUT_REF_SECONDS: 7,     // Schnitt-Score      = (1 - Zeit/7) * 100 + 25
    SCORE_CUT_BONUS: 25,
    // Video-Score = gewichteter Mittelwert der (auf 0-100 begrenzten) Einzelscores
    W_COLOR: 1, W_MOTION: 1, W_CUT: 1
  };

  // ---------- Farbraum ----------
  const srgbToLinear = (v) => { const n = v / 255; return n <= 0.04045 ? n / 12.92 : Math.pow((n + 0.055) / 1.055, 2.4); };
  const LAB_EPS = 216 / 24389, LAB_KAPPA = 24389 / 27;
  const labF = (t) => (t > LAB_EPS ? Math.cbrt(t) : (LAB_KAPPA * t + 16) / 116);
  function rgbToLab(r, g, b) {
    const lr = srgbToLinear(r), lg = srgbToLinear(g), lb = srgbToLinear(b);
    const x = lr * 0.4124564 + lg * 0.3575761 + lb * 0.1804375;
    const y = lr * 0.2126729 + lg * 0.7151522 + lb * 0.0721750;
    const z = lr * 0.0193339 + lg * 0.1191920 + lb * 0.9503041;
    const fx = labF(x / 0.95047), fy = labF(y), fz = labF(z / 1.08883);
    return { l: 116 * fy - 16, a: 500 * (fx - fy), b: 200 * (fy - fz) };
  }
  function avgPairwiseDeltaE(L, A, B, count) {
    if (count < 2) return 0;
    let sum = 0, pairs = 0;
    for (let i = 0; i < count; i++) {
      for (let j = i + 1; j < count; j++) {
        const dL = L[i] - L[j], dA = A[i] - A[j], dB = B[i] - B[j];
        sum += Math.sqrt(dL * dL + dA * dA + dB * dB);
        pairs++;
      }
    }
    return pairs > 0 ? sum / pairs : 0;
  }

  // ---------- Frame-Analyse (Puffer pro Analyse, keine Globals) ----------
  function createFrameAnalyzer(cfg) {
    const W = cfg.ANALYSIS_WIDTH, H = cfg.ANALYSIS_HEIGHT, COLS = cfg.GRID_COLS, ROWS = cfg.GRID_ROWS, N = COLS * ROWS;
    const sumR = new Float64Array(N), sumG = new Float64Array(N), sumB = new Float64Array(N), cnt = new Int32Array(N);
    const LL = new Float64Array(N), LA = new Float64Array(N), LB = new Float64Array(N);
    const bw = W / COLS, bh = H / ROWS;

    return function analyzeFrame(data, prev, deltaTime) {
      const hasPrev = !!prev;
      sumR.fill(0); sumG.fill(0); sumB.fill(0); cnt.fill(0);
      let lumSum = 0, diffSum = 0, count = 0;

      for (let i = 0; i < data.length; i += 4) {
        const r = data[i], g = data[i + 1], b = data[i + 2];
        lumSum += 0.2126 * r + 0.7152 * g + 0.0722 * b;
        count++;
        const px = i / 4, x = px % W, y = Math.floor(px / W);
        const bi = Math.min(ROWS - 1, Math.floor(y / bh)) * COLS + Math.min(COLS - 1, Math.floor(x / bw));
        sumR[bi] += r; sumG[bi] += g; sumB[bi] += b; cnt[bi]++;
        if (hasPrev) diffSum += (Math.abs(r - prev[i]) + Math.abs(g - prev[i + 1]) + Math.abs(b - prev[i + 2])) / 3;
      }
      const meanLum = lumSum / count;
      for (let k = 0; k < N; k++) {
        const c = cnt[k] || 1;
        const lab = rgbToLab(sumR[k] / c, sumG[k] / c, sumB[k] / c);
        LL[k] = lab.l; LA[k] = lab.a; LB[k] = lab.b;
      }
      const colorContrastRaw = avgPairwiseDeltaE(LL, LA, LB, N);
      if (!hasPrev) return { colorContrastRaw, motionRaw: 0, cutDiffRaw: 0, meanLuminance: meanLum, pixelDiff: 0 };
      const avgDiff = diffSum / count;
      const motionRaw = avgDiff / Math.max(deltaTime || 0, cfg.MOTION_MIN_DELTA_TIME);
      return { colorContrastRaw, motionRaw, cutDiffRaw: avgDiff, meanLuminance: meanLum, pixelDiff: avgDiff };
    };
  }

  // ---------- Hilfen ----------
  const abortError = () => new DOMException('Analyse abgebrochen', 'AbortError');
  const average = (a) => (a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0);
  const nextFrame = () => new Promise((res) => {
    if ('requestIdleCallback' in window) requestIdleCallback(() => res(), { timeout: 50 });
    else setTimeout(res, 0);
  });

  function effectiveDuration(video) {
    let d = video.duration;
    try {
      if (video.seekable && video.seekable.length > 0) {
        const end = video.seekable.end(video.seekable.length - 1);
        if (isFinite(end) && end > 0 && end < d) d = end;
      }
    } catch (e) { /* ignorieren */ }
    return d;
  }

  function seekVideo(video, time, timeoutMs) {
    return new Promise((resolve, reject) => {
      let settled = false;
      const cleanup = () => {
        clearTimeout(tid);
        video.removeEventListener('seeked', onSeeked);
        video.removeEventListener('error', onError);
      };
      const onSeeked = () => { if (settled) return; settled = true; cleanup(); resolve(); };
      const onError = () => { if (settled) return; settled = true; cleanup(); reject(new Error('Video konnte nicht gelesen werden.')); };
      const tid = setTimeout(() => {
        if (settled) return; settled = true; cleanup(); reject(new Error('Zeitüberschreitung beim Springen im Video.'));
      }, timeoutMs);
      video.addEventListener('seeked', onSeeked, { once: true });
      video.addEventListener('error', onError, { once: true });
      video.currentTime = time;
    });
  }

  function detectCuts(data, cfg) {
    const cuts = [];
    let last = -Infinity;
    for (let i = 1; i < data.length - 1; i++) {
      const c = data[i], p = data[i - 1], n = data[i + 1];
      const isPeak = c.cutDiffRaw >= p.cutDiffRaw && c.cutDiffRaw >= n.cutDiffRaw;
      if (isPeak && c.cutDiffRaw >= cfg.CUT_THRESHOLD && c.time - last >= cfg.CUT_MIN_DISTANCE) {
        cuts.push({ time: c.time, diffRaw: c.cutDiffRaw });
        last = c.time;
      }
    }
    return cuts;
  }

  // ---------- Hauptfunktion ----------
  async function analyzeVideoElement(video, options = {}, onProgress = () => {}) {
    const cfg = Object.assign({}, DEFAULTS, options.config || {});
    const signal = options.signal;
    if (!video.duration || !isFinite(video.duration)) throw new Error('Videodauer unbekannt (Metadaten nicht geladen).');

    const W = cfg.ANALYSIS_WIDTH, H = cfg.ANALYSIS_HEIGHT;
    const canvas = document.createElement('canvas');
    canvas.width = W; canvas.height = H;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const bufs = [new Uint8ClampedArray(W * H * 4), new Uint8ClampedArray(W * H * 4)];
    const analyzeFrame = createFrameAnalyzer(cfg);

    const draw = () => {
      const vw = video.videoWidth || W, vh = video.videoHeight || H;
      const ca = W / H, va = vw / vh;
      let dw, dh;
      if (va > ca) { dw = W; dh = W / va; } else { dh = H; dw = H * va; }
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, W, H);
      ctx.drawImage(video, (W - dw) / 2, (H - dh) / 2, dw, dh);
    };

    const results = [], warnings = [];
    const duration = effectiveDuration(video);
    const totalSamples = Math.max(1, Math.ceil(duration * cfg.SAMPLE_FPS));
    let previousFrame = null, bufIdx = 0, lastActual = -1, stuck = 0, blankStreak = 0;

    for (let i = 0; i < totalSamples; i++) {
      if (signal && signal.aborted) throw abortError();
      const target = i / cfg.SAMPLE_FPS;
      if (target >= duration) break;

      await seekVideo(video, target, cfg.SEEK_TIMEOUT_MS);
      const time = video.currentTime;

      if (time <= lastActual + 0.001) {
        stuck++;
        if (stuck >= cfg.STUCK_SAMPLES_LIMIT) {
          warnings.push(`Video-Seek hängt bei ${time.toFixed(1)} s – Analyse dort beendet.`);
          break;
        }
      } else stuck = 0;
      lastActual = time;

      draw();
      const imageData = ctx.getImageData(0, 0, W, H);
      const prevTime = results.length ? results[results.length - 1].time : null;
      const frame = analyzeFrame(imageData.data, previousFrame, prevTime === null ? null : time - prevTime);
      results.push({ time, colorContrastRaw: frame.colorContrastRaw, motionRaw: frame.motionRaw, cutDiffRaw: frame.cutDiffRaw });

      const isBlank = previousFrame !== null &&
        frame.meanLuminance < cfg.BLANK_LUMINANCE_THRESHOLD && frame.pixelDiff < cfg.BLANK_DIFF_THRESHOLD;
      blankStreak = isBlank ? blankStreak + 1 : 0;

      const buf = bufs[bufIdx];
      buf.set(imageData.data);
      previousFrame = buf;
      bufIdx = 1 - bufIdx;

      if (blankStreak >= cfg.BLANK_STREAK_LIMIT) {
        results.splice(results.length - blankStreak, blankStreak);
        warnings.push('Leeres/schwarzes Videoende erkannt und abgeschnitten.');
        break;
      }
      onProgress(Math.min(1, (i + 1) / totalSamples));
      if (i % 4 === 0) await nextFrame();
    }

    if (!results.length) throw new Error('Keine Videoergebnisse.');
    const cuts = detectCuts(results, cfg);
    const dur = results[results.length - 1].time;
    const colorRaw = average(results.map((r) => r.colorContrastRaw));
    const motionRaw = average(results.map((r) => r.motionRaw));
    const secPerCut = cuts.length > 0 ? dur / cuts.length : null;

    // Einzelscores (wie in videoanalyse.html, unbegrenzt)
    const scoreColor = (colorRaw / cfg.SCORE_COLOR_MAX) * 100;
    const scoreMotion = (motionRaw / cfg.SCORE_MOTION_MAX) * 100;
    const scoreCut = secPerCut === null ? null : (1 - secPerCut / cfg.SCORE_CUT_REF_SECONDS) * 100 + cfg.SCORE_CUT_BONUS;
    // Video-Score 0-100: jeder Einzelscore wird auf 0-100 begrenzt; ohne Schnitte zaehlt der Schnitt-Score als 0
    const clamp = (v) => Math.min(100, Math.max(0, v));
    const wSum = cfg.W_COLOR + cfg.W_MOTION + cfg.W_CUT;
    const videoScore = (cfg.W_COLOR * clamp(scoreColor) + cfg.W_MOTION * clamp(scoreMotion) +
      cfg.W_CUT * clamp(scoreCut === null ? 0 : scoreCut)) / wSum;
    onProgress(1);
    return {
      color_contrast_raw: colorRaw,
      motion_raw: motionRaw,
      cut_count: cuts.length,
      seconds_per_cut: secPerCut,
      scores: {
        color_contrast: scoreColor,
        motion: scoreMotion,
        cut: scoreCut,
        video_score: videoScore,
        weights: { color: cfg.W_COLOR, motion: cfg.W_MOTION, cut: cfg.W_CUT },
        cut_counted_as_zero: scoreCut === null
      },
      duration_seconds: dur,
      sample_count: results.length,
      cuts,
      warnings
    };
  }

  window.analyzeVideoElement = analyzeVideoElement;
})();

/* ============================================================
 * Orchestrierung + Widget
 *   const res = await analyzeMedia(file, { onProgress, exportJson, workerUrl, signal })
 *   mountMediaAnalyzer('#ziel', { onResult })     oder  <div data-media-analyzer></div>
 * Ablauf: splitten -> Video analysieren -> Audio analysieren (nacheinander)
 * ============================================================ */
(function () {
  'use strict';

  const SCRIPT_SRC = (document.currentScript && document.currentScript.src) || null;
  const defaultWorkerUrl = () => {
    try { return new URL('worker.js', SCRIPT_SRC || location.href).href; } catch (e) { return 'worker.js'; }
  };
  const abortError = () => new DOMException('Analyse abgebrochen', 'AbortError');

  function runWorker(samples, sampleRate, options, workerUrl, signal, onProgress) {
    return new Promise((resolve, reject) => {
      if (signal && signal.aborted) return reject(abortError());
      const worker = new Worker(workerUrl);
      const onAbort = () => { worker.terminate(); reject(abortError()); };
      if (signal) signal.addEventListener('abort', onAbort, { once: true });
      const done = (fn, v) => { if (signal) signal.removeEventListener('abort', onAbort); worker.terminate(); fn(v); };
      worker.onmessage = (e) => {
        const m = e.data;
        if (m.type === 'progress') onProgress(m.value, m.stage);
        else if (m.type === 'result') done(resolve, m.result);
        else if (m.type === 'error') done(reject, new Error(m.message));
      };
      worker.onerror = (e) => done(reject, new Error(e.message || 'Worker-Fehler (worker.js nicht gefunden? -> option workerUrl)'));
      worker.postMessage({ type: 'analyze', samples, sampleRate, options }, [samples.buffer]);
    });
  }

  /**
   * Datei -> { kind, split, video, audio, errors, warnings, elapsed_ms }
   * onProgress({ stage: 'split'|'video'|'audio'|'done', progress: 0..1, label })
   */
  async function analyzeMedia(file, options = {}) {
    const o = Object.assign({
      maxAudioDuration: 1500, exportJson: false, workerUrl: defaultWorkerUrl(),
      onProgress: () => {}, video: {}, signal: null, keepMedia: false
    }, options);
    const t0 = performance.now();
    const timings = {};
    let tLap = t0;
    const lap = (name) => { const n = performance.now(); timings[name] = Math.round(n - tLap); tLap = n; };
    const errors = {}, warnings = [];
    const report = (stage, progress, label) => o.onProgress({ stage, progress: Math.min(1, progress), label });
    const checkAbort = () => { if (o.signal && o.signal.aborted) throw abortError(); };

    // 1) Splitten
    report('split', 0, 'Datei wird getrennt (Video / Audio)');
    const media = await splitMedia(file, {
      maxAudioDuration: o.maxAudioDuration,
      onProgress: (v) => report('split', v * 0.1, 'Audio wird extrahiert')
    });
    warnings.push(...media.warnings);
    lap('split_ms');
    const stages = (media.video ? 1 : 0) + (media.audio ? 1 : 0);
    const span = 0.9 / Math.max(1, stages);
    let cursor = 0.1;
    let videoRes = null, audioRes = null;

    try {
      // 2) Video
      if (media.video) {
        checkAbort();
        try {
          videoRes = await analyzeVideoElement(media.video.element, { config: o.video.config, signal: o.signal },
            (v) => report('video', cursor + v * span, 'Video wird analysiert'));
          warnings.push(...videoRes.warnings);
        } catch (e) {
          if (e.name === 'AbortError') throw e;
          errors.video = e.message;
        }
        cursor += span;
        lap('video_ms');
      }

      // 3) Audio
      if (media.audio) {
        checkAbort();
        const { samples, sampleRate, method, nativeSampleRateVerified } = media.audio;
        try {
          audioRes = await runWorker(samples, sampleRate, { exportJson: o.exportJson }, o.workerUrl, o.signal,
            (v, stage) => report('audio', cursor + v * span, 'Audio: ' + stage));
          audioRes.meta.native_samplerate_verified = nativeSampleRateVerified;
          audioRes.meta.audio_extraction = method;
        } catch (e) {
          if (e.name === 'AbortError') throw e;
          errors.audio = e.message;
        }
        cursor += span;
        lap('audio_ms');
      }
    } finally {
      if (!o.keepMedia) media.release();
    }

    report('done', 1, 'Fertig');
    return {
      file: { name: file.name, size: file.size, type: file.type },
      kind: media.kind,
      split: {
        video: media.video && { duration: media.video.duration, width: media.video.width, height: media.video.height },
        audio: audioRes ? { sampleRate: audioRes.meta.sample_rate, duration: audioRes.meta.duration_seconds,
          method: audioRes.meta.audio_extraction, nativeSampleRateVerified: audioRes.meta.native_samplerate_verified } : null
      },
      video: videoRes,
      audio: audioRes,
      errors, warnings,
      elapsed_ms: Math.round(performance.now() - t0),
      timings: Object.assign(timings, { total_ms: Math.round(performance.now() - t0) }),
      media: o.keepMedia ? media : undefined
    };
  }

  // ============================================================
  // Formatierung
  // ============================================================
  const fmt = (v, d = 2) => (typeof v === 'number' && isFinite(v) ? v.toFixed(d).replace('.', ',') : '–');

  function audioText(out) {
    const lw = Math.max(...out.zeilen.filter(([l]) => l).map(([l]) => l.length));
    let s = '';
    for (const [l, v] of out.zeilen) s += l ? `${l.padEnd(lw)} : ${v}\n` : '\n';
    s += `\n${'>>> GESAMT-AUFMERKSAMKEITSSCORE (0-100)'.padEnd(lw)} : ${out.aufmerksamkeits_score}\n`;
    return s;
  }

  function resultText(res) {
    let s = '=============== VIDEO-ANALYSE ===============\n\n';
    if (res.video) {
      const v = res.video, sc = v.scores;
      s += `Farbkontrast (roh) : ${fmt(v.color_contrast_raw)}\n`;
      s += `Farbkontrast-Score : ${fmt(sc.color_contrast)}\n\n`;
      s += `Bewegung (roh)     : ${fmt(v.motion_raw)}\n`;
      s += `Bewegungs-Score    : ${fmt(sc.motion)}\n\n`;
      s += `Zeit pro Schnitt   : ${v.seconds_per_cut === null ? '–' : fmt(v.seconds_per_cut) + ' s'}  (${v.cut_count} Schnitte)\n`;
      s += `Schnitt-Score (+25): ${fmt(sc.cut)}\n\n`;
      s += `>>> VIDEO-SCORE (0-100, Gewichtung ${sc.weights.color}:${sc.weights.motion}:${sc.weights.cut}) : ${fmt(sc.video_score)}\n`;
      s += '    (Einzelscores fuer den Video-Score auf 0-100 begrenzt' + (sc.cut_counted_as_zero ? '; ohne Schnitte: Schnitt-Score = 0' : '') + ')\n';
    } else s += res.errors.video ? `Fehler: ${res.errors.video}\n` : 'Kein Video vorhanden.\n';

    s += '\n=============== AUDIO-ANALYSE ===============\n\n';
    if (res.audio) s += audioText(res.audio.output);
    else s += res.errors.audio ? `Fehler: ${res.errors.audio}\n` : 'Keine Audiospur gefunden.\n';

    if (res.timings) {
      const t = res.timings;
      const sec = (ms) => (ms / 1000).toFixed(1).replace('.', ',') + ' s';
      s += '\n=============== ZEIT ===============\n\n';
      s += `Trennen (Video/Audio) : ${sec(t.split_ms)}\n`;
      if (t.video_ms != null) s += `Videoanalyse          : ${sec(t.video_ms)}\n`;
      if (t.audio_ms != null) s += `Audioanalyse          : ${sec(t.audio_ms)}\n`;
      s += `GESAMT                : ${sec(t.total_ms)}\n`;
    }

    if (res.warnings.length) s += '\n⚠ ' + res.warnings.join('\n⚠ ') + '\n';
    return s;
  }

  // ============================================================
  // Widget
  // ============================================================
  const STYLE = `
.ma-root{font-family:system-ui,sans-serif;max-width:760px}
.ma-root .ma-row,.ma-root .ma-opt{display:block;margin:8px 0}
.ma-root button{padding:8px 14px;margin-right:6px;font-size:14px}
.ma-root progress{width:100%;height:18px}
.ma-root pre{background:rgba(127,127,127,.15);padding:8px;overflow-x:auto;font-size:12px}
.ma-root .ma-status{white-space:pre-line;opacity:.8}
.ma-root .ma-dl a{display:block}`;

  const ACCEPT = 'video/*,.mp4,.m4v,.mov,.webm,.mkv,.ogv,.3gp';
  const STAGE_LABEL = { split: 'Trenne Video/Audio', video: 'Analysiere Video', audio: 'Analysiere Audio', done: 'Fertig' };

  function mountMediaAnalyzer(target, options = {}) {
    const root = typeof target === 'string' ? document.querySelector(target) : target;
    if (!root) throw new Error('mountMediaAnalyzer: Ziel-Element nicht gefunden');
    if (!document.getElementById('ma-style')) {
      const st = document.createElement('style'); st.id = 'ma-style'; st.textContent = STYLE; document.head.appendChild(st);
    }
    root.classList.add('ma-root');
    root.innerHTML = `
      <input type="file" class="ma-file" accept="${ACCEPT}">
      <label class="ma-opt"><input type="checkbox" class="ma-export"> onset.json / pitch.json exportieren</label>
      <div class="ma-row"><button class="ma-run">Analysieren</button><button class="ma-cancel" disabled>Abbrechen</button></div>
      <p class="ma-status">Bereit.</p>
      <progress class="ma-bar" value="0" max="1"></progress>
      <p>Laufzeit: <span class="ma-timer">0,0 s</span></p>
      <pre class="ma-out"></pre>
      <div class="ma-dl"></div>
      <details><summary>Rohes Ergebnis-JSON</summary><pre class="ma-json"></pre></details>`;
    const q = (c) => root.querySelector(c);
    const timerEl = q('.ma-timer');
    const [fileEl, runBtn, cancelBtn, statusEl, bar, outEl, dlEl, jsonEl, exportEl] =
      ['.ma-file', '.ma-run', '.ma-cancel', '.ma-status', '.ma-bar', '.ma-out', '.ma-dl', '.ma-json', '.ma-export'].map(q);
    let ac = null;

    const addDownload = (name, text) => {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
      a.download = name; a.textContent = '⬇ ' + name;
      dlEl.appendChild(a);
    };

    async function analyze(file) {
      file = file || fileEl.files[0];
      if (!file) { alert('Bitte zuerst eine Videodatei wählen.'); return null; }
      runBtn.disabled = true; cancelBtn.disabled = false;
      outEl.textContent = ''; jsonEl.textContent = ''; dlEl.innerHTML = '';
      bar.value = 0; statusEl.textContent = 'Starte …';
      ac = new AbortController();
      const tStart = performance.now();
      const tick = setInterval(() => {
        timerEl.textContent = ((performance.now() - tStart) / 1000).toFixed(1).replace('.', ',') + ' s';
      }, 100);
      try {
        const res = await analyzeMedia(file, Object.assign({}, options, {
          exportJson: exportEl.checked, signal: ac.signal,
          onProgress: (p) => {
            bar.value = p.progress;
            statusEl.textContent = `${STAGE_LABEL[p.stage] || p.stage} – ${Math.round(p.progress * 100)} %`;
            if (options.onProgress) options.onProgress(p);
          }
        }));
        statusEl.textContent = `Fertig in ${(res.elapsed_ms / 1000).toFixed(1)} s`;
        outEl.textContent = resultText(res);

        const { json_export, ...audioRest } = res.audio || {};
        jsonEl.textContent = JSON.stringify(Object.assign({}, res, { audio: res.audio ? audioRest : null }), null, 2);
        addDownload('analyse-ergebnis.json', jsonEl.textContent);
        if (json_export) for (const [n, t] of Object.entries(json_export)) addDownload(n, t);

        root.dispatchEvent(new CustomEvent('mediaanalyse:result', { detail: res, bubbles: true }));
        if (options.onResult) options.onResult(res);
        return res;
      } catch (err) {
        statusEl.textContent = err.name === 'AbortError' ? 'Abgebrochen.' : 'Fehler: ' + err.message;
        return null;
      } finally {
        runBtn.disabled = false; cancelBtn.disabled = true; ac = null;
        clearInterval(tick);
        timerEl.textContent = ((performance.now() - tStart) / 1000).toFixed(1).replace('.', ',') + ' s (gesamt)';
      }
    }

    runBtn.addEventListener('click', () => analyze());
    cancelBtn.addEventListener('click', () => ac && ac.abort());
    return { analyze, cancel: () => ac && ac.abort(), destroy() { root.innerHTML = ''; root.classList.remove('ma-root'); } };
  }

  window.MediaAnalyzer = { analyzeMedia, mountMediaAnalyzer, splitMedia, analyzeVideoElement, resultText };
  window.analyzeMedia = analyzeMedia;
  window.mountMediaAnalyzer = mountMediaAnalyzer;

  // Auto-Mount: <div data-media-analyzer></div>
  const auto = () => document.querySelectorAll('[data-media-analyzer]').forEach((el) => mountMediaAnalyzer(el));
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', auto); else auto();
})();
