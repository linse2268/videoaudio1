/* ============================================================
 * worker.js – Port von audioanalyse.py (librosa) nach Vanilla JS
 * Keine externen Bibliotheken: YIN, RMS, STFT/Mel, Onset-Detection
 * sind nach dem Quelltext von librosa 0.11.0 nachgebaut.
 * ============================================================ */
'use strict';

// ============================================================
// KONSTANTEN / GEWICHTE (identisch zum Python-Skript)
// ============================================================
const BORDER_LOW = 155.0;
const BORDER_HIGH = 165.0;
const MAX_VCPM = 2000;
const HALBTON_SCHWELLE = 1.0;
const MAX_EPM = 300;

const W_VCPM = 0.7;
const W_RMS_VC = 0.3;
const W_EPM = 0.7;
const W_RMS = 0.3;

const W_STIMME_VALENZ = 2 / 12;
const W_ACS = 4 / 12;
const W_VCS = 6 / 12;

const Config = {
  MAX_DURATION: 1500,
  HOP_LENGTH: 512,
  FMIN: 80,
  FMAX: 900,
  // librosa-Defaults (nicht im Python-Skript gesetzt, daher Defaults):
  FRAME_LENGTH: 2048,      // yin / rms / stft
  N_MELS: 128,             // onset_strength -> melspectrogram
  MEL_FMAX: null,          // librosa 0.11: onset_strength setzt fmax=0.5*sr (null -> sr/2)
  TROUGH_THRESHOLD: 0.1,   // yin
  PAD_MODE: 'constant'     // librosa >= 0.10 ('reflect' fuer librosa < 0.10)
};

const TINY_F32 = 1.1754943508222875e-38; // np.finfo(float32).tiny

// ============================================================
// HILFSFUNKTIONEN
// ============================================================

/** Python round(x, nd): exakte Dezimalrundung, Ties -> gerade Ziffer (banker's rounding). */
function pyRound(x, nd = 0) {
  if (!Number.isFinite(x)) return x;
  const p = 10 ** nd;
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const sc = ax * p;
  const fl = Math.floor(sc);
  const fr = sc - fl;
  let k;
  if (Math.abs(fr - 0.5) > 1e-4 && sc < 1e12) {
    k = fr > 0.5 ? fl + 1 : fl;
  } else {
    // Grenzfall: exakte Dezimalentwicklung des Doubles auswerten
    const s = ax.toFixed(100);
    const dot = s.indexOf('.');
    const kk = Number(s.slice(0, dot) + s.slice(dot + 1, dot + 1 + nd));
    const rest = s.slice(dot + 1 + nd);
    if (/^50*$/.test(rest)) k = kk % 2 === 0 ? kk : kk + 1;
    else k = rest.charCodeAt(0) >= 53 ? kk + 1 : kk;
  }
  return sign * (k / p);
}

function mean(a) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  return s / a.length;
}

function maxOf(a) {
  let m = -Infinity;
  for (let i = 0; i < a.length; i++) if (a[i] > m) m = a[i];
  return m;
}

let _lastPost = 0;
function progress(value, stage) {
  const now = performance.now();
  if (value >= 1 || now - _lastPost > 40) {
    _lastPost = now;
    self.postMessage({ type: 'progress', value, stage });
  }
}

// ============================================================
// FFT (radix-2, in-place, komplex, Vorwaerts e^{-i...})
// ============================================================
const _fftCache = {};
function makeFFT(N) {
  if (_fftCache[N]) return _fftCache[N];
  const levels = Math.round(Math.log2(N));
  const rev = new Uint32Array(N);
  for (let i = 0; i < N; i++) {
    let r = 0, x = i;
    for (let b = 0; b < levels; b++) { r = (r << 1) | (x & 1); x >>= 1; }
    rev[i] = r;
  }
  const cs = new Float64Array(N / 2), sn = new Float64Array(N / 2);
  for (let i = 0; i < N / 2; i++) {
    cs[i] = Math.cos((2 * Math.PI * i) / N);
    sn[i] = Math.sin((2 * Math.PI * i) / N);
  }
  const fft = function (re, im) {
    for (let i = 0; i < N; i++) {
      const j = rev[i];
      if (j > i) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    for (let size = 2; size <= N; size <<= 1) {
      const half = size >> 1, step = N / size;
      for (let i = 0; i < N; i += size) {
        for (let j = i, k = 0; j < i + half; j++, k += step) {
          const l = j + half;
          const tr = re[l] * cs[k] + im[l] * sn[k];
          const ti = im[l] * cs[k] - re[l] * sn[k];
          re[l] = re[j] - tr; im[l] = im[j] - ti;
          re[j] += tr; im[j] += ti;
        }
      }
    }
  };
  _fftCache[N] = fft;
  return fft;
}

// ============================================================
// FRAMING (librosa center=True, Padding N/2 links und rechts)
// ============================================================
function reflectIdx(i, n) {
  if (n === 1) return 0;
  const p = 2 * (n - 1);
  i = ((i % p) + p) % p;
  return i < n ? i : p - i;
}

function fillFrame(y, start, out, padMode) {
  const n = y.length, N = out.length;
  if (start >= 0 && start + N <= n) {
    for (let k = 0; k < N; k++) out[k] = y[start + k];
    return;
  }
  for (let k = 0; k < N; k++) {
    let i = start + k;
    if (i < 0 || i >= n) {
      if (padMode === 'reflect') i = reflectIdx(i, n);
      else { out[k] = 0; continue; }
    }
    out[k] = y[i];
  }
}

function timesLike(n, sr, hop) {
  const t = new Float64Array(n);
  for (let i = 0; i < n; i++) t[i] = (i * hop) / sr;
  return t;
}

// ============================================================
// AUDIO-EXTRAKTION / PIPELINE
// ============================================================
function trimAudio(y, sr, maxDuration) {
  const maxSamples = Math.trunc(sr * maxDuration);
  return y.length > maxSamples ? y.subarray(0, maxSamples) : y;
}

/**
 * librosa.yin (+ librosa.feature.rms im selben Durchlauf, gleiche Frames).
 * Autokorrelation per FFT; zwei Frames werden zu einer komplexen FFT gepackt.
 */
function computePitchAndRms(y, sr, cfg) {
  // librosa 0.11: autocorrelate() ueber den GANZEN Frame (linear, FFT-Laenge 2N),
  // d[tau] = 2*(acf[0]-acf[tau]) - cumsum(x^2)[tau-1]; max_period = min(ceil(sr/fmin), N-1)
  const N = cfg.FRAME_LENGTH, M = 2 * N, HOP = cfg.HOP_LENGTH;
  const nFrames = 1 + Math.floor(y.length / HOP);
  const minPeriod = Math.floor(sr / cfg.FMAX);
  const maxPeriod = Math.min(Math.ceil(sr / cfg.FMIN), N - 1);
  const L = maxPeriod - minPeriod + 1;
  if (L < 3) throw new Error('Samplerate zu niedrig fuer YIN-Parameter');
  const thr = cfg.TROUGH_THRESHOLD;
  const fft = makeFFT(M);

  const f0 = new Float64Array(nFrames);
  const rms = new Float64Array(nFrames);
  const x = new Float64Array(N);
  const zr = new Float64Array(M), zi = new Float64Array(M);
  const qr = new Float64Array(M), qi = new Float64Array(M);
  const cs = [new Float64Array(N), new Float64Array(N)];
  const yin = new Float64Array(L);

  for (let f = 0; f < nFrames; f += 2) {
    const cnt = f + 1 < nFrames ? 2 : 1;
    zr.fill(0); zi.fill(0);

    for (let p = 0; p < cnt; p++) {
      const fi = f + p;
      fillFrame(y, fi * HOP - (N >> 1), x, cfg.PAD_MODE);
      const tgt = p === 0 ? zr : zi;
      const c = cs[p];
      let acc = 0;
      for (let k = 0; k < N; k++) { tgt[k] = x[k]; acc += x[k] * x[k]; c[k] = acc; }
      rms[fi] = Math.sqrt(acc / N);
    }

    // zwei reelle Frames in einer komplexen FFT: z = x1 + i*x2
    fft(zr, zi);
    for (let k = 0; k < M; k++) {
      const m = (M - k) & (M - 1);
      const ar = (zr[k] + zr[m]) * 0.5, ai = (zi[k] - zi[m]) * 0.5;
      const br = (zi[k] + zi[m]) * 0.5, bi = -(zr[k] - zr[m]) * 0.5;
      qr[k] = ar * ar + ai * ai;       // |X1|^2
      qi[k] = -(br * br + bi * bi);    // -|X2|^2  (Konjugation fuer inverse FFT)
    }
    fft(qr, qi); // ifft(Q) = conj(fft(conj(Q)))/M -> acf1 = qr/M, acf2 = -qi/M

    for (let p = 0; p < cnt; p++) {
      const fi = f + p;
      const c = cs[p];
      const acfAt = p === 0 ? (t) => qr[t] / M : (t) => -qi[t] / M;
      const acf0 = acfAt(0);

      let cum = 0;
      for (let tau = 1; tau <= maxPeriod; tau++) {
        const d = 2 * (acf0 - acfAt(tau)) - c[tau - 1];
        cum += d;
        if (tau >= minPeriod) yin[tau - minPeriod] = d / (cum / tau + TINY_F32);
      }

      let globalMin = 0, minV = yin[0];
      for (let i = 1; i < L; i++) if (yin[i] < minV) { minV = yin[i]; globalMin = i; }

      let chosen = -1;
      for (let i = 0; i < L; i++) {
        let trough;
        if (i === 0) trough = yin[0] < yin[1];
        else {
          const next = i < L - 1 ? yin[i + 1] : yin[i];
          trough = yin[i] < yin[i - 1] && yin[i] <= next;
        }
        if (trough && yin[i] < thr) { chosen = i; break; }
      }
      if (chosen < 0) chosen = globalMin;

      // parabolische Interpolation (librosa _pi_stencil; Rand = 0)
      let shift = 0;
      if (chosen > 0 && chosen < L - 1) {
        const a = yin[chosen - 1] + yin[chosen + 1] - 2 * yin[chosen];
        const b = (yin[chosen + 1] - yin[chosen - 1]) / 2;
        shift = Math.abs(b) >= Math.abs(a) ? 0 : -b / a;
      }
      f0[fi] = sr / (minPeriod + chosen + shift);
    }

    if ((f & 1023) === 0) progress(0.45 * (f / nFrames), 'Pitch (YIN) & RMS');
  }
  return { f0, rms };
}

/** librosa.amplitude_to_db(rms, ref=np.max) inkl. top_db=80 */
function rmsToDb(rms) {
  const amin = 1e-5 * 1e-5; // amin**2
  const refMax = maxOf(rms);
  const refPow = refMax * refMax;
  const refDb = 10 * Math.log10(Math.max(amin, refPow));
  const db = new Float64Array(rms.length);
  let mx = -Infinity;
  for (let i = 0; i < rms.length; i++) {
    db[i] = 10 * Math.log10(Math.max(amin, rms[i] * rms[i])) - refDb;
    if (db[i] > mx) mx = db[i];
  }
  const floor = mx - 80.0;
  for (let i = 0; i < db.length; i++) if (db[i] < floor) db[i] = floor;
  return db;
}

// ---------- Mel-Filterbank (Slaney, norm='slaney', htk=False) ----------
function hzToMel(f) {
  const fSp = 200 / 3, minLogHz = 1000, minLogMel = minLogHz / fSp, logstep = Math.log(6.4) / 27;
  return f >= minLogHz ? minLogMel + Math.log(f / minLogHz) / logstep : f / fSp;
}
function melToHz(m) {
  const fSp = 200 / 3, minLogHz = 1000, minLogMel = minLogHz / fSp, logstep = Math.log(6.4) / 27;
  return m >= minLogMel ? minLogHz * Math.exp(logstep * (m - minLogMel)) : fSp * m;
}

function melFilterbank(sr, nFft, nMels, fmin, fmax) {
  const nb = (nFft >> 1) + 1;
  const fftF = new Float64Array(nb);
  const fstep = sr / 2 / (nb - 1);
  for (let i = 0; i < nb; i++) fftF[i] = i * fstep;

  const minMel = hzToMel(fmin), maxMel = hzToMel(fmax);
  const mstep = (maxMel - minMel) / (nMels + 1);
  const melF = new Float64Array(nMels + 2);
  for (let i = 0; i < nMels + 1; i++) melF[i] = melToHz(minMel + i * mstep);
  melF[nMels + 1] = melToHz(maxMel);

  const bands = [];
  const w = new Float64Array(nb);
  for (let i = 0; i < nMels; i++) {
    const fd0 = melF[i + 1] - melF[i], fd1 = melF[i + 2] - melF[i + 1];
    const enorm = 2.0 / (melF[i + 2] - melF[i]);
    let s = -1, e = -1;
    for (let k = 0; k < nb; k++) {
      const lower = -(melF[i] - fftF[k]) / fd0;
      const upper = (melF[i + 2] - fftF[k]) / fd1;
      let v = Math.max(0, Math.min(lower, upper));
      v = Math.fround(Math.fround(v) * enorm); // librosa: float32-Gewichte
      v = Math.fround(v);
      w[k] = v;
      if (v > 0) { if (s < 0) s = k; e = k; }
    }
    bands.push(s < 0 ? { start: 0, w: new Float64Array(0) } : { start: s, w: w.slice(s, e + 1) });
  }
  return bands;
}

/** melspectrogram(power=2) -> power_to_db(ref=1.0, amin=1e-10, top_db=80) (Top-dB-Clip folgt in onsetStrength) */
function computeMelDb(y, sr, cfg) {
  const N = cfg.FRAME_LENGTH, HOP = cfg.HOP_LENGTH, NM = cfg.N_MELS, NB = (N >> 1) + 1;
  const nFrames = 1 + Math.floor(y.length / HOP);
  const bands = melFilterbank(sr, N, NM, 0.0, cfg.MEL_FMAX || sr / 2);
  const fft = makeFFT(N);

  const win = new Float64Array(N); // Hann, periodisch (fftbins=True)
  for (let n = 0; n < N; n++) win[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / N);

  const out = new Float32Array(nFrames * NM);
  const x = new Float64Array(N);
  const zr = new Float64Array(N), zi = new Float64Array(N);
  const pa = new Float64Array(NB), pb = new Float64Array(NB);

  for (let f = 0; f < nFrames; f += 2) {
    const two = f + 1 < nFrames;
    fillFrame(y, f * HOP - (N >> 1), x, cfg.PAD_MODE);
    for (let k = 0; k < N; k++) zr[k] = x[k] * win[k];
    if (two) {
      fillFrame(y, (f + 1) * HOP - (N >> 1), x, cfg.PAD_MODE);
      for (let k = 0; k < N; k++) zi[k] = x[k] * win[k];
    } else zi.fill(0);
    fft(zr, zi);

    for (let k = 0; k < NB; k++) {
      const m = (N - k) & (N - 1);
      const ar = (zr[k] + zr[m]) * 0.5, ai = (zi[k] - zi[m]) * 0.5;
      const br = (zi[k] + zi[m]) * 0.5, bi = -(zr[k] - zr[m]) * 0.5;
      pa[k] = ar * ar + ai * ai;
      pb[k] = br * br + bi * bi;
    }

    for (let m = 0; m < NM; m++) {
      const { start, w } = bands[m];
      let sa = 0, sb = 0;
      for (let k = 0; k < w.length; k++) { sa += w[k] * pa[start + k]; sb += w[k] * pb[start + k]; }
      out[f * NM + m] = 10 * Math.log10(Math.max(1e-10, sa));
      if (two) out[(f + 1) * NM + m] = 10 * Math.log10(Math.max(1e-10, sb));
    }
    if ((f & 1023) === 0) progress(0.45 + 0.47 * (f / nFrames), 'Spektrum / Onset-Staerke');
  }
  return { melDb: out, nFrames };
}

/** librosa.onset.onset_strength(y, sr, hop_length) */
function onsetStrength(melDb, nFrames, cfg) {
  const NM = cfg.N_MELS, HOP = cfg.HOP_LENGTH;
  let mx = -Infinity;
  for (let i = 0; i < melDb.length; i++) if (melDb[i] > mx) mx = melDb[i];
  const top = mx - 80.0; // power_to_db top_db

  const lag = 1;
  const pad = lag + Math.floor(cfg.FRAME_LENGTH / (2 * HOP));
  const env = new Float32Array(nFrames);
  for (let k = 0; k < nFrames - 1; k++) {
    const j = k + pad;
    if (j >= nFrames) break;
    const a = k * NM, b = (k + 1) * NM;
    let s = 0;
    for (let m = 0; m < NM; m++) {
      const d = Math.max(melDb[b + m], top) - Math.max(melDb[a + m], top);
      if (d > 0) s += d;
    }
    env[j] = s / NM;
  }
  return env;
}

/** librosa.onset.onset_detect(onset_envelope=..., sr, hop_length) inkl. util.peak_pick */
function onsetDetect(envIn, sr, hop) {
  const n = envIn.length;
  if (n === 0) return [];
  const x = new Float64Array(n);
  let mn = Infinity;
  for (let i = 0; i < n; i++) if (envIn[i] < mn) mn = envIn[i];
  let mx = -Infinity;
  for (let i = 0; i < n; i++) { x[i] = envIn[i] - mn; if (x[i] > mx) mx = x[i]; }
  let any = false, finite = true;
  for (let i = 0; i < n; i++) {
    x[i] = x[i] / (mx + TINY_F32);
    if (x[i] !== 0) any = true;
    if (!Number.isFinite(x[i])) finite = false;
  }
  if (!any || !finite) return [];

  const preMax = Math.ceil(Math.floor((0.03 * sr) / hop));
  const postMax = Math.ceil(Math.floor((0.00 * sr) / hop) + 1);
  const preAvg = Math.ceil(Math.floor((0.10 * sr) / hop));
  const postAvg = Math.ceil(Math.floor((0.10 * sr) / hop) + 1);
  const wait = Math.ceil(Math.floor((0.03 * sr) / hop));
  const delta = 0.07;

  // gleitendes Maximum (scipy maximum_filter1d, mode='constant', cval=x.min())
  let xmin = Infinity;
  for (let i = 0; i < n; i++) if (x[i] < xmin) xmin = x[i];
  const maxLen = preMax + postMax;
  const maxOrigin = Math.ceil(0.5 * (preMax - postMax)) || 0;
  const maxHalf = Math.floor(maxLen / 2);
  const movMax = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const s = i - maxHalf - maxOrigin, e = s + maxLen - 1;
    let m = s < 0 || e >= n ? xmin : -Infinity;
    for (let k = Math.max(0, s); k <= Math.min(n - 1, e); k++) if (x[k] > m) m = x[k];
    movMax[i] = m;
  }

  // gleitender Mittelwert x[i-preAvg : i+postAvg] (mit Abschneiden an den Raendern,
  // entspricht uniform_filter1d + Randkorrektur in librosa)
  const pre = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) pre[i + 1] = pre[i] + x[i];

  const peaks = [];
  let last = -Infinity;
  for (let i = 0; i < n; i++) {
    const det = x[i] === movMax[i] ? x[i] : 0;
    if (det === 0) continue;
    const s = Math.max(0, i - preAvg), e = Math.min(n, i + postAvg);
    const avg = (pre[e] - pre[s]) / (e - s);
    if (det >= avg + delta && i > last + wait) { peaks.push(i); last = i; }
  }
  return peaks;
}

/** numpy.interp */
function interp(xq, xp, fp) {
  const n = xp.length;
  return xq.map((q) => {
    if (q <= xp[0]) return fp[0];
    if (q >= xp[n - 1]) return fp[n - 1];
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (xp[mid] <= q) lo = mid; else hi = mid; }
    const slope = (fp[lo + 1] - fp[lo]) / (xp[lo + 1] - xp[lo]);
    return slope * (q - xp[lo]) + fp[lo];
  });
}

// ============================================================
// ERGEBNISSE (RAM-Pfad, entspricht compute_*_results)
// ============================================================
function computeOnsetResults(tOnsets, rmsAtOnsets, intensities) {
  const count = tOnsets.length;
  const duration = count > 0 ? tOnsets[count - 1] : 0.0;
  return {
    events_per_minute: duration > 0 ? pyRound(count / (duration / 60), 2) : 0.0,
    rms_mean_at_onsets: count > 0 ? pyRound(mean(rmsAtOnsets), 4) : 0.0,
    intensity_mean: count > 0 ? pyRound(mean(intensities), 4) : 0.0
  };
}

function computePitchResults(f0, borderLow = BORDER_LOW, borderHigh = BORDER_HIGH) {
  const total = f0.length;
  let unter = 0, grenze = 0, ueber = 0, stumm = 0;
  for (let i = 0; i < total; i++) {
    const hz = f0[i];
    if (hz == null || Number.isNaN(hz)) stumm++;
    else if (hz < borderLow) unter++;
    else if (hz > borderHigh) ueber++;
    else grenze++;
  }
  const voiced = total - stumm;
  if (voiced === 0) return { unter_pct: 0.0, grenze_pct: 0.0, ueber_pct: 0.0 };
  return {
    unter_pct: pyRound((unter / voiced) * 100, 2),
    grenze_pct: pyRound((grenze / voiced) * 100, 2),
    ueber_pct: pyRound((ueber / voiced) * 100, 2)
  };
}

function detectVoiceChanges(pitchHzList, thresholdSemitones = HALBTON_SCHWELLE) {
  const idx = [];
  let prev = null;
  for (let i = 0; i < pitchHzList.length; i++) {
    const hz = pitchHzList[i];
    if (hz == null || Number.isNaN(hz)) continue;
    if (prev !== null && prev > 0 && hz > 0) {
      if (Math.abs(12 * Math.log2(hz / prev)) > thresholdSemitones) idx.push(i);
    }
    prev = hz;
  }
  return idx;
}

function computeVoiceChangeResults(f0, tPitch, rmsDb, thresholdSemitones = HALBTON_SCHWELLE) {
  const idx = detectVoiceChanges(f0, thresholdSemitones);
  const count = idx.length;
  const duration = tPitch.length > 0 ? tPitch[tPitch.length - 1] : 0.0;
  return {
    voice_change_count: count,
    voice_changes_per_minute: duration > 0 && count > 0 ? pyRound(count / (duration / 60), 2) : 0.0,
    rms_mean_at_changes: count > 0 ? pyRound(mean(idx.map((i) => rmsDb[i])), 4) : 0.0
  };
}

// ============================================================
// SCORES (1:1 aus dem Python-Skript)
// ============================================================
function calcVcs(vc, maxVcpm = MAX_VCPM, wVcpm = W_VCPM, wRms = W_RMS_VC) {
  const vcpm = vc.voice_changes_per_minute;
  const rmsLinear = 10 ** (vc.rms_mean_at_changes / 20);
  const vcpmNormiert = Math.min(100, (vcpm / maxVcpm) * 100);
  return pyRound(wVcpm * vcpmNormiert + wRms * (rmsLinear * 100), 2);
}

function calcStimmeScore(pitch) {
  const diff = pitch.ueber_pct - pitch.unter_pct;
  return pyRound(Math.min(100, Math.max(0, 50 + diff)), 2);
}

function calcActivityScoreNormiert(onset, maxEpm = MAX_EPM, wEpm = W_EPM, wRms = W_RMS) {
  const epm = onset.events_per_minute;
  const rmsLinear = 10 ** (onset.rms_mean_at_onsets / 20);
  const epmNormiert = Math.min(100, (epm / maxEpm) * 100);
  return pyRound(wEpm * epmNormiert + wRms * (rmsLinear * 100), 2);
}

function calcAufmerksamkeitsScore(onset, pitch, vc) {
  const stimmeValenz = calcStimmeScore(pitch);
  const acs = calcActivityScoreNormiert(onset);
  const vcs = calcVcs(vc);
  return pyRound(W_STIMME_VALENZ * stimmeValenz + W_ACS * acs + W_VCS * vcs, 2);
}

// ============================================================
// JSON-PFAD (entspricht export_analysis_json + analyze_* + output())
// Werte werden wie in den JSON-Dateien auf 4 Nachkommastellen gerundet.
// ============================================================
function buildJsonData(tOnsets, intensities, rmsAtOnsets, tPitch, f0, rmsDb) {
  const count = tOnsets.length;
  const duration = count > 0 ? tOnsets[count - 1] : 0.0;
  const onsetData = {
    onset_count: count,
    duration_seconds: pyRound(duration, 4),
    summary: {
      events_per_minute: duration > 0 ? pyRound(count / (duration / 60), 2) : 0,
      intensity_mean: count > 0 ? pyRound(mean(intensities), 4) : 0,
      intensity_max: count > 0 ? pyRound(maxOf(intensities), 4) : 0
    },
    onsets: tOnsets.map((t, i) => ({
      index: i,
      time_seconds: pyRound(t, 4),
      intensity_raw: pyRound(intensities[i], 4),
      rms_db: pyRound(rmsAtOnsets[i], 4)
    }))
  };
  const n = f0.length;
  const pitchData = { sample_count: n, time: new Float64Array(n), hz: new Array(n), rms: new Float64Array(n) };
  for (let i = 0; i < n; i++) {
    pitchData.time[i] = pyRound(tPitch[i], 4);
    pitchData.hz[i] = Number.isNaN(f0[i]) ? null : pyRound(f0[i], 4);
    pitchData.rms[i] = pyRound(rmsDb[i], 4);
  }
  return { onsetData, pitchData };
}

function analyzeOnsets(data) {
  const rms = data.onsets.map((o) => o.rms_db);
  const rmsMean = rms.length ? pyRound(mean(rms), 4) : 0.0;
  const rmsLinear = 10 ** (rmsMean / 20);
  return {
    onset_count: data.onset_count,
    events_per_minute: data.summary.events_per_minute,
    intensity_mean: data.summary.intensity_mean,
    intensity_max: data.summary.intensity_max,
    rms_mean_at_onsets: rmsMean,
    rms_max_at_onsets: rms.length ? pyRound(maxOf(rms), 4) : 0.0,
    activity_score: pyRound(data.summary.events_per_minute * rmsLinear, 2)
  };
}

function analyzePitchDistribution(pd) {
  let unter = 0, grenze = 0, ueber = 0, stumm = 0;
  for (const hz of pd.hz) {
    if (hz === null) stumm++;
    else if (hz < BORDER_LOW) unter++;
    else if (hz > BORDER_HIGH) ueber++;
    else grenze++;
  }
  const total = pd.hz.length, voiced = total - stumm;
  if (voiced === 0) return { total, voiced: 0, stumm, unter_pct: 0.0, grenze_pct: 0.0, ueber_pct: 0.0 };
  return {
    total, voiced, stumm,
    unter_pct: pyRound((unter / voiced) * 100, 2),
    grenze_pct: pyRound((grenze / voiced) * 100, 2),
    ueber_pct: pyRound((ueber / voiced) * 100, 2)
  };
}

function analyzeVoiceChanges(pd, thresholdSemitones = HALBTON_SCHWELLE) {
  const idx = detectVoiceChanges(pd.hz, thresholdSemitones);
  const count = idx.length;
  const duration = pd.time.length ? pd.time[pd.time.length - 1] : 0.0;
  return {
    voice_change_count: count,
    voice_changes_per_minute: duration > 0 && count > 0 ? pyRound(count / (duration / 60), 2) : 0.0,
    rms_mean_at_changes: count > 0 ? pyRound(mean(idx.map((i) => pd.rms[i])), 4) : 0.0
  };
}

/** entspricht output() – liefert die Zeilen statt zu printen */
function outputFromJson(onsetData, pitchData) {
  const onsetResults = analyzeOnsets(onsetData);
  const pitchResults = analyzePitchDistribution(pitchData);
  const vcResults = analyzeVoiceChanges(pitchData);

  const stimmeValenz = calcStimmeScore(pitchResults);
  const acs = calcActivityScoreNormiert(onsetResults);
  const vcs = calcVcs(vcResults);
  const score = calcAufmerksamkeitsScore(onsetResults, pitchResults, vcResults);

  const rmsLinOnsets = pyRound(10 ** (onsetResults.rms_mean_at_onsets / 20), 4);
  const rmsLinChanges = pyRound(10 ** (vcResults.rms_mean_at_changes / 20), 4);

  const zeilen = [
    ['EPM (Events/Min, Onsets)', onsetResults.events_per_minute],
    ['RMS an Onsets (dB)', onsetResults.rms_mean_at_onsets],
    ['RMS-linear an Onsets (0-1, → ACS)', rmsLinOnsets],
    ['ACS – Activity-Score normiert', acs],
    ['', null],
    ['VCPM (Stimmwechsel/Min)', vcResults.voice_changes_per_minute],
    ['RMS an Stimmwechseln (dB)', vcResults.rms_mean_at_changes],
    ['RMS-linear an Stimmwechseln (0-1, → VCS)', rmsLinChanges],
    ['VCS – Stimmwechsel-Score normiert', vcs],
    ['', null],
    ['Pitch unter 155Hz (Männerstimme)', `${pitchResults.unter_pct}%`],
    ['Pitch Grauzone 155-165Hz', `${pitchResults.grenze_pct}%`],
    ['Pitch über 165Hz (Frauen/Kinder)', `${pitchResults.ueber_pct}%`],
    ['Stimme-Valenz (0-100, 50=neutral)', stimmeValenz]
  ];
  return {
    zeilen,
    aufmerksamkeits_score: score,
    onset_results: onsetResults,
    pitch_results: pitchResults,
    voice_change_results: vcResults,
    stimme_valenz: stimmeValenz,
    activity_score_normiert: acs,
    voice_change_score: vcs
  };
}

function exportJsonStrings(onsetData, pitchData, tOnsetEnvTimes, env) {
  const envelope = [];
  for (let i = 0; i < env.length; i++) {
    envelope.push({ time_seconds: pyRound(tOnsetEnvTimes[i], 4), strength: pyRound(env[i], 4) });
  }
  const onsetJson = Object.assign({}, onsetData, { envelope });
  const pitch = [];
  for (let i = 0; i < pitchData.hz.length; i++) {
    pitch.push({ index: i, time_seconds: pitchData.time[i], pitch_hz: pitchData.hz[i], rms_db: pitchData.rms[i] });
  }
  return {
    'onset.json': JSON.stringify(onsetJson, null, 2),
    'pitch.json': JSON.stringify({ sample_count: pitchData.sample_count, pitch }, null, 2)
  };
}

// ============================================================
// HAUPT-PIPELINE (entspricht analyze_file() + main() + output())
// ============================================================
function analyzeAudio(yIn, sr, options) {
  const t0 = performance.now();
  const cfg = Object.assign({}, Config, options.config || {});
  const y = trimAudio(yIn, sr, cfg.MAX_DURATION);
  const HOP = cfg.HOP_LENGTH;

  // Pitch (YIN) + RMS
  progress(0, 'Pitch (YIN) & RMS');
  const { f0, rms } = computePitchAndRms(y, sr, cfg);
  const tPitch = timesLike(f0.length, sr, HOP);
  const rmsDb = rmsToDb(rms);
  const tRms = tPitch; // gleiche Frames / gleiche Zeitachse

  // Onsets
  const { melDb, nFrames } = computeMelDb(y, sr, cfg);
  progress(0.93, 'Onset-Detection');
  const env = onsetStrength(melDb, nFrames, cfg);
  const onsets = onsetDetect(env, sr, HOP);
  const tOnsets = onsets.map((o) => (o * HOP) / sr);
  const intensities = onsets.map((o) => env[o]);
  const rmsAtOnsets = interp(tOnsets, tRms, rmsDb);

  // Ergebnisse + Scores (RAM-Pfad = analyze_file)
  progress(0.97, 'Scores');
  const onsetResults = computeOnsetResults(tOnsets, rmsAtOnsets, intensities);
  const pitchResults = computePitchResults(f0);
  const vcResults = computeVoiceChangeResults(f0, tPitch, rmsDb);

  const stimmeValenz = calcStimmeScore(pitchResults);
  const acs = calcActivityScoreNormiert(onsetResults);
  const vcs = calcVcs(vcResults);
  const score = calcAufmerksamkeitsScore(onsetResults, pitchResults, vcResults);

  const analyzeFile = {
    stimme_valenz: stimmeValenz,
    activity_score_normiert: acs,
    voice_change_score: vcs,
    aufmerksamkeits_score: score,
    duration_seconds: pyRound(y.length / sr, 4),
    onset_count: tOnsets.length,
    events_per_minute: onsetResults.events_per_minute,
    rms_mean_at_onsets: onsetResults.rms_mean_at_onsets,
    voice_change_count: vcResults.voice_change_count,
    voice_changes_per_minute: vcResults.voice_changes_per_minute,
    rms_mean_at_changes: vcResults.rms_mean_at_changes,
    unter_pct: pitchResults.unter_pct,
    grenze_pct: pitchResults.grenze_pct,
    ueber_pct: pitchResults.ueber_pct
  };

  // JSON-Pfad (wie Python: export -> output()), mit gerundeten Werten
  const { onsetData, pitchData } = buildJsonData(tOnsets, intensities, rmsAtOnsets, tPitch, f0, rmsDb);
  const output = outputFromJson(onsetData, pitchData);

  let jsonExport = null;
  if (options.exportJson) {
    jsonExport = exportJsonStrings(onsetData, pitchData, timesLike(env.length, sr, HOP), env);
  }

  progress(1, 'Fertig');
  return {
    meta: {
      sample_rate: sr,
      samples: y.length,
      frames: nFrames,
      duration_seconds: analyzeFile.duration_seconds,
      elapsed_ms: Math.round(performance.now() - t0)
    },
    analyze_file: analyzeFile,
    details: { onset_results: onsetResults, pitch_results: pitchResults, voice_change_results: vcResults },
    output,
    json_export: jsonExport
  };
}

self.onmessage = (e) => {
  const { type, samples, sampleRate, options } = e.data;
  if (type !== 'analyze') return;
  try {
    const result = analyzeAudio(samples, sampleRate, options || {});
    self.postMessage({ type: 'result', result });
  } catch (err) {
    self.postMessage({ type: 'error', message: String((err && err.message) || err) });
  }
};
