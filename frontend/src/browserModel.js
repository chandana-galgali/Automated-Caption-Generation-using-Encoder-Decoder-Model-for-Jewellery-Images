// In-browser inference with ONNX Runtime Web: the same trained VGG-16 + GRU models
// (exported by training/export_onnx.py) run entirely on the visitor's device.
// Pure helpers (beamSearch, detokenize, nearest) mirror backend/caption_model.py.

const ORT_VERSION = '1.23.2';
const ORT_URL = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

let ortPromise = null;
function loadOrt() {
  if (window.ort) return Promise.resolve(window.ort);
  if (!ortPromise) {
    ortPromise = new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = `${ORT_URL}ort.min.js`;
      s.async = true;
      s.onload = () => {
        window.ort.env.wasm.wasmPaths = ORT_URL;
        window.ort.env.wasm.numThreads = 1; // GitHub Pages is not cross-origin isolated
        resolve(window.ort);
      };
      s.onerror = () => reject(new Error('Could not load ONNX Runtime'));
      document.head.appendChild(s);
    });
  }
  return ortPromise;
}

// ---------------------------------------------------------------- text helpers
const SPECIAL = new Set(['<PAD>', '<START>', '<END>', '<UNK>']);

export function detokenize(words) {
  let s = words.filter((w) => !SPECIAL.has(w)).join(' ');
  s = s.split('``').join('"').split("''").join('"');
  for (const p of [',', '.', '!', '?', ';', ':', ')', "'s"]) s = s.split(' ' + p).join(p);
  s = s.split('( ').join('(');
  s = s.replace(/^[ ,;:]+|[ ,;:]+$/g, '');
  if (s) {
    s = s[0].toUpperCase() + s.slice(1);
    if (!'.!?'.includes(s[s.length - 1])) s += '.';
  }
  return s;
}

// ---------------------------------------------------------------- beam search
// step(tokens: number[], hidden: Float32Array (B*H)) -> Promise<{ logp: Float32Array (B*V), hidden: Float32Array (B*H) }>
export async function beamSearch(step, h0, H, V, special, k = 3, maxLen = 40, noRepeat = 3) {
  const { start, end, pad, unk } = special;
  const banned = [pad, unk, start];
  let seqs = [[start]];
  let scores = [0];
  let hidden = h0;
  const finished = [];
  for (let t = 0; t < maxLen; t++) {
    const B = seqs.length;
    const { logp, hidden: hNew } = await step(seqs.map((s) => s[s.length - 1]), hidden);
    const cand = [];
    for (let b = 0; b < B; b++) {
      const row = logp.slice(b * V, (b + 1) * V);
      for (const x of banned) row[x] = -1e9;
      const s = seqs[b];
      if (noRepeat && s.length >= noRepeat) {
        const prefix = s.slice(s.length - (noRepeat - 1)).join(',');
        for (let i = 0; i + noRepeat <= s.length; i++) {
          if (s.slice(i, i + noRepeat - 1).join(',') === prefix) row[s[i + noRepeat - 1]] = -1e9;
        }
      }
      // top-k of this row is enough to find the global top-k
      const idx = topK(row, k);
      for (const j of idx) cand.push({ score: scores[b] + row[j], b, tok: j });
    }
    cand.sort((x, y) => y.score - x.score || x.b * V + x.tok - (y.b * V + y.tok));
    const top = cand.slice(0, Math.min(k, B * V));
    const newSeqs = [];
    const newScores = [];
    const rows = [];
    for (const c of top) {
      if (c.tok === end) finished.push({ seq: [...seqs[c.b], c.tok], score: c.score / seqs[c.b].length });
      else {
        newSeqs.push([...seqs[c.b], c.tok]);
        newScores.push(c.score);
        rows.push(c.b);
      }
    }
    if (!newSeqs.length || finished.length >= k) break;
    const hb = new Float32Array(rows.length * H);
    rows.forEach((r, i) => hb.set(hNew.subarray(r * H, (r + 1) * H), i * H));
    seqs = newSeqs;
    scores = newScores;
    hidden = hb;
  }
  if (!finished.length) seqs.forEach((s, i) => finished.push({ seq: s, score: scores[i] / s.length }));
  let best = finished[0];
  for (const f of finished) if (f.score > best.score) best = f;
  return best.seq.slice(1).filter((t) => t !== end);
}

function topK(row, k) {
  const out = [];
  for (let i = 0; i < row.length; i++) {
    if (out.length < k) {
      out.push(i);
      out.sort((a, b) => row[b] - row[a] || a - b);
    } else if (row[i] > row[out[k - 1]]) {
      out[k - 1] = i;
      out.sort((a, b) => row[b] - row[a] || a - b);
    }
  }
  return out;
}

// ---------------------------------------------------------------- retrieval
function fp16ToFloat32(u16) {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i];
    const s = h & 0x8000 ? -1 : 1;
    const e = (h >> 10) & 0x1f;
    const f = h & 0x3ff;
    out[i] = e === 0 ? s * 2 ** -14 * (f / 1024) : e === 31 ? (f ? NaN : s * Infinity) : s * 2 ** (e - 15) * (1 + f / 1024);
  }
  return out;
}

export function nearest(index, pooled, label) {
  const D = pooled.length;
  let norm = 0;
  for (let i = 0; i < D; i++) norm += pooled[i] * pooled[i];
  norm = Math.sqrt(norm) || 1;
  let best = -1;
  let bestSim = -Infinity;
  for (let n = 0; n < index.labels.length; n++) {
    if (index.labels[n] !== label) continue;
    let dot = 0;
    const off = n * D;
    for (let i = 0; i < D; i++) dot += index.features[off + i] * pooled[i];
    dot /= norm;
    if (dot > bestSim) {
      bestSim = dot;
      best = n;
    }
  }
  return { caption: index.captions[best], similarity: bestSim };
}

// ---------------------------------------------------------------- image -> tensor
// Decoding + resizing reproduce the server bit-for-bit: the browser decodes without
// colour conversion (identical to Pillow), and the resize is an exact port of Pillow.
// Exact port of Pillow's Image.resize(size, Image.BILINEAR) for 8-bit RGB (libImaging/Resample.c)
const PRECISION_BITS = 22;

function precomputeCoeffs(inSize, outSize) {
  const scale = inSize / outSize;
  const filterscale = scale < 1 ? 1 : scale;
  const support = 1.0 * filterscale; // bilinear filter support = 1
  const ksize = Math.ceil(support) * 2 + 1;
  const kk = new Int32Array(outSize * ksize);
  const bounds = new Int32Array(outSize * 2);
  const pre = new Float64Array(ksize);
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    const ss = 1 / filterscale;
    let xmin = Math.trunc(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5);
    if (xmax > inSize) xmax = inSize;
    xmax -= xmin;
    let ww = 0;
    for (let x = 0; x < xmax; x++) {
      let t = (x + xmin - center + 0.5) * ss;
      if (t < 0) t = -t;
      const w = t < 1 ? 1 - t : 0;
      pre[x] = w;
      ww += w;
    }
    for (let x = 0; x < ksize; x++) {
      const v = x < xmax && ww !== 0 ? pre[x] / ww : 0;
      kk[xx * ksize + x] = Math.trunc(v < 0 ? -0.5 + v * (1 << PRECISION_BITS) : 0.5 + v * (1 << PRECISION_BITS));
    }
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  return { ksize, kk, bounds };
}

const clip8 = (v) => {
  const x = v >> PRECISION_BITS;
  return x < 0 ? 0 : x > 255 ? 255 : x;
};

// src: Uint8 RGB (w*h*3) -> Uint8 RGB (ow*oh*3)
export function pillowResizeBilinear(src, w, h, ow, oh) {
  let cur = src, cw = w;
  if (ow !== w) {
    const { ksize, kk, bounds } = precomputeCoeffs(w, ow);
    const out = new Uint8Array(ow * h * 3);
    for (let y = 0; y < h; y++) {
      for (let xx = 0; xx < ow; xx++) {
        const xmin = bounds[xx * 2], xmax = bounds[xx * 2 + 1], k = xx * ksize;
        for (let c = 0; c < 3; c++) {
          let ss = 1 << (PRECISION_BITS - 1);
          for (let x = 0; x < xmax; x++) ss += cur[(y * cw + x + xmin) * 3 + c] * kk[k + x];
          out[(y * ow + xx) * 3 + c] = clip8(ss);
        }
      }
    }
    cur = out; cw = ow;
  }
  if (oh !== h) {
    const { ksize, kk, bounds } = precomputeCoeffs(h, oh);
    const out = new Uint8Array(cw * oh * 3);
    for (let yy = 0; yy < oh; yy++) {
      const ymin = bounds[yy * 2], ymax = bounds[yy * 2 + 1], k = yy * ksize;
      for (let x = 0; x < cw; x++) {
        for (let c = 0; c < 3; c++) {
          let ss = 1 << (PRECISION_BITS - 1);
          for (let y = 0; y < ymax; y++) ss += cur[((y + ymin) * cw + x) * 3 + c] * kk[k + y];
          out[(yy * cw + x) * 3 + c] = clip8(ss);
        }
      }
    }
    cur = out;
  }
  return cur;
}

async function imageToTensorData(file, cfg) {
  let bmp;
  try {
    bmp = await createImageBitmap(file, { colorSpaceConversion: 'none', premultiplyAlpha: 'none', imageOrientation: 'from-image' });
  } catch {
    throw new Error('This image format is not supported by your browser.');
  }
  const { width: w, height: h } = bmp;
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#ffffff'; // transparent pixels -> white, like the backend
  ctx.fillRect(0, 0, w, h);
  ctx.drawImage(bmp, 0, 0);
  bmp.close();
  const rgba = ctx.getImageData(0, 0, w, h).data;
  const rgb = new Uint8Array(w * h * 3);
  for (let i = 0, j = 0; i < rgba.length; i += 4) {
    rgb[j++] = rgba[i];
    rgb[j++] = rgba[i + 1];
    rgb[j++] = rgba[i + 2];
  }
  const S = cfg.size;
  const small = pillowResizeBilinear(rgb, w, h, S, S); // transforms.Resize((224, 224))
  const out = new Float32Array(3 * S * S);
  for (let i = 0; i < S * S; i++) {
    for (let c = 0; c < 3; c++) out[c * S * S + i] = (small[i * 3 + c] / 255 - cfg.mean[c]) / cfg.std[c];
  }
  return out;
}

// ---------------------------------------------------------------- model
export class BrowserModel {
  static async load(baseUrl, onProgress = () => {}) {
    const ort = await loadOrt();
    const m = new BrowserModel();
    m.ort = ort;
    onProgress('Loading model configuration…');
    m.cfg = await (await fetch(`${baseUrl}config.json`)).json();
    const opts = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
    const names = ['backbone', 'classifier', 'cap_init', 'cap_step'];
    m.sessions = {};
    for (const n of names) {
      onProgress(`Downloading ${n} model…`);
      m.sessions[n] = await ort.InferenceSession.create(`${baseUrl}${n}.onnx`, opts);
    }
    if (m.cfg.has_index) {
      onProgress('Downloading description index…');
      const [f, l, c] = await Promise.all([
        fetch(`${baseUrl}index_features.bin`).then((r) => r.arrayBuffer()),
        fetch(`${baseUrl}index_labels.bin`).then((r) => r.arrayBuffer()),
        fetch(`${baseUrl}index_captions.json`).then((r) => r.json()),
      ]);
      m.index = { features: fp16ToFloat32(new Uint16Array(f)), labels: new Int8Array(l), captions: c };
    }
    m.special = {
      start: m.cfg.special['<START>'], end: m.cfg.special['<END>'],
      pad: m.cfg.special['<PAD>'], unk: m.cfg.special['<UNK>'],
    };
    return m;
  }

  async analyze(file, mode = 'auto') {
    const { ort, cfg, sessions } = this;
    const t0 = performance.now();
    const image = new ort.Tensor('float32', await imageToTensorData(file, cfg), [1, 3, cfg.size, cfg.size]);

    const { fmap } = await sessions.backbone.run({ image });
    const { probs } = await sessions.classifier.run({ fmap });
    const p = probs.data;
    const label = p[1] > p[0] ? 1 : 0;

    const init = await sessions.cap_init.run({ fmap });
    const H = init.h.dims[1];
    const feats = init.feats;
    const proj = init.proj;
    const V = cfg.itos.length;
    const tile = (t, B) => {
      const one = t.data;
      const data = new Float32Array(one.length * B);
      for (let b = 0; b < B; b++) data.set(one, b * one.length);
      return new ort.Tensor('float32', data, [B, ...t.dims.slice(1)]);
    };
    const stepNames = sessions.cap_step.inputNames;
    const step = async (tokens, hidden) => {
      const B = tokens.length;
      const feeds = {
        token: new ort.Tensor('int32', Int32Array.from(tokens), [B]),
        h: new ort.Tensor('float32', hidden, [B, H]),
      };
      if (stepNames.includes('feats')) feeds.feats = tile(feats, B);
      if (stepNames.includes('proj')) feeds.proj = tile(proj, B);
      const out = await sessions.cap_step.run(feeds);
      return { logp: out.logp.data, hidden: out.h_out.data };
    };
    const ids = await beamSearch(step, init.h.data, H, V, this.special, cfg.beam, cfg.max_len, cfg.no_repeat_ngram);
    const generated = detokenize(ids.map((i) => cfg.itos[i]));

    const result = {
      type: cfg.class_names[label],
      confidence: Math.round(p[label] * 1e4) / 1e4,
      description: generated,
      method: 'generated',
      generated_description: generated,
    };
    const useRetrieval = mode === 'retrieve' || (mode === 'auto' && cfg.default_mode === 'retrieve');
    if (useRetrieval && this.index) {
      const d = fmap.dims[1];
      const n = fmap.dims[2] * fmap.dims[3];
      const pooled = new Float32Array(d);
      for (let c = 0; c < d; c++) {
        let s = 0;
        for (let i = 0; i < n; i++) s += fmap.data[c * n + i];
        pooled[c] = s / n;
      }
      const { caption, similarity } = nearest(this.index, pooled, label);
      Object.assign(result, { description: caption, method: 'retrieved', similarity: Math.round(similarity * 1e4) / 1e4 });
    }
    result.inference_ms = Math.round(performance.now() - t0);
    return result;
  }
}
