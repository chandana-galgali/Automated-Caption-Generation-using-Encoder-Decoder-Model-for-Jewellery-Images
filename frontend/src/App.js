import React, { useCallback, useEffect, useRef, useState } from 'react';
import axios from 'axios';
import { Upload, Loader2, Scan, Gem, AlertCircle, RotateCcw, Code, Cpu } from 'lucide-react';

// Same origin by default (the FastAPI backend serves this app). During development
// the "proxy" field in package.json forwards API calls to http://localhost:8000.
const API_URL = process.env.REACT_APP_API_URL || '';
const REPO_URL =
  'https://github.com/chandana-galgali/Automated-Caption-Generation-using-Encoder-Decoder-Model-for-Jewellery-Images';

const SAMPLES = [
  { src: 'samples/earring-1.jpg', label: 'Gold hoop earrings' },
  { src: 'samples/earring-2.jpg', label: 'Ruby flower earrings' },
  { src: 'samples/earring-3.jpg', label: 'Pearl jhumkas' },
  { src: 'samples/necklace-1.jpg', label: 'Gold necklace' },
  { src: 'samples/necklace-2.jpg', label: 'Diamond necklace' },
  { src: 'samples/necklace-3.jpg', label: 'Layered necklace' },
];

const MAX_SIDE = 640; // the model only needs 224x224; shrinking first makes uploads fast

// Downscale large photos in the browser and flatten transparency onto white
// (the training images are mostly product shots on white backgrounds).
async function prepareImage(file) {
  const url = URL.createObjectURL(file);
  try {
    const img = await new Promise((resolve, reject) => {
      const i = new Image();
      i.onload = () => resolve(i);
      i.onerror = reject;
      i.src = url;
    });
    const scale = Math.min(1, MAX_SIDE / Math.max(img.width, img.height));
    const canvas = document.createElement('canvas');
    canvas.width = Math.round(img.width * scale);
    canvas.height = Math.round(img.height * scale);
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    return await new Promise((resolve) => canvas.toBlob((b) => resolve(b || file), 'image/jpeg', 0.92));
  } catch {
    return file; // unsupported in this browser (e.g. some AVIF) -> let the server handle it
  } finally {
    URL.revokeObjectURL(url);
  }
}

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [dragging, setDragging] = useState(false);
  const [status, setStatus] = useState('checking'); // checking | online | offline
  const inputRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    const ping = () =>
      axios
        .get(`${API_URL}/health`, { timeout: 10000 })
        .then(() => 'online')
        .catch(() => 'offline')
        .then((s) => {
          if (!cancelled) setStatus(s);
        });
    ping();
    const id = setInterval(ping, 30000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  useEffect(() => () => preview && URL.revokeObjectURL(preview), [preview]);

  const selectFile = useCallback((selected) => {
    if (!selected) return;
    if (!selected.type.startsWith('image/')) {
      setError('Please choose an image file (JPG, PNG or WEBP).');
      return;
    }
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setResult(null);
    setError('');
  }, []);

  const analyze = useCallback(async (target) => {
    if (!target) return;
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const body = new FormData();
      body.append('file', await prepareImage(target), target.name || 'image.jpg');
      const { data } = await axios.post(`${API_URL}/analyze`, body, { timeout: 60000 });
      setResult(data);
      setStatus('online');
    } catch (err) {
      setError(
        err.response
          ? err.response.data?.detail || `Server error (${err.response.status})`
          : 'Cannot reach the model server. It may be waking up, so try again in a few seconds.'
      );
    } finally {
      setLoading(false);
    }
  }, []);

  const trySample = async (sample) => {
    const blob = await (await fetch(`${process.env.PUBLIC_URL}/${sample.src}`)).blob();
    const f = new File([blob], sample.src.split('/').pop(), { type: blob.type || 'image/jpeg' });
    selectFile(f);
    analyze(f);
  };

  useEffect(() => {
    const onPaste = (e) => {
      const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith('image/'));
      if (item) selectFile(item.getAsFile());
    };
    window.addEventListener('paste', onPaste);
    return () => window.removeEventListener('paste', onPaste);
  }, [selectFile]);

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError('');
    if (inputRef.current) inputRef.current.value = '';
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    selectFile(e.dataTransfer.files?.[0]);
  };

  const statusColor = { online: 'bg-emerald-500', offline: 'bg-red-400', checking: 'bg-amber-400' }[status];
  const statusText = { online: 'Model online', offline: 'Model offline', checking: 'Connecting…' }[status];

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-4 md:p-6 bg-stone-100 text-stone-800">
      <div className="max-w-5xl w-full bg-white rounded-xl shadow-xl overflow-hidden flex flex-col md:flex-row border border-stone-200">
        {/* Left: upload */}
        <div className="w-full md:w-1/2 p-6 md:p-10 flex flex-col items-center bg-stone-50">
          <div className="mb-6 text-center">
            <h1 className="text-3xl font-serif font-bold text-emerald-900 mb-2">Jewellery Analyzer</h1>
            <p className="text-stone-500 text-xs tracking-widest uppercase">
              Upload, drop or paste a jewellery image
            </p>
          </div>

          <div
            className="w-full relative"
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
          >
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              aria-label="Choose an image"
              onChange={(e) => selectFile(e.target.files[0])}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
            />
            <div
              className={`border-2 border-dashed rounded-lg h-72 flex flex-col items-center justify-center transition-all duration-200 overflow-hidden bg-white
              ${dragging ? 'border-emerald-500 bg-emerald-50' : error ? 'border-red-300' : preview ? 'border-emerald-500' : 'border-stone-300 hover:border-emerald-400'}`}
            >
              {preview ? (
                <img src={preview} alt="Selected jewellery" className="h-full w-full object-contain p-4" />
              ) : (
                <>
                  <Upload className="w-10 h-10 text-stone-400 mb-3" />
                  <span className="text-stone-500 font-medium">Click or drag an image here</span>
                  <span className="text-stone-400 text-xs mt-1">JPG · PNG · WEBP · up to 10 MB</span>
                </>
              )}
            </div>
          </div>

          {error && (
            <div role="alert" className="mt-4 flex items-start gap-2 text-red-600 text-sm">
              <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
              {error}
            </div>
          )}

          <div className="mt-6 w-full flex gap-3">
            <button
              onClick={() => analyze(file)}
              disabled={!file || loading}
              className={`flex-1 py-4 px-6 rounded-lg font-bold flex items-center justify-center gap-2 transition-all uppercase tracking-widest text-sm
              ${!file || loading ? 'bg-stone-200 text-stone-400 cursor-not-allowed' : 'bg-emerald-900 text-white hover:bg-emerald-800 shadow-md'}`}
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" /> Processing…
                </>
              ) : (
                <>
                  <Scan className="w-4 h-4" /> Identify Item
                </>
              )}
            </button>
            {file && !loading && (
              <button
                onClick={reset}
                title="Start over"
                className="px-4 rounded-lg border border-stone-300 text-stone-500 hover:bg-stone-100"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
            )}
          </div>

          <div className="mt-8 w-full">
            <p className="text-[11px] font-bold text-stone-400 uppercase tracking-widest mb-2">Or try a sample</p>
            <div className="grid grid-cols-6 gap-2">
              {SAMPLES.map((s) => (
                <button
                  key={s.src}
                  onClick={() => trySample(s)}
                  disabled={loading}
                  title={s.label}
                  className="aspect-square rounded-md overflow-hidden border border-stone-200 bg-white hover:border-emerald-500 disabled:opacity-50"
                >
                  <img src={`${process.env.PUBLIC_URL}/${s.src}`} alt={s.label} className="w-full h-full object-cover" />
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Right: results */}
        <div className="w-full md:w-1/2 p-6 md:p-10 bg-white flex flex-col justify-center border-t md:border-t-0 md:border-l border-stone-100 min-h-[22rem]">
          {loading ? (
            <div className="text-center text-stone-400 flex flex-col items-center">
              <Loader2 className="w-10 h-10 mb-4 animate-spin text-emerald-700" />
              <p className="font-serif italic">Encoding image and generating description…</p>
            </div>
          ) : !result ? (
            <div className="text-center text-stone-300 flex flex-col items-center">
              <Gem className="w-16 h-16 mb-4 opacity-20" />
              <p className="font-serif italic">Analysis details will appear here</p>
            </div>
          ) : (
            <div>
              <div className="mb-8">
                <h3 className="text-xs font-bold text-stone-400 uppercase tracking-widest mb-2">Type</h3>
                <p className="text-4xl font-serif text-stone-900 border-b-2 border-emerald-500 inline-block pb-2">
                  {result.type}
                </p>
                {result.confidence !== undefined && (
                  <div className="mt-4 max-w-xs">
                    <div className="flex justify-between text-xs text-stone-500 mb-1">
                      <span>Confidence</span>
                      <span>{(result.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="h-1.5 bg-stone-100 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-emerald-600 rounded-full transition-all duration-700"
                        style={{ width: `${result.confidence * 100}%` }}
                      />
                    </div>
                    {result.confidence < 0.85 && (
                      <p className="mt-2 text-xs text-amber-700">
                        Low confidence: the model was trained on earrings and necklaces only, mostly product photos.
                      </p>
                    )}
                  </div>
                )}
              </div>

              <div>
                <h3 className="text-xs font-bold text-stone-400 uppercase tracking-widest mb-3">
                  Description
                  {result.method === 'retrieved' && (
                    <span className="ml-2 normal-case tracking-normal font-normal text-stone-400">
                      (closest catalogue item, {Math.round((result.similarity || 0) * 100)}% visual match)
                    </span>
                  )}
                </h3>
                <p className="text-stone-600 leading-loose font-light text-lg">{result.description}</p>
                {result.method === 'retrieved' && result.generated_description && (
                  <details className="mt-4 text-sm text-stone-500">
                    <summary className="cursor-pointer select-none hover:text-emerald-800">
                      Raw GRU decoder output
                    </summary>
                    <p className="mt-2 italic">{result.generated_description}</p>
                  </details>
                )}
              </div>

              {result.inference_ms !== undefined && (
                <p className="mt-8 flex items-center gap-2 text-xs text-stone-400">
                  <Cpu className="w-3.5 h-3.5" />
                  Inference {result.inference_ms} ms · VGG-16 encoder + GRU decoder
                </p>
              )}
            </div>
          )}
        </div>
      </div>

      <footer className="mt-6 flex flex-wrap items-center justify-center gap-x-6 gap-y-2 text-xs text-stone-500">
        <span className="flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full ${statusColor}`} />
          {statusText}
        </span>
        <a href={REPO_URL} target="_blank" rel="noreferrer" className="flex items-center gap-1 hover:text-emerald-800">
          <Code className="w-3.5 h-3.5" /> Source code
        </a>
        <span>B.Tech IT Project · KJSCE, Somaiya Vidyavihar University</span>
      </footer>
    </div>
  );
}

export default App;
