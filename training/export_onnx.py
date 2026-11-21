"""Export the trained models to ONNX so the website can run them in the browser.

    python training/export_onnx.py --weights backend/weights --out frontend/public/model

Works for both the original (v1) and the retrained (v2) weights. Produces
    backbone.onnx     image (1,3,224,224)            -> fmap (1,512,7,7)      VGG-16, run once
    classifier.onnx   fmap                           -> probs (1,2)
    cap_init.onnx     fmap                           -> h, feats, proj        decoder start state
    cap_step.onnx     token (B,) int32, h, feats, proj -> logp (B,V), h_out   one decoding step
    config.json       vocabulary, special tokens, decoding + preprocessing settings
    index_*.bin/json  retrieval index (if present)
and then checks that ONNX Runtime reproduces the PyTorch outputs and the
complete beam-search description for the sample images.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from caption_model import CLASS_NAMES, JewelleryAnalyzer, beam_search, detokenize, transform  # noqa: E402

OPSET = 17


class Backbone(nn.Module):
    def __init__(self, a):
        super().__init__(); self.f = a.classifier.features

    def forward(self, image):
        return self.f(image)


class ClassifierHead(nn.Module):
    def __init__(self, a):
        super().__init__(); self.c = a.classifier

    def forward(self, fmap):
        return torch.softmax(self.c.head(fmap), dim=1)


class CapInit(nn.Module):
    def __init__(self, a):
        super().__init__()
        self.version = a.version
        if a.version == 2:
            self.cap = a.captioner  # registered as sub-modules so their weights export as initializers
        else:
            self.enc = a.encoder

    def forward(self, fmap):
        if self.version == 2:
            s = self.cap.encode(fmap)
            return s["h"], s["feats"], s["proj"]
        h = self.enc.head(fmap)
        z = torch.zeros(1, 1, 1) + 0 * h.sum()  # placeholders (v1 has no attention)
        return h, z, z + 0


class CapStep(nn.Module):
    def __init__(self, a):
        super().__init__()
        self.version = a.version
        if a.version == 2:
            self.cap = a.captioner
        else:
            self.dec = a.decoder

    def forward(self, token, h, feats, proj):
        tok = token.long()
        if self.version == 2:
            logits, st, _ = self.cap.step(tok, {"h": h, "feats": feats, "proj": proj})
            h_out = st["h"]
        else:
            logits, hid = self.dec.step(tok, h.unsqueeze(0))
            h_out = hid.squeeze(0) + 0 * feats.sum() + 0 * proj.sum()  # keep inputs in the graph
        return F.log_softmax(logits, dim=-1), h_out


def export(module, args, path, inputs, outputs, dynamic):
    torch.onnx.export(module, args, str(path), input_names=inputs, output_names=outputs,
                      dynamic_axes=dynamic, opset_version=OPSET, do_constant_folding=True, dynamo=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=ROOT / "backend" / "weights", type=Path)
    ap.add_argument("--out", default=ROOT / "frontend" / "public" / "model", type=Path)
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    a = JewelleryAnalyzer(args.weights, device="cpu")
    for m in (a.classifier, getattr(a, "captioner", None), getattr(a, "encoder", None), getattr(a, "decoder", None)):
        if m is not None:
            m.eval()
            m.requires_grad_(False)  # export needs frozen weights
    print(f"loaded model v{a.version}, vocab {len(a.vocab)}, default mode {a.default_mode}")

    img = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        fmap = a.classifier.features(img)
        h, feats, proj = CapInit(a)(fmap)
        # Wrappers are put in eval mode BEFORE export: torch.onnx.export restores the wrapper's
        # original mode afterwards, and a fresh nn.Module starts in training mode, which would
        # silently switch the registered caption model (with dropout) into training mode.
        export(Backbone(a).eval(), (img,), out / "backbone.onnx", ["image"], ["fmap"], None)
        export(ClassifierHead(a).eval(), (fmap,), out / "classifier.onnx", ["fmap"], ["probs"], None)
        export(CapInit(a).eval(), (fmap,), out / "cap_init.onnx", ["fmap"], ["h", "feats", "proj"], None)
        tok = torch.tensor([1, 4, 5], dtype=torch.int32)
        H, Fe, P = (t.expand(3, *t.shape[1:]).contiguous() for t in (h, feats, proj))
        export(CapStep(a).eval(), (tok, H, Fe, P), out / "cap_step.onnx", ["token", "h", "feats", "proj"], ["logp", "h_out"],
               {"token": {0: "B"}, "h": {0: "B"}, "feats": {0: "B"}, "proj": {0: "B"}, "logp": {0: "B"}, "h_out": {0: "B"}})

    itos = [a.vocab.itos[i] for i in range(len(a.vocab))]
    cfg = {
        "version": a.version, "class_names": CLASS_NAMES, "itos": itos,
        "special": {k: a.vocab.stoi[k] for k in ("<PAD>", "<START>", "<END>", "<UNK>")},
        "beam": a.beam_size, "max_len": a.max_len, "no_repeat_ngram": 3,
        "default_mode": a.default_mode, "has_index": a.index is not None,
        "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "size": 224,
    }
    if a.index is not None:
        a.index.features.numpy().astype(np.float16).tofile(out / "index_features.bin")
        a.index.labels.numpy().astype(np.int8).tofile(out / "index_labels.bin")
        (out / "index_captions.json").write_text(json.dumps(a.index.captions), encoding="utf-8")
        cfg["index_size"] = len(a.index)
    # file sizes (for the download progress bar) and a content hash used as the browser cache key
    import hashlib
    digest = hashlib.sha256()
    cfg["files"] = {}
    for p in sorted(out.iterdir()):
        if p.name != "config.json":
            cfg["files"][p.name] = p.stat().st_size
            digest.update(p.name.encode()); digest.update(p.read_bytes())
    cfg["model_id"] = digest.hexdigest()[:16]
    (out / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    for p in sorted(out.iterdir()):
        print(f"  {p.name:22s} {p.stat().st_size / 1e6:7.2f} MB")

    if not args.no_check:
        check(a, out)


def check(a, out, tie_margin=0.01):
    """Verify the ONNX graphs against PyTorch.

    * encoder, classifier, decoder start state and every decoding step must match
      PyTorch numerically (ONNX Runtime and PyTorch use different maths kernels,
      so outputs agree to ~1e-5, not bit-for-bit);
    * the full beam-search caption must be identical - or, when two candidate
      captions are a near-tie, PyTorch itself must score them within `tie_margin`
      (average log-probability per word). A real export bug gives large gaps.
    """
    import onnxruntime as ort
    for m in (a.classifier, getattr(a, "captioner", None), getattr(a, "encoder", None), getattr(a, "decoder", None)):
        if m is not None:
            m.eval()
    if a.version == 2:
        assert not a.captioner.training and not a.captioner.drop.training, "reference model must be in eval mode"
    s = {n: ort.InferenceSession(str(out / f"{n}.onnx"), providers=["CPUExecutionProvider"])
         for n in ("backbone", "classifier", "cap_init", "cap_step")}
    stoi = a.vocab.stoi
    start, end = stoi["<START>"], stoi["<END>"]

    def torch_state(fmap_t, x):
        if a.version == 2:
            st = a.captioner.encode(fmap_t)
            return st, lambda tok, st: a.captioner.step(tok, st)[:2], st["h"]
        h0 = a.encoder.head(fmap_t)
        return h0.unsqueeze(0), a.decoder.step, h0

    def torch_score(fmap_t, x, words):
        """Average log-probability per token (incl. <END>) that PyTorch gives a caption."""
        st, step, _ = torch_state(fmap_t, x)
        ids = [start] + [stoi[w] for w in words] + [end]
        total = 0.0
        for cur, nxt in zip(ids[:-1], ids[1:]):
            logits, st = step(torch.tensor([cur]), st)
            total += F.log_softmax(logits.float(), -1)[0, nxt].item()
        return total / (len(ids) - 1)

    exact = ties = 0
    samples = sorted((ROOT / "test_images").glob("*"))[:5]
    for path in samples:
        x = transform(Image.open(path).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            fmap_t = a.classifier.features(x)
            probs_t = torch.softmax(a.classifier.head(fmap_t), 1)
            st_t, step_t, h_t = torch_state(fmap_t, x)
            words_t = beam_search(step_t, st_t, a.vocab, a.beam_size, a.max_len)

            fmap = s["backbone"].run(None, {"image": x.numpy()})[0]
            probs = s["classifier"].run(None, {"fmap": fmap})[0]
            h, feats, proj = s["cap_init"].run(None, {"fmap": fmap_t.numpy()})
            assert np.allclose(fmap, fmap_t.numpy(), atol=1e-3, rtol=1e-3), "backbone mismatch"
            assert np.allclose(probs, probs_t.numpy(), atol=1e-4), "classifier mismatch"
            assert np.allclose(h, h_t.numpy(), atol=1e-4), "decoder start state mismatch"

            # every step of the torch caption, fed to both implementations
            ids = [start] + [stoi[w] for w in words_t]
            st, oh, worst = st_t, h, 0.0
            for tok in ids:
                logits_t, st = step_t(torch.tensor([tok]), st)
                logp_o, oh = s["cap_step"].run(None, {"token": np.array([tok], np.int32), "h": oh,
                                                       "feats": feats, "proj": proj})
                worst = max(worst, float(np.abs(F.log_softmax(logits_t.float(), -1).numpy() - logp_o).max()))
            assert worst < 1e-3, f"decoder step mismatch (max |delta log p| = {worst:.2e})"

            # full beam search with the ONNX graphs, starting from the ONNX encoder output
            h_o, feats_o, proj_o = s["cap_init"].run(None, {"fmap": fmap})

            def step_o(tokens, st):
                B = tokens.shape[0]
                logp, h_out = s["cap_step"].run(None, {
                    "token": tokens.numpy().astype(np.int32), "h": st["h"].numpy(),
                    "feats": np.repeat(feats_o, B, 0), "proj": np.repeat(proj_o, B, 0)})
                return torch.from_numpy(logp), {"h": torch.from_numpy(h_out)}

            words_o = beam_search(step_o, {"h": torch.from_numpy(h_o)}, a.vocab, a.beam_size, a.max_len)
            if words_o == words_t:
                exact += 1
                note = "identical"
            else:
                gap = abs(torch_score(fmap_t, x, words_o) - torch_score(fmap_t, x, words_t))
                assert gap < tie_margin, (f"caption mismatch beyond a near-tie (score gap {gap:.4f}):\n"
                                          f" onnx  {detokenize(words_o)}\n torch {detokenize(words_t)}")
                ties += 1
                note = f"near-tie between two candidates (score gap {gap:.4f}): torch picked \"{detokenize(words_t)}\""
        print(f"  OK {path.name}: {CLASS_NAMES[int(probs.argmax())]} ({probs.max():.3f}) | {detokenize(words_o)}  [{note}]")
    print(f"ONNX export verified against PyTorch: step outputs match; captions identical {exact}/{len(samples)}, "
          f"near-ties {ties}/{len(samples)}.")


if __name__ == "__main__":
    main()
