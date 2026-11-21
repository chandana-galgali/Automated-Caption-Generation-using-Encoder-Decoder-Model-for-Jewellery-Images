"""
Model definitions and inference pipeline for the Jewellery Analyzer.

Two model generations are supported (selected automatically from backend/weights):

v2 (training/run_all.py, default once retrained)
    * VGG-16 (frozen, ImageNet) -> 7x7x512 feature map, computed ONCE per image
    * Classifier: the 49 spatial vectors are read by a GRU (hidden 256) -> Earring / Necklace
    * Captioner: GRU decoder (embed 256, hidden 512) with soft attention over the
      49 VGG-16 regions ("Show, Attend and Tell"), beam search decoding
    * Optional description retrieval from the training set (cosine similarity of
      the same VGG-16 features); which mode is the default is decided on the
      validation set and stored in model_config.json
    Files: classifier.pth, captioner.pth, vocab.pkl, model_config.json, caption_index.npz

v1 (original project notebooks)
    * Classifier as above; captioner = VGG-16 -> avg-pool -> Linear(512) initialising
      a plain GRU decoder. Files: classifier.pth, encoder.pth, decoder.pth, vocab.pkl
"""

import json
import pickle
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

EMBED_SIZE = 256
HIDDEN_SIZE = 512
CLASSIFIER_HIDDEN = 256
CLASS_NAMES = ["Earring", "Necklace"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------
class VGG16_GRU_Classifier(nn.Module):
    def __init__(self, num_classes=2, hidden_size=CLASSIFIER_HIDDEN):
        super().__init__()
        self.features = models.vgg16(weights=None).features
        self.gru = nn.GRU(512, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def head(self, fmap):
        b, c, h, w = fmap.shape
        _, hidden = self.gru(fmap.reshape(b, c, h * w).permute(0, 2, 1))
        return self.fc(hidden.squeeze(0))

    def forward(self, x):
        return self.head(self.features(x))


class EncoderCNN(nn.Module):  # v1
    def __init__(self, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.features = models.vgg16(weights=None).features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, hidden_size)

    def head(self, fmap):
        return self.fc(self.pool(fmap).flatten(1))

    def forward(self, images):
        return self.head(self.features(images))


class DecoderRNN(nn.Module):  # v1
    def __init__(self, embed_size, hidden_size, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        outputs, _ = self.gru(self.embed(captions), features.unsqueeze(0))
        return self.fc(outputs)

    def step(self, tokens, hidden):
        out, hidden = self.gru(self.embed(tokens).unsqueeze(1), hidden)
        return self.fc(out.squeeze(1)), hidden


class AttentionCaptioner(nn.Module):  # v2
    """GRU decoder with additive soft attention over the 7x7 VGG-16 feature map."""

    def __init__(self, vocab_size, embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE,
                 feat_size=512, attn_size=256, dropout=0.3):
        super().__init__()
        self.feat_proj = nn.Sequential(nn.Linear(feat_size, feat_size), nn.ReLU())
        self.init_h = nn.Linear(feat_size, hidden_size)
        self.att_feat = nn.Linear(feat_size, attn_size)
        self.att_hid = nn.Linear(hidden_size, attn_size)
        self.att_v = nn.Linear(attn_size, 1)
        self.gate = nn.Linear(hidden_size, feat_size)
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRUCell(embed_size + feat_size, hidden_size)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def encode(self, fmap):
        """fmap (B, 512, 7, 7) -> state dict used by step()."""
        feats = self.feat_proj(fmap.flatten(2).transpose(1, 2))  # (B, 49, 512)
        return {"h": torch.tanh(self.init_h(feats.mean(1))), "feats": feats, "proj": self.att_feat(feats)}

    def step(self, tokens, state):
        h, feats, proj = state["h"], state["feats"], state["proj"]
        alpha = torch.softmax(self.att_v(torch.tanh(proj + self.att_hid(h).unsqueeze(1))).squeeze(-1), dim=1)
        context = (alpha.unsqueeze(-1) * feats).sum(1) * torch.sigmoid(self.gate(h))
        h = self.gru(torch.cat([self.embed(tokens), context], dim=1), h)
        return self.fc(self.drop(h)), {"h": h, "feats": feats, "proj": proj}, alpha

    def forward(self, fmap, captions):
        """Teacher forcing. captions (B, T) incl. <START>; returns logits (B, T, V) and alphas."""
        state = self.encode(fmap)
        logits, alphas = [], []
        for t in range(captions.size(1)):
            out, state, alpha = self.step(captions[:, t], state)
            logits.append(out); alphas.append(alpha)
        return torch.stack(logits, 1), torch.stack(alphas, 1)


# ---------------------------------------------------------------------------
# Vocabulary + text helpers
# ---------------------------------------------------------------------------
class Vocabulary:
    """Word <-> index mapping (also unpickles the original notebook's vocab.pkl)."""

    def __init__(self, freq_threshold=1):
        self.itos, self.stoi, self.freq_threshold = {}, {}, freq_threshold

    def __len__(self):
        return len(self.itos)


class _VocabUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "Vocabulary":
            return Vocabulary
        return super().find_class(module, name)


def load_vocab(path):
    with open(path, "rb") as f:
        return _VocabUnpickler(f).load()


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-.'][a-z0-9]+)*|[^\sa-z0-9]")
_SPECIAL = {"<PAD>", "<START>", "<END>", "<UNK>"}


def tokenize(text):
    return _TOKEN_RE.findall(text.lower())


def detokenize(words):
    sentence = " ".join(w for w in words if w not in _SPECIAL)
    sentence = sentence.replace("``", '"').replace("''", '"')
    for p in [",", ".", "!", "?", ";", ":", ")", "'s"]:
        sentence = sentence.replace(" " + p, p)
    sentence = sentence.replace("( ", "(").strip(" ,;:")
    if sentence:
        sentence = sentence[0].upper() + sentence[1:]
        if sentence[-1] not in ".!?":
            sentence += "."
    return sentence


# ---------------------------------------------------------------------------
# Decoding (works for both decoder generations)
# ---------------------------------------------------------------------------
def _index_state(state, rows):
    if isinstance(state, dict):
        return {k: v[rows] for k, v in state.items()}
    return state[:, rows]  # v1 GRU hidden (1, B, H)


def _expand_state(state, n):
    if isinstance(state, dict):
        return {k: v.expand(n, *v.shape[1:]).contiguous() for k, v in state.items()}
    return state.expand(-1, n, -1).contiguous()


def beam_search(step_fn, state, vocab, k=3, max_len=40, no_repeat_ngram=3):
    """Beam search. step_fn(tokens (B,), state) -> (logits (B, V), state)."""
    stoi = vocab.stoi
    start, end = stoi["<START>"], stoi["<END>"]
    banned = [stoi["<PAD>"], stoi["<UNK>"], start]
    device = next(iter(state.values())).device if isinstance(state, dict) else state.device

    seqs, scores, finished = [[start]], torch.zeros(1, device=device), []
    for _ in range(max_len):
        logits, state = step_fn(torch.tensor([s[-1] for s in seqs], device=device), state)
        logp = F.log_softmax(logits.float(), dim=-1)
        logp[:, banned] = -1e9
        if no_repeat_ngram:
            n = no_repeat_ngram
            for b, s in enumerate(seqs):
                if len(s) >= n:
                    prefix = tuple(s[-(n - 1):])
                    for i in range(len(s) - n + 1):
                        if tuple(s[i:i + n - 1]) == prefix:
                            logp[b, s[i + n - 1]] = -1e9
        V = logp.size(1)
        total = (scores.unsqueeze(1) + logp).view(-1)
        top = torch.topk(total, min(k, total.numel()))
        new_seqs, new_scores, rows = [], [], []
        for sc, idx in zip(top.values.tolist(), top.indices.tolist()):
            b, tok = divmod(idx, V)
            if tok == end:
                finished.append((seqs[b] + [tok], sc / len(seqs[b])))
            else:
                new_seqs.append(seqs[b] + [tok]); new_scores.append(sc); rows.append(b)
        if not new_seqs or len(finished) >= k:
            break
        seqs, scores = new_seqs, torch.tensor(new_scores, device=device)
        state = _index_state(state, torch.tensor(rows, device=device))
    if not finished:
        finished = [(s, sc / len(s)) for s, sc in zip(seqs, scores.tolist())]
    best = max(finished, key=lambda x: x[1])[0]
    return [vocab.itos[t] for t in best[1:] if t != end]


@torch.no_grad()
def greedy_batch(step_fn, state, vocab, batch_size, max_len=40):
    """Batched greedy decoding (used for fast validation during training)."""
    stoi = vocab.stoi
    device = next(iter(state.values())).device if isinstance(state, dict) else state.device
    tokens = torch.full((batch_size,), stoi["<START>"], dtype=torch.long, device=device)
    out, done = [], torch.zeros(batch_size, dtype=torch.bool, device=device)
    for _ in range(max_len):
        logits, state = step_fn(tokens, state)
        logits[:, [stoi["<PAD>"], stoi["<UNK>"], stoi["<START>"]]] = -1e9
        tokens = logits.argmax(-1)
        out.append(tokens)
        done |= tokens == stoi["<END>"]
        if done.all():
            break
    out = torch.stack(out, 1).tolist()
    result = []
    for seq in out:
        words = []
        for t in seq:
            if t == stoi["<END>"]:
                break
            words.append(vocab.itos[t])
        result.append(words)
    return result


# ---------------------------------------------------------------------------
# Description retrieval index
# ---------------------------------------------------------------------------
class CaptionIndex:
    def __init__(self, path):
        data = np.load(path, allow_pickle=False)
        self.features = torch.from_numpy(data["features"].astype(np.float32))  # (N, 512), L2-normalised
        self.labels = torch.from_numpy(data["labels"].astype(np.int64))
        self.captions = [str(c) for c in data["captions"]]

    def __len__(self):
        return len(self.captions)

    def nearest(self, pooled, label):
        q = F.normalize(pooled.float(), dim=-1).squeeze(0).cpu()
        sims = self.features @ q
        sims[self.labels != label] = -1.0
        i = int(sims.argmax())
        return self.captions[i], float(sims[i])


# ---------------------------------------------------------------------------
# Inference pipeline
# ---------------------------------------------------------------------------
class JewelleryAnalyzer:
    def __init__(self, weights_dir, device="cpu", caption_index=None, beam_size=3, max_len=40):
        weights_dir = Path(weights_dir)
        self.device = torch.device(device)
        self.vocab = load_vocab(weights_dir / "vocab.pkl")
        self.beam_size, self.max_len = beam_size, max_len

        def load(name):
            return torch.load(weights_dir / name, map_location="cpu", weights_only=True)

        cfg_path = weights_dir / "model_config.json"
        self.config = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        self.version = 2 if (weights_dir / "captioner.pth").exists() else 1

        self.classifier = VGG16_GRU_Classifier(len(CLASS_NAMES))
        cls_state = load("classifier.pth")
        self.classifier.load_state_dict(cls_state)
        self.shared_backbone = True

        if self.version == 2:
            self.captioner = AttentionCaptioner(len(self.vocab), **self.config.get("captioner", {}))
            self.captioner.load_state_dict(load("captioner.pth"))
            modules = [self.classifier, self.captioner]
        else:
            enc_state = load("encoder.pth")
            self.encoder = EncoderCNN(HIDDEN_SIZE)
            shared = all(torch.equal(cls_state[k], enc_state[k]) for k in enc_state if k.startswith("features."))
            if shared:  # identical frozen VGG-16 -> run it once
                self.encoder.features = nn.Identity()
                self.encoder.load_state_dict({k: v for k, v in enc_state.items() if not k.startswith("features.")})
            else:
                self.encoder.load_state_dict(enc_state)
            self.shared_backbone = shared
            self.decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, len(self.vocab))
            self.decoder.load_state_dict(load("decoder.pth"))
            modules = [self.classifier, self.encoder, self.decoder]
        for m in modules:
            m.to(self.device).eval()

        index_path = caption_index or weights_dir / "caption_index.npz"
        self.index = CaptionIndex(index_path) if Path(index_path).exists() else None
        # v2: decided on the validation set by run_all.py; v1: retrieval measured better on the test images
        self.default_mode = self.config.get("default_caption_mode", "retrieve" if self.index else "generate")

    def generate(self, fmap, x=None):
        if self.version == 2:
            state = self.captioner.encode(fmap)
            step = lambda tok, st: self.captioner.step(tok, st)[:2]  # noqa: E731
        else:
            h0 = self.encoder.head(fmap) if self.shared_backbone else self.encoder(x)
            state, step = h0.unsqueeze(0), self.decoder.step
        return detokenize(beam_search(step, state, self.vocab, self.beam_size, self.max_len))

    @torch.inference_mode()
    def analyze(self, image, mode="auto"):
        x = transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        fmap = self.classifier.features(x)  # VGG-16 once: (1, 512, 7, 7)
        probs = torch.softmax(self.classifier.head(fmap), dim=1)[0]
        label = int(probs.argmax())

        generated = self.generate(fmap, x)
        result = {
            "type": CLASS_NAMES[label],
            "confidence": round(float(probs[label]), 4),
            "description": generated,
            "method": "generated",
            "generated_description": generated,
        }
        use_retrieval = mode == "retrieve" or (mode == "auto" and self.default_mode == "retrieve")
        if use_retrieval and self.index is not None:
            caption, similarity = self.index.nearest(fmap.mean(dim=(2, 3)), label)
            result.update(description=caption, method="retrieved", similarity=round(similarity, 4))
        return result
