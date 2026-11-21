"""
Model definitions and inference pipeline for the Jewellery Analyzer.

Architecture (unchanged from training, see notebooks/):

* VGG16_GRU_Classifier - VGG-16 conv features (7x7x512) read as a sequence of
  49 spatial vectors by a GRU (hidden 256) -> Earring / Necklace.
* EncoderCNN + DecoderRNN - VGG-16 features -> global average pool -> Linear(512)
  initialises a GRU language model (embed 256, hidden 512) that writes the
  description word by word.

Inference optimisations (same weights, same outputs):

1. Shared backbone - both models were trained with the *frozen* ImageNet VGG-16,
   so their convolutional weights are identical. The image goes through VGG-16
   once and the 7x7x512 feature map feeds both heads (~2x faster, ~60 MB less RAM).
2. Incremental decoding - the GRU hidden state is carried forward instead of
   re-running the decoder over the whole prefix at every step (O(n) not O(n^2)).
3. Batched beam search over all beams at once.

Feature-based description retrieval (``CaptionIndex``, optional): the same
VGG-16 encoder features are compared (cosine similarity) with those of the 5,180
original training photos, restricted to the type predicted by the classifier, and
the reference description of the closest item is returned. On the 53-image
external test set this scores BLEU-4 0.045 versus 0.011 for free GRU generation
and names the correct jewellery type 94% of the time versus 13%, so it is the
default ("auto") mode. The GRU decoder's own caption is always returned too.
"""

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
# Modules (names match the saved state dicts)
# ---------------------------------------------------------------------------
class VGG16_GRU_Classifier(nn.Module):
    def __init__(self, num_classes=2, hidden_size=CLASSIFIER_HIDDEN):
        super().__init__()
        self.features = models.vgg16(weights=None).features
        self.gru = nn.GRU(512, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def head(self, fmap):
        b, c, h, w = fmap.shape
        _, hidden = self.gru(fmap.view(b, c, h * w).permute(0, 2, 1))
        return self.fc(hidden.squeeze(0))

    def forward(self, x):
        return self.head(self.features(x))


class EncoderCNN(nn.Module):
    def __init__(self, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.features = models.vgg16(weights=None).features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, hidden_size)

    def head(self, fmap):
        return self.fc(self.pool(fmap).flatten(1))

    def forward(self, images):
        return self.head(self.features(images))


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        outputs, _ = self.gru(self.embed(captions), features.unsqueeze(0))
        return self.fc(outputs)

    def step(self, tokens, hidden):
        """One decoding step for a batch. tokens: (B,), hidden: (1, B, H)."""
        out, hidden = self.gru(self.embed(tokens).unsqueeze(1), hidden)
        return self.fc(out.squeeze(1)), hidden


# ---------------------------------------------------------------------------
# Vocabulary + text helpers
# ---------------------------------------------------------------------------
class Vocabulary:
    """Stand-in for the Vocabulary class pickled by the training notebook."""

    def __init__(self, freq_threshold=1):
        self.itos, self.stoi, self.freq_threshold = {}, {}, freq_threshold

    def __len__(self):
        return len(self.itos)


class _VocabUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "Vocabulary":  # pickled as __main__.Vocabulary
            return Vocabulary
        return super().find_class(module, name)


def load_vocab(path):
    with open(path, "rb") as f:
        return _VocabUnpickler(f).load()


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-.'][a-z0-9]+)*|[^\sa-z0-9]")
_SPECIAL = {"<PAD>", "<START>", "<END>", "<UNK>"}


def tokenize(text):
    """Lightweight approximation of nltk.word_tokenize used during training."""
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
# Caption index (optional, built by training/build_caption_index.py)
# ---------------------------------------------------------------------------
class CaptionIndex:
    def __init__(self, path):
        data = np.load(path, allow_pickle=False)
        self.features = torch.from_numpy(data["features"].astype(np.float32))  # (N, 512), L2-normalised
        self.labels = torch.from_numpy(data["labels"].astype(np.int64))       # (N,) class index
        self.captions = [str(c) for c in data["captions"]]

    def __len__(self):
        return len(self.captions)

    def nearest(self, pooled, label):
        """Description of the most similar training image of the predicted type."""
        q = F.normalize(pooled.float(), dim=-1).squeeze(0).cpu()
        sims = self.features @ q
        sims[self.labels != label] = -1.0
        i = int(sims.argmax())
        return self.captions[i], float(sims[i])


# ---------------------------------------------------------------------------
# Inference pipeline
# ---------------------------------------------------------------------------
class JewelleryAnalyzer:
    def __init__(self, weights_dir, device="cpu", caption_index=None,
                 beam_size=3, max_len=40):
        weights_dir = Path(weights_dir)
        self.device = torch.device(device)
        self.vocab = load_vocab(weights_dir / "vocab.pkl")
        self.beam_size, self.max_len = beam_size, max_len

        def load(name):
            return torch.load(weights_dir / name, map_location="cpu", weights_only=True)

        cls_state, enc_state = load("classifier.pth"), load("encoder.pth")
        self.classifier = VGG16_GRU_Classifier(len(CLASS_NAMES))
        self.classifier.load_state_dict(cls_state)

        self.encoder = EncoderCNN(HIDDEN_SIZE)
        shared = all(torch.equal(cls_state[k], enc_state[k]) for k in enc_state if k.startswith("features."))
        if shared:  # identical frozen VGG-16 -> run it once and drop the duplicate
            self.encoder.features = nn.Identity()
            self.encoder.load_state_dict({k: v for k, v in enc_state.items() if not k.startswith("features.")})
        else:
            self.encoder.load_state_dict(enc_state)
        self.shared_backbone = shared

        self.decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, len(self.vocab))
        self.decoder.load_state_dict(load("decoder.pth"))

        for m in (self.classifier, self.encoder, self.decoder):
            m.to(self.device).eval()

        self.index = CaptionIndex(caption_index) if caption_index and Path(caption_index).exists() else None

        stoi = self.vocab.stoi
        self.start, self.end, self.pad, self.unk = stoi["<START>"], stoi["<END>"], stoi["<PAD>"], stoi["<UNK>"]

    # -- decoding -------------------------------------------------------------
    def beam_search(self, h0, k=None, no_repeat_ngram=3):
        """Batched beam search with carried hidden state. Returns [(tokens, avg_logprob)]."""
        k = k or self.beam_size
        V = len(self.vocab)
        banned = torch.tensor([self.pad, self.unk, self.start], device=self.device)

        seqs = [[self.start]]
        scores = torch.zeros(1, device=self.device)
        hidden = h0.unsqueeze(0)  # (1, 1, H)
        finished = []

        for _ in range(self.max_len):
            tokens = torch.tensor([s[-1] for s in seqs], device=self.device)
            logits, hidden = self.decoder.step(tokens, hidden)
            logp = F.log_softmax(logits, dim=-1)
            logp[:, banned] = -1e9
            if no_repeat_ngram:
                n = no_repeat_ngram
                for b, s in enumerate(seqs):
                    if len(s) >= n:
                        prefix = tuple(s[-(n - 1):])
                        for i in range(len(s) - n + 1):
                            if tuple(s[i:i + n - 1]) == prefix:
                                logp[b, s[i + n - 1]] = -1e9
            total = (scores.unsqueeze(1) + logp).view(-1)
            top = torch.topk(total, min(k, total.numel()))
            new_seqs, new_scores, rows = [], [], []
            for sc, idx in zip(top.values.tolist(), top.indices.tolist()):
                b, tok = divmod(idx, V)
                seq = seqs[b] + [tok]
                if tok == self.end:
                    finished.append((seq, sc / (len(seq) - 1)))
                else:
                    new_seqs.append(seq); new_scores.append(sc); rows.append(b)
            if not new_seqs or len(finished) >= k:
                break
            seqs = new_seqs
            scores = torch.tensor(new_scores, device=self.device)
            hidden = hidden[:, rows]
        if not finished:
            finished = [(s, sc / len(s)) for s, sc in zip(seqs, scores.tolist())]
        finished.sort(key=lambda x: x[1], reverse=True)
        return finished

    # -- full pipeline ----------------------------------------------------------
    @torch.inference_mode()
    def analyze(self, image, mode="auto"):
        x = transform(image.convert("RGB")).unsqueeze(0).to(self.device)

        fmap = self.classifier.features(x)  # VGG-16 once: (1, 512, 7, 7)
        probs = torch.softmax(self.classifier.head(fmap), dim=1)[0]
        label = int(probs.argmax())

        pooled = fmap.mean(dim=(2, 3))
        h0 = self.encoder.head(fmap) if self.shared_backbone else self.encoder(x)

        seq, _ = self.beam_search(h0)[0]
        generated = detokenize(self.vocab.itos[t] for t in seq[1:])

        result = {
            "type": CLASS_NAMES[label],
            "confidence": round(float(probs[label]), 4),
            "description": generated,
            "method": "generated",
            "generated_description": generated,
        }
        if mode == "auto" and self.index is not None:
            caption, similarity = self.index.nearest(pooled, label)
            result.update(description=caption, method="retrieved", similarity=round(similarity, 4))
        return result
