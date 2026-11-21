"""Evaluation metrics (pure Python / NumPy, no external evaluation toolkit needed).

Captioning: corpus BLEU-1..4 (Papineni et al. 2002, as in the COCO caption toolkit /
nltk.corpus_bleu), ROUGE-L (F-measure, beta = 1.2 as in COCO), CIDEr-D (Vedantam et
al. 2015, sigma = 6, clipping, x10 scaling, IDF from the reference corpus), and
METEOR via nltk when WordNet is available.
Classification: accuracy, per-class precision / recall / F1, macro averages,
confusion matrix, ROC-AUC.
"""

import math
from collections import Counter, defaultdict

import numpy as np


# ---------------------------------------------------------------- captioning
def _ngrams(words, n):
    return Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))


def corpus_bleu(refs, hyps, max_n=4):
    """refs: list of list-of-reference-token-lists; hyps: list of token lists.
    Returns [BLEU-1, ..., BLEU-max_n] (cumulative, uniform weights)."""
    matches, totals = [0] * max_n, [0] * max_n
    hyp_len = ref_len = 0
    for rs, h in zip(refs, hyps):
        hyp_len += len(h)
        ref_len += min((abs(len(r) - len(h)), len(r)) for r in rs)[1]  # closest ref length
        for n in range(1, max_n + 1):
            hc = _ngrams(h, n)
            maxref = Counter()
            for r in rs:
                for g, c in _ngrams(r, n).items():
                    maxref[g] = max(maxref[g], c)
            matches[n - 1] += sum(min(c, maxref[g]) for g, c in hc.items())
            totals[n - 1] += max(0, len(h) - n + 1)
    if hyp_len == 0:
        return [0.0] * max_n
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
    scores, log_sum = [], 0.0
    for n in range(max_n):
        p = matches[n] / totals[n] if totals[n] else 0.0
        if p == 0:
            scores.extend([0.0] * (max_n - n))
            break
        log_sum += math.log(p)
        scores.append(bp * math.exp(log_sum / (n + 1)))
    return scores


def _lcs(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_l(refs, hyps, beta=1.2):
    scores = []
    for rs, h in zip(refs, hyps):
        precs, recs = [], []
        for r in rs:
            l = _lcs(r, h)
            precs.append(l / len(h) if h else 0.0)
            recs.append(l / len(r) if r else 0.0)
        p, r = max(precs), max(recs)
        scores.append(((1 + beta ** 2) * p * r) / (r + beta ** 2 * p) if p and r else 0.0)
    return float(np.mean(scores))


def cider_d(refs, hyps, n=4, sigma=6.0):
    """CIDEr-D as in pycocoevalcap (document frequency computed over the references)."""
    def counts(words):
        c = Counter()
        for k in range(1, n + 1):
            c.update(_ngrams(words, k))
        return c

    ref_counts = [[counts(r) for r in rs] for rs in refs]
    df = defaultdict(float)
    for rcs in ref_counts:
        for g in set(g for rc in rcs for g in rc):
            df[g] += 1
    log_n = math.log(float(len(refs)))

    def vec(cnt):
        v = [defaultdict(float) for _ in range(n)]
        norm = [0.0] * n
        for g, tf in cnt.items():
            k = len(g) - 1
            v[k][g] = tf * (log_n - math.log(max(1.0, df[g])))
            norm[k] += v[k][g] ** 2
        return v, [math.sqrt(x) for x in norm]

    def sim(vh, vr, nh, nr, lh, lr):
        delta = lh - lr
        out = np.zeros(n)
        for k in range(n):
            for g, val in vh[k].items():
                out[k] += min(val, vr[k].get(g, 0.0)) * vr[k].get(g, 0.0)
            if nh[k] and nr[k]:
                out[k] /= nh[k] * nr[k]
            out[k] *= math.exp(-(delta ** 2) / (2 * sigma ** 2))
        return out

    scores = []
    for rs, rcs, h in zip(refs, ref_counts, hyps):
        vh, nh = vec(counts(h))
        acc = np.zeros(n)
        for r, rc in zip(rs, rcs):
            vr, nr = vec(rc)
            acc += sim(vh, vr, nh, nr, len(h), len(r))
        scores.append(float(np.mean(acc) / len(rs) * 10.0))
    return float(np.mean(scores))


def meteor(refs, hyps):
    try:
        import nltk
        from nltk.translate.meteor_score import meteor_score
        for pkg in ("wordnet", "omw-1.4"):
            nltk.download(pkg, quiet=True)
        return float(np.mean([meteor_score(rs, h) for rs, h in zip(refs, hyps)]))
    except Exception:
        return None


def caption_metrics(refs, hyps):
    """refs: list of reference token lists (one or more per image); hyps: token lists."""
    refs = [r if r and isinstance(r[0], list) else [r] for r in refs]
    b = corpus_bleu(refs, hyps)
    out = {f"BLEU-{i + 1}": b[i] for i in range(4)}
    out["METEOR"] = meteor(refs, hyps)
    out["ROUGE-L"] = rouge_l(refs, hyps)
    out["CIDEr-D"] = cider_d(refs, hyps)
    return out


# ------------------------------------------------------------ classification
def classification_metrics(y_true, y_pred, y_score, class_names):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    k = len(class_names)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    per_class = {}
    for i, name in enumerate(class_names):
        tp, fp, fn = cm[i, i], cm[:, i].sum() - cm[i, i], cm[i, :].sum() - cm[i, i]
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[name] = {"precision": prec, "recall": rec, "f1": f1, "support": int(cm[i].sum())}
    out = {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_precision": float(np.mean([v["precision"] for v in per_class.values()])),
        "macro_recall": float(np.mean([v["recall"] for v in per_class.values()])),
        "macro_f1": float(np.mean([v["f1"] for v in per_class.values()])),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }
    if k == 2 and y_score is not None:
        out["roc_auc"] = roc_auc(y_true, np.asarray(y_score))
    return out


def roc_auc(y_true, score):
    """Binary ROC-AUC via the rank statistic (ties averaged)."""
    order = np.argsort(score)
    ranks = np.empty(len(score))
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    pos = y_true == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if not n_pos or not n_neg:
        return None
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
