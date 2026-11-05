import time
import numpy as np
from . import config as c
from .data import load
from .util import dump


def candidates(key, ids, text, image, k=10):
    index = {v: i for i, v in enumerate(ids)}
    i = index[key]
    scores = 0.5 * (text @ text[i]) + 0.5 * (image @ image[i])
    scores[i] = -np.inf
    top = np.argsort(-scores, kind="stable")[: min(k, len(ids) - 1)]
    return [(ids[j], float(scores[j])) for j in top]


def evaluate_retrieval(k=10):
    p, q = load()
    with np.load(c.ART / "embeddings.npz", allow_pickle=False) as z:
        mask = p.split.eq("test").to_numpy()
        ids = z["ids"][mask].tolist()
        text = z["text"][mask]
        image = z["image"][mask]
    relevant = {}
    for r in q[(q.split == "test") & (q.label == 1)].itertuples():
        relevant.setdefault(r.left_id, set()).add(r.right_id)
        relevant.setdefault(r.right_id, set()).add(r.left_id)
    recalls = []
    times = []
    for key, matches in relevant.items():
        start = time.perf_counter()
        found = candidates(key, ids, text, image, k)
        times.append(1000 * (time.perf_counter() - start))
        recalls.append(len({x[0] for x in found} & matches) / len(matches))
    out = {
        "k": k,
        "eligible_queries": len(recalls),
        "test_catalog_listings": len(ids),
        "mean_known_positive_recall_at_k": float(np.mean(recalls)),
        "p50_candidate_ms": float(np.median(times)),
        "p95_candidate_ms": float(np.quantile(times, 0.95)),
        "scope": "Known labelled positive neighbours only; incomplete relevance labels, not exhaustive catalog recall. Frozen 50/50 text-image retrieval; embeddings precomputed.",
    }
    dump(c.REPORTS / "retrieval_metrics.json", out)
    return out
