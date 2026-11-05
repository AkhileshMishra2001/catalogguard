import argparse, json, time, platform, shutil
import numpy as np
from . import config as c
from .util import dump


def latency(runs=50):
    from .data import load
    from .features import FeatureStore
    from .modeling import load_champion, predict_scores

    _, q = load()
    b = load_champion()
    s = FeatureStore(neural=b["neural"])
    sample = q[q.split == "test"].reset_index(drop=True)

    def score(i):
        return predict_scores(b, s.pairs(sample.iloc[[i % len(sample)]]))

    for i in range(5):
        score(i)
    times = []
    for i in range(runs):
        start = time.perf_counter()
        score(i)
        times.append(1000 * (time.perf_counter() - start))
    out = {
        "model": b["mode"],
        "runs": runs,
        "p50_ms": float(np.median(times)),
        "p95_ms": float(np.quantile(times, 0.95)),
        "platform": platform.platform(),
        "scope": "Warm feature assembly and classifier only; cached embeddings. Excludes encoding, database, HTTP and cold start.",
    }
    dump(c.REPORTS / "latency.json", out)
    return out


def export_evidence():
    target = c.ROOT / "docs/my_results"
    target.mkdir(parents=True, exist_ok=True)
    files = []
    for name in [
        "data_audit.json",
        "analysis.json",
        "catalog_profile.csv",
        "validation_models.csv",
        "test_metrics.json",
        "retrieval_metrics.json",
        "latency.json",
        "abo_scale.json",
    ]:
        if (c.REPORTS / name).exists():
            shutil.copy2(c.REPORTS / name, target / name)
            files.append(name)
    return {
        "exported": files,
        "folder": str(target),
        "note": "Aggregates only: no secrets, images or reviewer annotations.",
    }


def main():
    p = argparse.ArgumentParser(description="CatalogGuard reproducible pipeline")
    sub = p.add_subparsers(dest="command", required=True)
    for name in [
        "download",
        "prepare",
        "analyze",
        "sync",
        "evaluate",
        "export-evidence",
    ]:
        sub.add_parser(name)
    e = sub.add_parser("embed")
    e.add_argument("--batch-size", type=int, default=16)
    e.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    t = sub.add_parser("train")
    t.add_argument(
        "--lexical-only",
        action="store_true",
        help="Smoke tests only, not final multimodal run",
    )
    r = sub.add_parser("retrieval")
    r.add_argument("--k", type=int, default=10)
    l = sub.add_parser("latency")
    l.add_argument("--runs", type=int, default=50)
    a = sub.add_parser("abo-scale")
    a.add_argument("--limit", type=int, default=0, help="0 means full metadata archive")
    args = p.parse_args()
    c.directories()
    if args.command == "download":
        from .data import fetch

        out = fetch()
    elif args.command == "prepare":
        from .data import prepare

        out = prepare()
    elif args.command == "analyze":
        from .analysis import analyze

        out = analyze()
    elif args.command == "sync":
        from .store import sync

        out = sync()
    elif args.command == "embed":
        if args.batch_size < 1:
            p.error("Positive batch size required")
        from .features import embed

        out = embed(args.batch_size, args.device)
    elif args.command == "train":
        from .modeling import train

        out = train(args.lexical_only)
    elif args.command == "evaluate":
        from .modeling import evaluate

        out = evaluate()
    elif args.command == "retrieval":
        if args.k < 1:
            p.error("Positive k required")
        from .retrieval import evaluate_retrieval

        out = evaluate_retrieval(args.k)
    elif args.command == "latency":
        if args.runs < 10:
            p.error("At least 10 runs required")
        out = latency(args.runs)
    elif args.command == "abo-scale":
        if args.limit < 0:
            p.error("Nonnegative limit required")
        from .scale import run

        out = run(args.limit)
    else:
        out = export_evidence()
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
