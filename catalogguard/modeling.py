import time, platform, importlib.metadata, joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
)
from lightgbm import LGBMClassifier
from . import config as c
from .data import load, UnionFind
from .features import FeatureStore, FEATURES, MODES, fingerprint
from .util import read, dump, sha


def metrics(y, scores, threshold=0.5):
    pred = scores >= threshold
    return {
        "rows": len(y),
        "positive_rate": float(np.mean(y)),
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, scores)),
        "threshold": float(threshold),
    }


def thresholds(y, scores, target=0.95, min_selected=20):
    grid = np.linspace(0.05, 0.99, 95)
    best = max(grid, key=lambda t: f1_score(y, scores >= t, zero_division=0))
    candidates = [
        float(t)
        for t in grid
        if (scores >= t).sum() >= min_selected
        and float(np.mean(np.asarray(y)[scores >= t])) >= target
    ]
    return {
        "f1_threshold": float(best),
        "suggestion_threshold": min(candidates) if candidates else None,
        "validation_precision_target": target,
        "minimum_validation_suggestions": min_selected,
        "note": "Empirical validation target, not a production precision guarantee. Human approval required.",
    }


def predict_scores(bundle, x):
    values = x[:, bundle["columns"]]
    if bundle["mode"] == "fusion_lgbm":
        values = pd.DataFrame(values, columns=bundle["model"].feature_name_)
    return bundle["model"].predict_proba(values)[:, 1]


def versions():
    return {
        p: importlib.metadata.version(p)
        for p in ["numpy", "scipy", "scikit-learn", "lightgbm", "mlflow", "torch"]
    }


def tracking():
    import mlflow

    mlflow.set_tracking_uri("sqlite:///" + (c.ROOT / "mlflow.db").as_posix())
    mlflow.set_experiment("CatalogGuard")
    return mlflow


def train(lexical_only=False):
    c.directories()
    _, pairs = load()
    store = FeatureStore(fit=True, neural=not lexical_only)
    x = store.pairs(pairs)
    y = pairs.label.to_numpy()
    tr = pairs.split.eq("train").to_numpy()
    va = pairs.split.eq("validation").to_numpy()
    results = []
    bundles = {}
    mlflow = tracking()
    for mode in ["lexical_lr"] if lexical_only else MODES:
        columns = MODES[mode]
        model = (
            LGBMClassifier(
                n_estimators=180,
                num_leaves=7,
                max_depth=4,
                min_child_samples=30,
                learning_rate=0.04,
                reg_lambda=2,
                verbosity=-1,
                random_state=c.SEED,
                n_jobs=2,
            )
            if mode == "fusion_lgbm"
            else make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1, max_iter=1500, random_state=c.SEED),
            )
        )
        start = time.perf_counter()
        model.fit(x[tr][:, columns], y[tr])
        bundle = {
            "model": model,
            "mode": mode,
            "columns": columns,
            "fingerprint": fingerprint(),
            "neural": not lexical_only,
            "versions": versions(),
        }
        scores = predict_scores(bundle, x[va])
        bundle["policy"] = thresholds(y[va], scores)
        bundles[mode] = bundle
        result = {
            "model": mode,
            **metrics(y[va], scores, bundle["policy"]["f1_threshold"]),
            "fit_seconds": time.perf_counter() - start,
        }
        joblib.dump(bundle, c.ART / (mode + ".joblib"))
        with mlflow.start_run(run_name=mode) as run:
            mlflow.log_params(
                {
                    "mode": mode,
                    "seed": c.SEED,
                    "features": ",".join(FEATURES[i] for i in columns),
                    "train_rows": int(tr.sum()),
                    "pairs_sha256": fingerprint()["pairs"],
                    "backbones": "frozen pretrained",
                }
            )
            mlflow.log_metrics(
                {k: v for k, v in result.items() if k != "model" and v is not None}
            )
            mlflow.log_artifact(str(c.ART / (mode + ".joblib")), "model_bundle")
            mlflow.log_artifact(str(c.REPORTS / "data_audit.json"), "data")
            result["mlflow_run_id"] = run.info.run_id
        results.append(result)
    table = pd.DataFrame(results).sort_values(
        "average_precision", ascending=False, kind="stable"
    )
    winner = table.iloc[0]["model"]
    joblib.dump(bundles[winner], c.ART / "champion.joblib")
    table.to_csv(c.REPORTS / "validation_models.csv", index=False)
    selection = {
        "winner": winner,
        "selected_on": "validation average precision",
        "test_used_for_selection": False,
        "policy": bundles[winner]["policy"],
        "fingerprint": fingerprint(),
        "model_sha256": sha(c.ART / "champion.joblib"),
        "versions": versions(),
        "python": platform.python_version(),
        "lexical_smoke_only": lexical_only,
    }
    dump(c.ART / "selection.json", selection)
    return selection


def load_champion():
    b = joblib.load(c.ART / "champion.joblib")
    if b["fingerprint"] != fingerprint():
        raise ValueError("Stale model: rebuild artifacts after dataset changes")
    return b


def component_bootstrap(frame, scores, baseline, repetitions=300):
    uf = UnionFind(set(frame.left_group) | set(frame.right_group))
    for r in frame.itertuples():
        uf.union(r.left_group, r.right_group)
    clusters = {}
    for i, r in enumerate(frame.itertuples()):
        clusters.setdefault(uf.find(r.left_group), []).append(i)
    groups = list(clusters.values())
    if len(groups) < 10:
        return {
            "ci95": None,
            "independent_pair_graph_components": len(groups),
            "reason": "Too few independent components; no pair-iid interval reported.",
        }
    rng = np.random.default_rng(c.SEED)
    values = []
    y = frame.label.to_numpy()
    for _ in range(repetitions):
        ix = np.concatenate(
            [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        )
        if len(np.unique(y[ix])) < 2:
            continue
        values.append(
            average_precision_score(y[ix], scores[ix])
            - average_precision_score(y[ix], baseline[ix])
        )
    return {
        "ci95": np.quantile(values, [0.025, 0.975]).tolist() if values else None,
        "independent_pair_graph_components": len(groups),
        "repetitions": len(values),
    }


def evaluate():
    _, pairs = load()
    b = load_champion()
    store = FeatureStore(neural=b["neural"])
    test = pairs[pairs.split == "test"].reset_index(drop=True)
    x = store.pairs(test)
    scores = predict_scores(b, x)
    base = joblib.load(c.ART / "lexical_lr.joblib")
    bs = predict_scores(base, x)
    out = {
        "winner": b["mode"],
        "fingerprint": b["fingerprint"],
        "model_sha256": sha(c.ART / "champion.joblib"),
        "test": metrics(test.label.to_numpy(), scores, b["policy"]["f1_threshold"]),
        "lexical_baseline": metrics(
            test.label.to_numpy(), bs, base["policy"]["f1_threshold"]
        ),
        "paired_component_bootstrap_ap_delta": component_bootstrap(test, scores, bs),
        "scope": "Offline sampled labelled pairs, custom grouped split. No full-catalog accuracy or business savings claim.",
    }
    threshold = b["policy"]["suggestion_threshold"]
    selected = (
        scores >= threshold
        if threshold is not None
        else np.zeros(len(scores), dtype=bool)
    )
    out["suggestions"] = {
        "threshold": threshold,
        "pairs": int(selected.sum()),
        "coverage": float(selected.mean()),
        "precision": (
            float(test.label.to_numpy()[selected].mean()) if selected.any() else None
        ),
    }
    if b["neural"]:
        out["stress_tests"] = {
            f"zero_{mask}_features": metrics(
                test.label.to_numpy(),
                predict_scores(b, store.pairs(test, mask=mask)),
                b["policy"]["f1_threshold"],
            )
            for mask in ["image", "text"]
        }
    pred = test.copy()
    pred["score"] = scores
    pred["baseline_score"] = bs
    pred["prediction"] = (scores >= b["policy"]["f1_threshold"]).astype(int)
    pred["error"] = pred.prediction != pred.label
    pred.to_csv(c.REPORTS / "test_predictions.csv", index=False)
    q = test[["pair_id", "left_id", "right_id"]].copy()
    q["score"] = scores
    q["suggestion"] = np.where(selected, "likely_match", "needs_review")
    q["uncertainty"] = abs(scores - b["policy"]["f1_threshold"])
    q["model_sha256"] = out["model_sha256"]
    q.sort_values("uncertainty").to_csv(c.REPORTS / "review_queue.csv", index=False)
    dump(c.REPORTS / "test_metrics.json", out)
    mlflow = tracking()
    with mlflow.start_run(run_name="locked-test-evaluation"):
        mlflow.log_params({"champion": b["mode"], "model_sha256": out["model_sha256"]})
        mlflow.log_metrics(
            {"test_" + k: v for k, v in out["test"].items() if v is not None}
        )
        mlflow.log_artifact(str(c.REPORTS / "test_metrics.json"))
    return out
