import io, pickle
import numpy as np
import pandas as pd
import pytest
from catalogguard.data import PrimitiveUnpickler, split_records
from catalogguard.features import FeatureStore
from catalogguard.modeling import thresholds, component_bootstrap


def test_restricted_pickle():
    class Attack:
        def __reduce__(self):
            return (eval, ("1+1",))

    with pytest.raises(pickle.UnpicklingError):
        PrimitiveUnpickler(io.BytesIO(pickle.dumps(Attack()))).load()
    obj = ({"labels": [0, 1]},)
    assert PrimitiveUnpickler(io.BytesIO(pickle.dumps(obj))).load() == obj


def test_group_split():
    products = []
    pairs = []
    for g in range(160):
        for j in range(2):
            products.append(
                {
                    "listing_id": f"{g}_{j}",
                    "pixel_hash": f"p{g}_{j}",
                    "normalized_title": f"t{g}_{j}",
                }
            )
        pairs.append({"left_id": f"{g}_0", "right_id": f"{g}_1", "label": 1})
        for other in [(g + 1) % 160, (g + 2) % 160, (g + 3) % 160]:
            pairs.append({"left_id": f"{g}_0", "right_id": f"{other}_1", "label": 0})
    products, pairs, audit = split_records(pd.DataFrame(products), pd.DataFrame(pairs))
    lookup = products.set_index("listing_id").split
    assert products.groupby("group_id").split.nunique().max() == 1
    assert (pairs.left_id.map(lookup) == pairs.right_id.map(lookup)).all()
    assert audit["cross_partition_pairs_excluded"] > 0


def test_features_symmetric(workspace):
    _, p = workspace
    s = FeatureStore(fit=True)
    a = s.pairs(p)
    b = s.pairs(p.rename(columns={"left_id": "right_id", "right_id": "left_id"}))
    assert np.allclose(a, b) and np.isfinite(a).all()
    assert np.all(s.pairs(p, mask="image")[:, 6] == 0)


def test_train_only_vectorizer(workspace):
    from catalogguard import config as c

    p, _ = workspace
    p.loc[p.split == "test", "title"] = "zzzzunseentoken"
    p.to_csv(c.DATA / "processed/products.csv", index=False)
    assert "zzz" not in FeatureStore(fit=True, neural=False).tfidf.vocabulary_


def test_stale_embedding_guard(workspace):
    from catalogguard import config as c

    p = c.DATA / "processed/pairs.csv"
    p.write_text(p.read_text() + "\n")
    with pytest.raises(ValueError, match="Stale"):
        FeatureStore(fit=True)


def test_small_sample_policy():
    assert (
        thresholds(np.array([1, 1, 0]), np.array([0.99, 0.98, 0.1]))[
            "suggestion_threshold"
        ]
        is None
    )


def test_no_pair_independence_assumption(workspace):
    _, p = workspace
    q = p[p.split == "test"].reset_index(drop=True)
    assert (
        component_bootstrap(q, np.repeat(0.6, len(q)), np.repeat(0.5, len(q)))["ci95"]
        is None
    )


def test_all_models_and_label_free_queue(workspace):
    from catalogguard.modeling import train, evaluate
    from catalogguard import config as c

    assert train()["test_used_for_selection"] is False
    assert evaluate()["test"]["rows"] == 50
    assert len(pd.read_csv(c.REPORTS / "validation_models.csv")) == 5
    assert "label" not in pd.read_csv(c.REPORTS / "review_queue.csv")


def test_abo_market_key():
    from catalogguard.scale import document_key

    assert document_key({"item_id": "A", "domain_name": "amazon.com"}) != document_key(
        {"item_id": "A", "domain_name": "amazon.co.uk"}
    )
