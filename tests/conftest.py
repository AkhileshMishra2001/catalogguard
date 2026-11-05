import numpy as np
import pandas as pd
import pytest
from PIL import Image
from catalogguard import config as c
from catalogguard.util import dump


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    for name, path in [
        ("ROOT", tmp_path),
        ("DATA", tmp_path / "data"),
        ("ART", tmp_path / "artifacts"),
        ("REPORTS", tmp_path / "reports"),
    ]:
        monkeypatch.setattr(c, name, path)
    c.directories()
    products = []
    pairs = []
    rng = np.random.default_rng(42)
    for split in ["train", "validation", "test"]:
        for g in range(25):
            group = split + str(g)
            for j in range(2):
                key = f"{group}_{j}"
                path = f"wdc_shoes_images/{key}.png"
                (c.DATA / path).parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 8), (g * 5, j * 30, 100)).save(c.DATA / path)
                products.append(
                    {
                        "listing_id": key,
                        "title": f"brand shoe {g} style {j}",
                        "image_path": path,
                        "split": split,
                        "group_id": group,
                        "source": "synthetic",
                    }
                )
            pairs.append(
                {
                    "pair_id": group + "pos",
                    "left_id": group + "_0",
                    "right_id": group + "_1",
                    "label": 1,
                    "split": split,
                    "left_group": group,
                    "right_group": group,
                }
            )
            other = split + str((g + 1) % 25)
            pairs.append(
                {
                    "pair_id": group + "neg",
                    "left_id": group + "_0",
                    "right_id": other + "_1",
                    "label": 0,
                    "split": split,
                    "left_group": group,
                    "right_group": other,
                }
            )
    p = pd.DataFrame(products)
    q = pd.DataFrame(pairs)
    p.to_csv(c.DATA / "processed/products.csv", index=False)
    q.to_csv(c.DATA / "processed/pairs.csv", index=False)
    from catalogguard.features import fingerprint

    vec = rng.normal(size=(len(products), 12)).astype("float32")
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    np.savez_compressed(
        c.ART / "embeddings.npz",
        ids=p.listing_id.to_numpy(dtype=str),
        text=vec,
        image=vec,
    )
    dump(c.ART / "embedding_manifest.json", {"fingerprint": fingerprint()})
    dump(c.REPORTS / "data_audit.json", {"synthetic": True, "unique_listings": len(p)})
    return p, q
