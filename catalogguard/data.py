"""Pinned downloads, restricted pickle conversion, pixel audit and grouped splits."""

import hashlib, pickle, re, zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm
from . import config as c
from .util import sha, dump


class PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError("External pickle classes forbidden")

    def persistent_load(self, pid):
        raise pickle.UnpicklingError("Persistent references forbidden")


def download(url, target, expected_sha=None, expected_size=None):
    target = Path(target)
    if (
        target.exists()
        and (expected_sha is None or sha(target) == expected_sha)
        and (expected_size is None or target.stat().st_size == expected_size)
    ):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".part")
    with requests.get(url, stream=True, timeout=(30, 120)) as r:
        r.raise_for_status()
        with temp.open("wb") as f, tqdm(
            total=int(r.headers.get("content-length", 0)) or None,
            unit="B",
            unit_scale=True,
            desc=target.name,
        ) as bar:
            for chunk in r.iter_content(1024 * 1024):
                f.write(chunk)
                bar.update(len(chunk))
    if expected_size and temp.stat().st_size != expected_size:
        raise ValueError("Incomplete download")
    if expected_sha and sha(temp) != expected_sha:
        raise ValueError("Checksum mismatch")
    temp.replace(target)


def fetch():
    c.directories()
    manifest = {"source_commit": c.SOURCE_REV, "files": {}}
    for name in ["wdc_shoes_data.pickle", "wdc_shoes_images.zip"]:
        root = f"BstnGHub/Multimodal-Product-Matching/{c.SOURCE_REV}/datasets/{name}"
        response = requests.get("https://raw.githubusercontent.com/" + root, timeout=30)
        response.raise_for_status()
        checksum = re.search(r"oid sha256:([0-9a-f]{64})", response.text).group(1)
        size = int(re.search(r"size (\d+)", response.text).group(1))
        url = "https://media.githubusercontent.com/media/" + root
        download(url, c.DATA / "raw" / name, checksum, size)
        manifest["files"][name] = {"sha256": checksum, "bytes": size, "url": url}
    dump(c.DATA / "raw/source.json", manifest)
    return manifest


class UnionFind:
    def __init__(self, keys):
        self.p = {k: k for k in keys}

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def split_records(products, pairs):
    uf = UnionFind(products.listing_id)
    for column in ["pixel_hash", "normalized_title"]:
        for _, g in products.groupby(column):
            ids = g.listing_id.tolist()
            for key in ids[1:]:
                uf.union(ids[0], key)
    for r in pairs[pairs.label == 1].itertuples():
        uf.union(r.left_id, r.right_id)
    members = defaultdict(list)
    for key in products.listing_id:
        members[uf.find(key)].append(key)
    roots = {k: min(group) for group in members.values() for k in group}

    def assign(root):
        v = int(hashlib.sha256(f"{c.SEED}:{root}".encode()).hexdigest()[:8], 16) / 2**32
        return "train" if v < 0.6 else ("validation" if v < 0.8 else "test")

    products = products.copy()
    products["group_id"] = products.listing_id.map(roots)
    products["split"] = products.group_id.map(assign)
    lookup = products.set_index("listing_id").split
    pairs = pairs.copy()
    pairs["left_group"] = pairs.left_id.map(roots)
    pairs["right_group"] = pairs.right_id.map(roots)
    pairs["split"] = pairs.left_id.map(lookup)
    cross = pairs.split != pairs.right_id.map(lookup)
    inconsistent = (pairs.label == 0) & (pairs.left_group == pairs.right_group)
    kept = pairs[~cross & ~inconsistent].copy()
    counts = kept.groupby(["split", "label"]).size().unstack(fill_value=0)
    for split in ["train", "validation", "test"]:
        if split not in counts.index or any(
            counts.loc[split].get(label, 0) < 10 for label in [0, 1]
        ):
            raise ValueError(
                "Insufficient labelled rows after grouping. Inspect data; do not cherry-pick seed."
            )
    return (
        products,
        kept,
        {
            "cross_partition_pairs_excluded": int(cross.sum()),
            "negative_pairs_inside_identity_groups_excluded": int(inconsistent.sum()),
            "identity_groups": len(members),
            "largest_group_listings": max(map(len, members.values())),
        },
    )


def prepare():
    c.directories()
    with open(c.DATA / "raw/wdc_shoes_data.pickle", "rb") as f:
        splits = PrimitiveUnpickler(f).load()
    if not isinstance(splits, tuple) or len(splits) != 3:
        raise ValueError("Unexpected source format")
    products = {}
    rows = []
    for si, split in enumerate(splits):
        if (
            not len(split["text_pairs"])
            == len(split["image_pairs"])
            == len(split["labels"])
        ):
            raise ValueError("Unequal columns")
        for texts, paths, label in zip(
            split["text_pairs"], split["image_pairs"], split["labels"]
        ):
            if len(texts) != 2 or len(paths) != 2 or label not in (0, 1):
                raise ValueError("Invalid pair")
            ids = []
            for title, path in zip(texts, paths):
                path = str(PurePosixPath(path))
                if (
                    not path.startswith("wdc_shoes_images/")
                    or ".." in PurePosixPath(path).parts
                ):
                    raise ValueError("Unsafe path")
                key = hashlib.sha256((path + "\n" + title).encode()).hexdigest()[:20]
                ids.append(key)
                products[key] = {
                    "listing_id": key,
                    "title": title,
                    "image_path": path,
                    "normalized_title": " ".join(title.lower().split()),
                    "source": "wdc_shoes",
                }
            left, right = sorted(ids)
            if left != right:
                rows.append(
                    {
                        "pair_id": hashlib.sha256((left + right).encode()).hexdigest()[
                            :20
                        ],
                        "left_id": left,
                        "right_id": right,
                        "label": int(label),
                        "source_split_index": si,
                    }
                )
    raw = pd.DataFrame(rows)
    conflicts = raw.groupby("pair_id").label.nunique()
    bad = set(conflicts[conflicts > 1].index)
    pairs = raw[~raw.pair_id.isin(bad)].drop_duplicates("pair_id").copy()
    paths = sorted({p["image_path"] for p in products.values()})
    metadata = {}
    with zipfile.ZipFile(c.DATA / "raw/wdc_shoes_images.zip") as z:
        for path in tqdm(paths, desc="Validate referenced images"):
            info = z.getinfo(path)
            if info.file_size > 25_000_000:
                raise ValueError("Oversized image")
            target = c.DATA / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(info))
            with Image.open(target) as im:
                rgb = im.convert("RGB")
                rgb.load()
                metadata[path] = {
                    "pixel_hash": hashlib.sha256(
                        f"{rgb.size}".encode() + rgb.tobytes()
                    ).hexdigest(),
                    "width": rgb.width,
                    "height": rgb.height,
                }
    for p in products.values():
        p.update(metadata[p["image_path"]])
    products = (
        pd.DataFrame(products.values()).sort_values("listing_id").reset_index(drop=True)
    )
    sets = [
        set(raw[raw.source_split_index == i][["left_id", "right_id"]].values.ravel())
        for i in range(3)
    ]
    products, pairs, grouping = split_records(products, pairs)
    products.to_csv(c.DATA / "processed/products.csv", index=False)
    pairs.to_csv(c.DATA / "processed/pairs.csv", index=False)
    audit = {
        "original_pairs": sum(len(s["labels"]) for s in splits),
        "unique_nonself_pairs": raw.pair_id.nunique(),
        "unique_listings": len(products),
        "referenced_images_verified": len(paths),
        "conflicting_pair_ids_excluded": len(bad),
        "retained_pairs": len(pairs),
        "original_split_listing_overlap": {
            f"{a}_{b}": len(sets[a] & sets[b]) for a, b in [(0, 1), (0, 2), (1, 2)]
        },
        **grouping,
        "split_counts": {
            s: {
                "pairs": len(g),
                "matches": int(g.label.sum()),
                "match_rate": float(g.label.mean()),
            }
            for s, g in pairs.groupby("split")
        },
        "seed": c.SEED,
        "products_sha256": sha(c.DATA / "processed/products.csv"),
        "pairs_sha256": sha(c.DATA / "processed/pairs.csv"),
        "protocol": "Custom positive-component/pixel/title-group split; cross-partition pairs and inconsistent negatives excluded. Not official benchmark scores.",
    }
    dump(c.REPORTS / "data_audit.json", audit)
    return audit


def load():
    return pd.read_csv(
        c.DATA / "processed/products.csv", keep_default_na=False
    ), pd.read_csv(c.DATA / "processed/pairs.csv", keep_default_na=False)
