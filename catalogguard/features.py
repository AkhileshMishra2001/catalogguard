import re, joblib
import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from . import config as c
from .data import load
from .util import dump, read, sha

FEATURES = [
    "tfidf_cosine",
    "token_jaccard",
    "length_ratio",
    "digit_jaccard",
    "both_have_digits",
    "text_cosine",
    "image_cosine",
    "text_image_disagreement",
]
MODES = {
    "lexical_lr": [0, 1, 2, 3, 4],
    "text_lr": [0, 1, 2, 3, 4, 5],
    "vision_lr": [6],
    "fusion_lr": list(range(8)),
    "fusion_lgbm": list(range(8)),
}


def fingerprint():
    return {
        "products": sha(c.DATA / "processed/products.csv"),
        "pairs": sha(c.DATA / "processed/pairs.csv"),
        "text_model": c.TEXT_MODEL,
        "text_revision": c.TEXT_REV,
        "image_model": c.IMAGE_MODEL,
        "image_revision": c.IMAGE_REV,
    }


def embed(batch_size=16, device="cpu"):
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import CLIPModel, CLIPProcessor
    from tqdm import tqdm

    products, _ = load()
    c.directories()
    cache = c.ART / "embeddings.npz"
    if (
        cache.exists()
        and (c.ART / "embedding_manifest.json").exists()
        and read(c.ART / "embedding_manifest.json")["fingerprint"] == fingerprint()
    ):
        return {"cached": True, "listings": len(products)}
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    model = SentenceTransformer(c.TEXT_MODEL, revision=c.TEXT_REV, device=device)
    text = model.encode(
        products.title.tolist(),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")
    del model
    model = (
        CLIPModel.from_pretrained(
            c.IMAGE_MODEL,
            revision=c.IMAGE_REV,
            use_safetensors=False,
            weights_only=True,
        )
        .to(device)
        .eval()
    )
    processor = CLIPProcessor.from_pretrained(c.IMAGE_MODEL, revision=c.IMAGE_REV)
    paths = products.image_path.drop_duplicates().tolist()
    outputs = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(paths), batch_size), desc="CLIP image batches"):
            images = []
            for path in paths[start : start + batch_size]:
                with Image.open(c.DATA / path) as im:
                    images.append(im.convert("RGB"))
            inputs = processor(images=images, return_tensors="pt").to(device)
            vectors = model.get_image_features(**inputs)
            vectors = vectors / vectors.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            outputs.append(vectors.cpu().numpy().astype("float32"))
    vec = np.concatenate(outputs)
    index = {p: i for i, p in enumerate(paths)}
    image = vec[[index[p] for p in products.image_path]]
    np.savez_compressed(
        cache, ids=products.listing_id.to_numpy(dtype=str), text=text, image=image
    )
    manifest = {
        "fingerprint": fingerprint(),
        "listings_encoded": len(products),
        "unique_images_encoded": len(paths),
        "text_dimensions": text.shape[1],
        "image_dimensions": image.shape[1],
        "backbones": "Frozen pretrained encoders; no neural fine-tuning",
    }
    dump(c.ART / "embedding_manifest.json", manifest)
    return manifest


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 0.0


class FeatureStore:
    def __init__(self, fit=False, neural=True):
        products, _ = load()
        self.products = products.set_index("listing_id", drop=False)
        self.ids = products.listing_id.tolist()
        self.positions = {k: i for i, k in enumerate(self.ids)}
        self.titles = products.title.tolist()
        self.tokens = [set(re.findall(r"\w+", t.lower())) for t in self.titles]
        self.digits = [set(re.findall(r"\d+", t)) for t in self.titles]
        if fit:
            self.tfidf = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                max_features=30000,
                sublinear_tf=True,
            )
            self.tfidf.fit(products[products.split == "train"].title)
            joblib.dump(self.tfidf, c.ART / "tfidf.joblib")
        else:
            self.tfidf = joblib.load(c.ART / "tfidf.joblib")
        self.lexical = self.tfidf.transform(self.titles)
        self.text = self.image = None
        if neural:
            if read(c.ART / "embedding_manifest.json")["fingerprint"] != fingerprint():
                raise ValueError("Stale embeddings: rerun embed")
            with np.load(c.ART / "embeddings.npz", allow_pickle=False) as z:
                if z["ids"].tolist() != self.ids:
                    raise ValueError("Embedding order mismatch")
                self.text, self.image = z["text"], z["image"]

    def pairs(self, frame, mask=None):
        left = np.array([self.positions[k] for k in frame.left_id])
        right = np.array([self.positions[k] for k in frame.right_id])
        x = np.zeros((len(frame), 8), dtype="float32")
        x[:, 0] = np.asarray(
            self.lexical[left].multiply(self.lexical[right]).sum(axis=1)
        ).ravel()
        for row, (i, j) in enumerate(zip(left, right)):
            x[row, 1] = jaccard(self.tokens[i], self.tokens[j])
            x[row, 2] = min(len(self.titles[i]), len(self.titles[j])) / max(
                len(self.titles[i]), len(self.titles[j]), 1
            )
            x[row, 3] = jaccard(self.digits[i], self.digits[j])
            x[row, 4] = bool(self.digits[i] and self.digits[j])
        if self.text is not None:
            x[:, 5] = (self.text[left] * self.text[right]).sum(axis=1)
        if self.image is not None:
            x[:, 6] = (self.image[left] * self.image[right]).sum(axis=1)
        if mask == "image":
            x[:, 6] = 0
        if mask == "text":
            x[:, :6] = 0
        x[:, 7] = abs(x[:, 5] - x[:, 6])
        return x
