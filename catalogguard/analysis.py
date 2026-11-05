from . import config as c
from .data import load
from .util import dump, read


def analyze():
    p, _ = load()
    p.assign(title_characters=p.title.str.len()).groupby("split").agg(
        listings=("listing_id", "size"),
        mean_title_characters=("title_characters", "mean"),
        unique_images=("image_path", "nunique"),
    ).to_csv(c.REPORTS / "catalog_profile.csv")
    out = {
        "questions": [
            "How frequent are matches after grouped splitting?",
            "Does image evidence improve validation AP over text-only models?",
            "What fraction can be suggested at the validation precision target?",
        ],
        "data_audit": read(c.REPORTS / "data_audit.json"),
        "notes": [
            "Missing or corrupt images fail preparation.",
            "No seller IDs, sales outcomes or verified variant attributes are invented.",
            "Benchmark prevalence is not live marketplace prevalence.",
        ],
    }
    dump(c.REPORTS / "analysis.json", out)
    return out
