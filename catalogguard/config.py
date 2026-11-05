import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(os.getenv("CATALOGGUARD_HOME", Path(__file__).resolve().parents[1]))
load_dotenv(ROOT / ".env")
DATA = ROOT / "data"
ART = ROOT / "artifacts"
REPORTS = ROOT / "reports"
SEED = 42
SOURCE_REV = "48a8e9959c4edae0a1dd321fe195e5280422b9a9"
TEXT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TEXT_REV = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
IMAGE_MODEL = "openai/clip-vit-base-patch32"
IMAGE_REV = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


def directories():
    for p in [DATA / "raw", DATA / "processed", ART, REPORTS]:
        p.mkdir(parents=True, exist_ok=True)
