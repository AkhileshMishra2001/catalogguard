import hmac, os, time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
import pandas as pd
from fastapi import FastAPI, Depends, Header, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from . import config as c
from .store import database, feedback
from .features import FeatureStore, FEATURES
from .modeling import load_champion, predict_scores
from .util import read, sha

REQUESTS = Counter("catalogguard_http_requests_total", "HTTP requests", ["status"])
LATENCY = Histogram(
    "catalogguard_http_duration_seconds",
    "HTTP duration seconds",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)
PREDICT = Counter("catalogguard_pairs_scored_total", "Pairs scored")
REVIEWS = Counter("catalogguard_reviews_total", "Review decisions", ["decision"])


@asynccontextmanager
async def lifespan(app):
    if len(os.getenv("API_KEY", "")) < 24:
        raise RuntimeError("Run scripts/setup_env.py to create credentials")
    app.state.db = database()
    app.state.bundle = load_champion()
    app.state.store = FeatureStore(neural=app.state.bundle["neural"])
    app.state.model_sha = sha(c.ART / "champion.joblib")
    yield
    app.state.db.client.close()


app = FastAPI(
    title="CatalogGuard",
    version="1.0.0",
    lifespan=lifespan,
    description="Catalog-ID pair scoring using precomputed embeddings. Human review; no automatic merge.",
)
app.mount(
    "/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static"
)


@app.middleware("http")
async def instrumentation(request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
        REQUESTS.labels(str(response.status_code)).inc()
        return response
    except Exception:
        REQUESTS.labels("500").inc()
        raise
    finally:
        LATENCY.observe(time.perf_counter() - start)


def auth(x_api_key: str = Header(default="")):
    key = os.getenv("API_KEY", "")
    if not key or not hmac.compare_digest(x_api_key.encode(), key.encode()):
        raise HTTPException(401, "Valid X-API-Key required")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(Path(__file__).parent / "static/index.html")


@app.get("/health/live")
def live():
    return {"status": "alive"}


@app.get("/health/ready")
def ready():
    try:
        app.state.db.command("ping")
    except Exception:
        raise HTTPException(503, "Database unavailable")
    return {"status": "ready", "model": app.state.bundle["mode"]}


@app.get("/metrics", include_in_schema=False)
def prometheus():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/summary", dependencies=[Depends(auth)])
def summary():
    return {
        "audit": read(c.REPORTS / "data_audit.json"),
        "selection": read(c.ART / "selection.json"),
        "test": (
            read(c.REPORTS / "test_metrics.json")
            if (c.REPORTS / "test_metrics.json").exists()
            else None
        ),
        "validation": pd.read_csv(c.REPORTS / "validation_models.csv").to_dict(
            "records"
        ),
    }


@app.get("/api/queue", dependencies=[Depends(auth)])
def queue(offset: int = Query(0, ge=0), limit: int = Query(12, ge=1, le=50)):
    db = app.state.db
    filt = {
        "dataset_fingerprint": app.state.bundle["fingerprint"]["pairs"],
        "model_sha256": app.state.model_sha,
    }
    try:
        rows = list(
            db.review_queue.find(filt, {"_id": 0})
            .sort([("uncertainty", 1), ("pair_id", 1)])
            .skip(offset)
            .limit(limit)
        )
        for r in rows:
            for side in ["left", "right"]:
                listing = db.listings.find_one({"_id": r[side + "_id"]})
                r[side] = (
                    {"listing_id": listing["listing_id"], "title": listing["title"]}
                    if listing
                    else None
                )
            review = db.feedback.find_one(
                {"pair_id": r["pair_id"], "model_sha256": app.state.model_sha},
                sort=[("created_at", -1)],
            )
            r["reviewed_as"] = review["decision"] if review else None
        return {"items": rows, "total": db.review_queue.count_documents(filt)}
    except Exception:
        raise HTTPException(503, "Database query failed")


@app.get("/api/image/{listing_id}")
def image(listing_id: str):
    p = app.state.store.products
    if listing_id not in p.index:
        raise HTTPException(404, "Unknown listing")
    target = (c.DATA / p.loc[listing_id, "image_path"]).resolve()
    if not target.is_relative_to((c.DATA / "wdc_shoes_images").resolve()):
        raise HTTPException(400, "Invalid path")
    return FileResponse(target)


class PairRequest(BaseModel):
    left_id: str = Field(min_length=1, max_length=40)
    right_id: str = Field(min_length=1, max_length=40)


@app.post("/api/score", dependencies=[Depends(auth)])
def score(pair: PairRequest):
    s = app.state.store
    b = app.state.bundle
    if pair.left_id == pair.right_id:
        raise HTTPException(422, "Choose different listings")
    if any(k not in s.positions for k in [pair.left_id, pair.right_id]):
        raise HTTPException(404, "Unknown catalog listing")
    x = s.pairs(pd.DataFrame([pair.model_dump()]))
    value = float(predict_scores(b, x)[0])
    PREDICT.inc()
    threshold = b["policy"]["suggestion_threshold"]
    return {
        "score": value,
        "suggestion": (
            "likely_match"
            if threshold is not None and value >= threshold
            else "needs_review"
        ),
        "features": dict(zip(FEATURES, map(float, x[0]))),
        "model": b["mode"],
        "note": "Uncalibrated model score; feature signals are not causal explanations.",
    }


class ReviewRequest(BaseModel):
    pair_id: str = Field(min_length=1, max_length=40)
    decision: Literal["match", "different", "unsure"]
    note: str = Field(default="", max_length=1000)


@app.post("/api/review", dependencies=[Depends(auth)])
def review(body: ReviewRequest):
    db = app.state.db
    filt = {
        "_id": body.pair_id,
        "model_sha256": app.state.model_sha,
        "dataset_fingerprint": app.state.bundle["fingerprint"]["pairs"],
    }
    try:
        if db.review_queue.find_one(filt) is None:
            raise HTTPException(404, "Unknown current-model pair")
        feedback(db, body.pair_id, body.decision, body.note, app.state.model_sha)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(503, "Review not saved; database unavailable")
    REVIEWS.labels(body.decision).inc()
    return {"saved": True, "decision": body.decision}


@app.get("/api/candidates/{listing_id}", dependencies=[Depends(auth)])
def find_candidates(listing_id: str, k: int = Query(10, ge=1, le=30)):
    from .retrieval import candidates

    s = app.state.store
    if s.text is None:
        raise HTTPException(409, "Neural embeddings required")
    if listing_id not in s.positions:
        raise HTTPException(404, "Unknown listing")
    return {
        "items": [
            {
                "listing_id": key,
                "title": s.products.loc[key, "title"],
                "retrieval_similarity": value,
            }
            for key, value in candidates(listing_id, s.ids, s.text, s.image, k)
        ],
        "note": "Retrieval similarity is not a duplicate probability.",
    }
