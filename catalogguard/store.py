import os
from datetime import datetime, timezone
from urllib.parse import quote_plus
import pandas as pd
from pymongo import MongoClient, ReplaceOne
from . import config as c
from .data import load
from .util import read


def database():
    uri = os.getenv("MONGO_URI")
    password = os.getenv("MONGO_APP_PASSWORD")
    if not uri:
        if not password:
            raise RuntimeError("Run scripts/setup_env.py first")
        uri = f"mongodb://catalogguard:{quote_plus(password)}@{os.getenv('MONGO_HOST','localhost')}:27017/catalogguard?authSource=catalogguard"
    client = MongoClient(uri, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
    client.admin.command("ping")
    return client.catalogguard


def upsert_many(collection, docs):
    for start in range(0, len(docs), 500):
        collection.bulk_write(
            [
                ReplaceOne({"_id": d["_id"]}, d, upsert=True)
                for d in docs[start : start + 500]
            ],
            ordered=False,
        )


def sync():
    db = database()
    products, pairs = load()
    db.listings.create_index([("source", 1), ("split", 1)])
    upsert_many(
        db.listings,
        [{"_id": r["listing_id"], **r} for r in products.to_dict("records")],
    )
    db.benchmark_pairs.create_index("split")
    upsert_many(
        db.benchmark_pairs,
        [{"_id": r["pair_id"], **r} for r in pairs.to_dict("records")],
    )
    db.review_queue.create_index(
        [("dataset_fingerprint", 1), ("model_sha256", 1), ("uncertainty", 1)]
    )
    db.feedback.create_index([("pair_id", 1), ("created_at", -1)])
    if (c.REPORTS / "review_queue.csv").exists():
        selection = read(c.ART / "selection.json")
        q = pd.read_csv(c.REPORTS / "review_queue.csv")
        upsert_many(
            db.review_queue,
            [
                {
                    "_id": r["pair_id"],
                    **r,
                    "dataset_fingerprint": selection["fingerprint"]["pairs"],
                }
                for r in q.to_dict("records")
            ],
        )
    db.runs.replace_one(
        {"_id": "current_dataset"},
        {"_id": "current_dataset", **read(c.REPORTS / "data_audit.json")},
        upsert=True,
    )
    return {
        "wdc_listings_upserted": len(products),
        "labelled_pairs_upserted": len(pairs),
        "mongodb": "connected",
    }


def feedback(db, pair_id, decision, note, model_sha):
    return db.feedback.insert_one(
        {
            "pair_id": pair_id,
            "decision": decision,
            "note": note,
            "model_sha256": model_sha,
            "created_at": datetime.now(timezone.utc),
            "purpose": "Human annotation; not silently added to benchmark or training",
        }
    ).inserted_id
