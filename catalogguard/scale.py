"""ABO metadata ingestion scale test, not a labelled duplicate benchmark."""

import gzip, json, tarfile, time, hashlib
from . import config as c
from .data import download
from .store import database, upsert_many
from .util import dump, sha

URL = "https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-listings.tar"


def document_key(row):
    return hashlib.sha256(
        json.dumps([row.get("domain_name", ""), row["item_id"]]).encode()
    ).hexdigest()


def run(limit=0):
    c.directories()
    archive = c.DATA / "raw/abo-listings.tar"
    download(URL, archive, expected_size=87480320)
    db = database()
    collection = db.abo_listings
    collection.create_index("product_type.value")
    collection.create_index("brand.value")
    count = 0
    pending = []
    start = time.perf_counter()
    stop = False
    with tarfile.open(archive) as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".json.gz"):
                continue
            with gzip.GzipFile(fileobj=tar.extractfile(member)) as lines:
                for line in lines:
                    row = json.loads(line)
                    if not row.get("item_id"):
                        raise ValueError("Missing ABO item_id")
                    pending.append({"_id": document_key(row), "source": "ABO", **row})
                    count += 1
                    if len(pending) == 500:
                        upsert_many(collection, pending)
                        pending = []
                    if limit and count >= limit:
                        stop = True
                        break
            if stop:
                break
    if pending:
        upsert_many(collection, pending)
    elapsed = time.perf_counter() - start
    if count == 0:
        raise ValueError("No metadata found")
    top = list(
        collection.aggregate(
            [
                {"$unwind": "$product_type"},
                {"$group": {"_id": "$product_type.value", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 10},
            ]
        )
    )
    out = {
        "source_url": URL,
        "archive_sha256": sha(archive),
        "rows_read_this_run": count,
        "unique_documents_in_collection": collection.count_documents({}),
        "document_key": "SHA256(domain_name, item_id); preserves market-specific listings",
        "ingestion_seconds_excluding_download": elapsed,
        "rows_per_second": count / elapsed,
        "top_types_aggregation": top,
        "limit": limit,
        "scope": "Metadata ingestion and MongoDB aggregation only. No ABO images embedded; no ABO matching accuracy measured.",
    }
    dump(c.REPORTS / "abo_scale.json", out)
    return out
