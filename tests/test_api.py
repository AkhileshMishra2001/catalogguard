import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(workspace, monkeypatch):
    import mongomock, pandas as pd
    from catalogguard.modeling import train, evaluate
    from catalogguard import api, config as c
    from catalogguard.util import read

    train(lexical_only=True)
    evaluate()
    db = mongomock.MongoClient().catalogguard
    p, _ = workspace
    db.listings.insert_many(
        [{"_id": r["listing_id"], **r} for r in p.to_dict("records")]
    )
    q = pd.read_csv(c.REPORTS / "review_queue.csv")
    s = read(c.ART / "selection.json")
    db.review_queue.insert_many(
        [
            {"_id": r["pair_id"], **r, "dataset_fingerprint": s["fingerprint"]["pairs"]}
            for r in q.to_dict("records")
        ]
    )
    monkeypatch.setattr(api, "database", lambda: db)
    monkeypatch.setenv("API_KEY", "a" * 32)
    with TestClient(api.app) as client:
        yield client


def test_auth(client):
    assert client.get("/api/summary").status_code == 401
    assert (
        client.get("/api/summary", headers={"X-API-Key": "a" * 32}).status_code == 200
    )


def test_feedback(client):
    h = {"X-API-Key": "a" * 32}
    r = client.get("/api/queue", headers=h).json()["items"][0]
    assert "label" not in r
    assert client.post(
        "/api/review", headers=h, json={"pair_id": r["pair_id"], "decision": "match"}
    ).json()["saved"]
    assert (
        client.get("/api/queue", headers=h).json()["items"][0]["reviewed_as"] == "match"
    )
    assert (
        client.post(
            "/api/review", headers=h, json={"pair_id": "absent", "decision": "match"}
        ).status_code
        == 404
    )


def test_scoring_and_validation(client):
    h = {"X-API-Key": "a" * 32}
    r = client.get("/api/queue", headers=h).json()["items"][0]
    assert client.get("/api/queue?limit=999", headers=h).status_code == 422
    assert client.get("/api/image/" + r["left_id"]).status_code == 200
    s = client.post(
        "/api/score",
        headers=h,
        json={"left_id": r["left_id"], "right_id": r["right_id"]},
    )
    assert s.status_code == 200 and 0 <= s.json()["score"] <= 1
    assert client.get("/metrics").status_code == 200
    assert client.get("/health/ready").status_code == 200
