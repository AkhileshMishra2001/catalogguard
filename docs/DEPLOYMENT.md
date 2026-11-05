# Local deployment acceptance

Included: dedicated MongoDB application user, API-key checks, bounded inputs, whitelist-based catalog image lookup, append-only feedback, model/data hashes, non-root app container, read-only data mounts, readiness/liveness, persistent volumes, MLflow tracking, Prometheus service metrics/alert rules and CI tests.

All published ports bind to 127.0.0.1. Static UI, public-data images, health and metrics are unauthenticated. Other API routes require X-API-Key. No CORS wildcard. The browser stores the key only in sessionStorage. Use this local configuration on your own machine.

## Checks after training
1. Stop any local uvicorn process, then `docker compose --profile serve up -d --build`.
2. `docker compose ps` should show app and mongo healthy.
3. `Invoke-RestMethod http://127.0.0.1:8000/health/ready` should return ready.
4. Connect in the UI with your API_KEY. Save a decision, refresh, and confirm persistence.
5. Restart `docker compose restart app`, then confirm the decision remains.
6. Confirm unauthenticated `/api/summary` returns 401.
7. Open http://127.0.0.1:9090/targets and confirm catalogguard is UP.
8. Click Inspect signals and query `catalogguard_pairs_scored_total`.
9. After generating traffic, query `histogram_quantile(0.95, sum by (le) (rate(catalogguard_http_duration_seconds_bucket[5m])))`.

Prometheus alerts appear in its UI. No email/Slack notification receiver is configured. Service metrics do not establish model quality. Fresh independently labelled data is needed; reviewer-selected uncertain pairs are biased evaluation samples.

## Operation
`docker compose --profile serve stop` preserves volumes; `up -d` restarts. Do not use `down -v` unless intentionally deleting database/reviews. Do not regenerate .env passwords after database initialisation. Stop app before retraining. Keep old processed data and artifacts together for rollback; load/sync/restart the old complete bundle if needed. Feedback stays tied to the original model hash.

## Before public production exposure
Add TLS/reverse proxy, individual user authentication and authorisation, rate limits, managed secrets, dependency vulnerability updates, backup/restore tests, load tests, deployment rollback and data-licence approval. There is no availability SLA, autoscaling or actual customer rollout in this project. Describe it as a containerised local service after acceptance succeeds.
