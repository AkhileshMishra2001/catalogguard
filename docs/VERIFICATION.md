# Release verification

The packaged implementation was verified in a Linux / Python 3.12 environment. Windows instructions target Python 3.11.

## Passed
- 12 automated unit/API tests; two upstream deprecation warnings, no failures.
- Downloaded and checksum-verified the real WDC Shoes source; decoded all 1,453 referenced images.
- Generated real pinned MiniLM text and CLIP image embeddings; trained and compared five model configurations, then evaluated the validation-selected winner on 644 held-out pairs.
- Completed retrieval, latency and data-analysis commands. Aggregate reference reports are in reference_results.
- Native MongoDB 7.0.16 authenticated integration: synced listings and pairs, persisted reviewer decisions, and read them back.
- Ingested all 147,702 ABO metadata rows as 147,702 documents using market/domain plus item ID identity; ran aggregation queries.
- Exercised API authentication, queue label exclusion, pair scoring, candidate retrieval and metrics.
- Chromium desktop (1440 px) and mobile (390 px) review flows passed without JavaScript errors or horizontal overflow; visually reviewed desktop presentation.
- Container/monitoring YAML syntax checked.

## Local acceptance still required
Docker Desktop and a Docker daemon were unavailable in the verification environment. Docker image build, Compose networking, container permissions, Prometheus target scraping and Windows compatibility must be confirmed using README steps 1–9 and DEPLOYMENT.md. Native service testing does not replace these checks.

This is a locally deployable portfolio implementation, not a publicly operated production service. No cloud deployment, business savings, auto-merge safety guarantee or ABO matching accuracy was measured.
