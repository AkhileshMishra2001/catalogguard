# Explain CatalogGuard

## Problem
Seller titles and photos differ even for the same product. Score likely duplicate pairs and support human review, while avoiding incorrect merges of similar-looking products. The benchmark's product-identity labels do not certify every size/colour variant.

## Why this stack
MongoDB fits nested, variable product metadata, image references and review history; use indexed queries, bulk upserts and product-type aggregation. Character TF-IDF handles spelling variations cheaply. Multilingual MiniLM represents semantics across titles/languages. CLIP supplies visual embeddings. Logistic regression is an interpretable fusion baseline; LightGBM models nonlinear interactions. Deep encoders are frozen, classifiers are trained. MLflow records experiments, parameters and artifacts. Prometheus monitors service behaviour, not prediction accuracy. FastAPI serves existing catalog IDs and validates requests.

## Features
TF-IDF cosine, token Jaccard, title-length ratio, digit-token Jaccard, both-have-digits flag, text cosine, image cosine, absolute text/image disagreement. Digits are not guaranteed model numbers or shoe sizes; there is no structured variant extractor.

## Evaluation
1. Verify pinned downloads and all referenced image bytes.
2. Deduplicate unordered pairs and exclude contradictory labels.
3. Union positive-connected listings, identical RGB pixels and exact normalised titles. Assign groups via fixed hash/seed to train/validation/test. Exclude pairs spanning partitions and negative pairs within identity groups. Report resulting prevalence changes.
4. Fit TF-IDF and classifiers on train only. Compare five configurations on validation average precision. Select thresholds on validation, not test.
5. Evaluate the winner against the lexical baseline on test. Report AP, ROC AUC, F1, precision/recall, Brier and suggestion coverage. At least 20 validation suggestions and empirical precision >=95% are needed for a suggestion threshold. This is not a guarantee on deployment precision.
6. Resample connected components of the evaluated pair graph for a paired AP-difference bootstrap. If fewer than 10 components exist, suppress the confidence interval. Shared products invalidate naive pair-independent intervals; small component counts still mean considerable uncertainty.
7. Test text-only, image-only and fusion variants. Zeroing modalities is a sensitivity stress test, not learned support for missing images.
8. Candidate retrieval reports coverage of known labelled neighbours, not exhaustive catalog recall.

## Limits
Observed labels/exact pixels/titles do not rule out all unlabelled identity overlap, near-duplicates or pretrained-data contamination. The custom split is not directly comparable to paper benchmarks. Historical sampled-pair prevalence differs from marketplace prevalence. No online experiment, sales lift or saved labour has been measured.

The API uses cached embeddings for existing catalog IDs. Latency excludes new image encoding. New catalog ingestion requires validation and embedding first. WDC retrieval is brute-force over a small catalog. ABO is only metadata ingestion/aggregation; no large-scale visual search or duplicate ground truth is claimed on it.

## Why not add more tools?
A coherent experiment matters more than tool count. No extra LLM/agent is necessary for deterministic pair scoring. MongoDB is not the vector search engine here. Active learning is a possible future use of reviewer data, but this release does not implement retraining from feedback.

## Production discussion
Containerisation and health/auth checks demonstrate engineering readiness. Real shared production still needs TLS, individual identities, authorisation, rate limiting, secret management, tested backups, operational ownership, load testing and monitoring against fresh independent labels. Offline benchmark metrics are not production SLAs.
