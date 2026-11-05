# CatalogGuard
## Multimodal product matching and catalog review

**Python · MongoDB · NLP · Computer Vision · LightGBM · MLflow · FastAPI · Docker · Prometheus**

Compare product descriptions and photographs to identify likely matches, avoid incorrect catalog merges, and route uncertain cases to human review. This is an independent public-data project, not a Cigna deployment. The release includes ingestion, analysis, five model comparisons, evaluation, a review UI, API, database persistence, container configuration, monitoring and tests.

### What is actually implemented
- Character TF-IDF, multilingual MiniLM text embeddings and CLIP ViT-B/32 image embeddings.
- Five configurations: lexical logistic regression, text logistic regression, image logistic regression, multimodal logistic regression, multimodal LightGBM.
- MongoDB indexed collections, bulk upserts, nested JSON metadata and aggregation pipelines.
- MLflow experiment tracking, model artifacts and dataset fingerprints.
- FastAPI pair scoring, candidate retrieval, API-key checks, bounded inputs and append-only reviewer decisions.
- Prometheus HTTP counts, latency and review metrics with service alert rules. No Evidently dependency.
- Non-root Docker application, persistent MongoDB volume and GitHub Actions tests.

### Important boundaries
WDC Shoes is the labelled matching benchmark. ABO is a SEPARATE metadata ingestion/aggregation scale test. Do not combine their counts into a matching training-set claim. The model scores existing catalog IDs using cached embeddings; it does not accept arbitrary uploaded images or automatically merge products. Labels reflect the benchmark's notion of product identity, not a certified size/colour variant match. No financial savings or sales lift is measured.

## Windows / VS Code setup
Use Python **3.11, 64-bit**, VS Code, and Docker Desktop running Linux containers (WSL2). Keep about 10–12 GB disk space free for packages, images, model caches and containers. 16 GB RAM is comfortable. CPU is the default; a GPU is not required.

Extract the ZIP to `C:\Users\Hp\Desktop\CatalogGuard`. Open the folder containing `README.md`, `requirements.txt` and `catalogguard` in VS Code. Open Terminal → New Terminal → PowerShell. Run each block from this folder. You do not need to activate the virtual environment.

### 1. Install dependencies
```powershell
py -3.11 --version
docker --version
docker compose version
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/setup_env.py
.\.venv\Scripts\python.exe -m pytest -q
```
In VS Code select `.venv\Scripts\python.exe` as interpreter. The setup script creates local credentials in `.env` and will not overwrite an existing file. Never commit or share it.

### 2. Start MongoDB
Start Docker Desktop first, then:
```powershell
docker compose up -d --wait mongo
```
This creates an authenticated local database and a dedicated readWrite application user. Do not regenerate `.env` after database initialisation: the existing volume retains the original passwords.

### 3. Download, validate and analyse
```powershell
.\.venv\Scripts\python.exe -m catalogguard.cli download
.\.venv\Scripts\python.exe -m catalogguard.cli prepare
.\.venv\Scripts\python.exe -m catalogguard.cli analyze
.\.venv\Scripts\python.exe -m catalogguard.cli sync
```
About 581 MB is downloaded. Checksums are verified. Every referenced image is decoded. The primitive-only pickle is converted using a restricted unpickler that forbids executable classes.

Inspect `reports/data_audit.json` and `reports/catalog_profile.csv`. The optional `analysis_walkthrough.ipynb` contains five short analysis questions; install `ipykernel` into your virtual environment if VS Code requests a notebook kernel. Source: 8,428 pairs and 1,453 images. After deduplication, contradiction exclusions and grouped partitioning: **4,530 pairs: 3,266 train, 620 validation, 644 test**. Original splits have listing overlap, so this release uses a stricter custom split. One contradictory pair ID and 37 negative pairs inside identity groups are excluded. Cross-partition pairs are excluded too; exclusion categories can overlap.

### 4. Generate text and image embeddings
```powershell
.\.venv\Scripts\python.exe -m catalogguard.cli embed
```
First run downloads pinned pretrained encoders. Generates 384-dimensional text and 512-dimensional image vectors, then caches them. Backbones are frozen; do not claim training CLIP/MiniLM from scratch. If memory is tight, use `embed --batch-size 4` instead. Windows symlink warnings from Hugging Face are usually nonfatal and only affect cache storage efficiency.

### 5. Train and evaluate
```powershell
.\.venv\Scripts\python.exe -m catalogguard.cli train
.\.venv\Scripts\python.exe -m catalogguard.cli evaluate
.\.venv\Scripts\python.exe -m catalogguard.cli retrieval
.\.venv\Scripts\python.exe -m catalogguard.cli latency
.\.venv\Scripts\python.exe -m catalogguard.cli sync
```
The highest validation average precision wins. Classification and suggestion thresholds are selected on validation only. Test scores are not used to choose the model. Do not tune on test results and call a subsequent rerun independent validation.

Outputs:
- `validation_models.csv`: five models and validation metrics.
- `test_metrics.json`: selected model vs lexical baseline, threshold metrics and modality stress tests.
- `test_predictions.csv`: labelled offline error-analysis data, not exposed in the review queue.
- `review_queue.csv`: label-free historical review queue, ordered by distance to classification threshold.
- `retrieval_metrics.json`: known-positive recall@10; not exhaustive catalog recall.
- `latency.json`: warm feature assembly/classifier latency; excludes encoding, network and database.

### 6. Run the larger MongoDB metadata test
```powershell
.\.venv\Scripts\python.exe -m catalogguard.cli abo-scale
```
Downloads the ~87 MB ABO metadata archive and ingests all records into `abo_listings`. Market/domain and item ID jointly identify a record. Indexes and an aggregation summarise product types. `reports/abo_scale.json` records actual rows, unique documents and throughput. No AWS account or ABO image download is required.

**Only claim counts from your successful run.** This tests metadata processing at scale, not matching accuracy on 147,702 products.

### 7. Start the review application
```powershell
.\.venv\Scripts\python.exe -m uvicorn catalogguard.api:app --host 127.0.0.1 --port 8000
```
Leave the terminal running and open **http://127.0.0.1:8000**. Open `.env` in VS Code, copy only the value of `API_KEY`, paste it in the UI and click Connect. The key stays in this browser tab's sessionStorage.

Compare images/titles, inspect similarity signals, save Same product / Different / Unsure, then refresh to confirm persistence. Open Model comparison for real validation metrics. The displayed signals are not SHAP or causal explanations. Review annotations are stored separately; they are not automatically added to training labels.

API docs: http://127.0.0.1:8000/docs. Protected endpoints require `x-api-key`. Images, health and metrics expose only local public-benchmark content and operational data.

### 8. Open MLflow in a second terminal
From the same project root:
```powershell
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```
Open **http://127.0.0.1:5000**, select CatalogGuard, compare runs and inspect model artifacts. MLflow tracks experiments. It does not detect model degradation without fresh evaluation data.

### 9. Containerised serving and Prometheus
Stop local uvicorn using Ctrl+C to free port 8000. Then:
```powershell
docker compose --profile serve up -d --build
docker compose ps
Invoke-RestMethod http://127.0.0.1:8000/health/ready
```
- UI: http://127.0.0.1:8000
- Prometheus target health: http://127.0.0.1:9090/targets
- Prometheus queries/alerts: http://127.0.0.1:9090

Click Inspect signals on some pairs. Query `catalogguard_pairs_scored_total` in Prometheus. Verify the target is UP. Test a saved review after restarting `docker compose restart app`.

Stop with `docker compose --profile serve stop`; data remains in volumes. Do NOT use `down -v` unless you intend to delete database and review data. See `docs/DEPLOYMENT.md` for acceptance criteria and production boundaries.

### 10. Export evidence and publish clean source
```powershell
.\.venv\Scripts\python.exe -m catalogguard.cli export-evidence
git init
git check-ignore -v .venv/ .env data/ artifacts/
git add .
git --no-pager diff --cached --name-only
git commit -m "Build CatalogGuard multimodal matching and review service"
git branch -M main
```
Create an empty GitHub repository, then use YOUR real repository URL:
```powershell
git remote add origin https://github.com/YOUR_USERNAME/CatalogGuard.git
git push -u origin main
```
If origin exists, use `git remote set-url origin ...`. Sign in using Git Credential Manager/browser, not your account password. The ignore rules exclude credentials, environments, data/images, artifacts, runtime reports and MLflow databases. `docs/my_results` intentionally includes only exported aggregate evidence.

### Alternative: run the pipeline script
After installing requirements, you may run the equivalent steps 2–6 with:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_all.ps1
```
This applies only to that PowerShell process and does not change machine policy. The script stops on failures. It does not launch the UI. Choose either individual steps or this script to avoid duplicate training runs.

## Troubleshooting
- Docker unavailable: start Docker Desktop, check Linux containers/WSL2 and `docker compose logs mongo`.
- Authentication failed after changing .env: restore original credentials; the database volume retains existing users.
- Port occupied: stop the other service or change BOTH port mapping and connection settings.
- Interrupted download: rerun; checksum validation prevents silent partial-file use.
- Stale-artifact error: stop app, rerun prepare → embed → train → evaluate → sync.
- Missing/corrupt images: stop and inspect the source; do not substitute unrelated images.
- Keep the supplied pinned environment; SQLAlchemy is pinned for MLflow compatibility.

## Documentation and resume
`docs/INTERVIEW_GUIDE.md` explains methodology and trade-offs. `docs/DATA_SOURCES.md` records attribution and pins. `docs/VERIFICATION.md` records actual verification and remaining local acceptance checks. `docs/reference_results` contains build-verification aggregates; your exports go to `docs/my_results`.

After you reproduce the run and complete Docker acceptance, use:

**CatalogGuard | Multimodal Product Matching & Catalog Quality**
*Python, MongoDB, PyTorch, Sentence Transformers, CLIP, LightGBM, MLflow, FastAPI, Docker, Prometheus*

- Built a multimodal product-matching pipeline combining multilingual text and image embeddings; benchmarked five lexical, semantic, visual and fusion configurations using MLflow.
- Audited 8,428 labelled pairs and 1,453 images; resolved label conflicts and constructed grouped train/validation/test partitions retaining 4,530 pairs.
- Achieved **[actual test AP] average precision** versus **[actual baseline AP]**, with **[actual suggestion precision] precision at [actual coverage] coverage** on 644 held-out pairs.
- Containerised a FastAPI review service with authenticated access, MongoDB persistence and Prometheus metrics; processed **[actual ABO rows] metadata records** in a separate ingestion test.

Use three concise bullets if space is limited. Do not call this company production experience or claim financial impact without measured evidence.

## Results from my Windows run

Compared five model configurations and selected multimodal LightGBM
using validation average precision.

| Metric | Result |
|---|---:|
| Validation average precision | 0.9547 |
| Test average precision | 0.9517 |
| Lexical baseline test average precision | 0.8194 |
| Relative improvement in test AP | 16.1% |
| Test ROC-AUC | 0.9381 |
| Test F1 | 0.8631 |
| Suggested-match precision | 97.65% |
| Suggested-match coverage | 39.60% |
| Held-out test pairs | 644 |

In a separate MongoDB ingestion test, processed 147,702 product
metadata records in 90.5 seconds, excluding download time.

Matching results use the WDC Shoes benchmark with a custom grouped
split. The larger metadata ingestion test uses Amazon Berkeley Objects;
those records were not used to measure matching accuracy.

My aggregate evaluation reports are available in docs/my_results.