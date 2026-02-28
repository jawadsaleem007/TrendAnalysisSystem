# Trend Intelligence Pipeline

Reproducible NLP data engineering pipeline for TrendScope Analytics.

## Project Structure

```
trend_intelligence_pipeline/
├── dags/
│   └── nlp_trend_dag.py
├── src/
│   ├── scraper.py
│   ├── preprocess.py
│   ├── representation.py
│   └── statistics.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
├── reports/
├── dvc.yaml
├── requirements.txt
└── README.md
```

## Data Source

Public platform used: **Hacker News (Show HN)** via Algolia public API.

Each listing includes:

- product_name
- tagline
- tags/categories
- popularity_signal (points, num_comments)
- product_url
- scrape_timestamp_utc

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Stages Manually

### Stage 1: Data Acquisition

```bash
python src/scraper.py --limit 300
python src/scraper.py --limit 500
```

Output:

- `data/raw/products_raw.json`

### Stage 2: DVC Versioning + DagsHub remote

```bash
dvc init
dvc add data/raw/products_raw.json
git add .
git commit -m "Track raw dataset v1 (300)"

python src/scraper.py --limit 500
dvc add data/raw/products_raw.json
git add .
git commit -m "Track raw dataset v2 (500)"
```

Configure DagsHub remote (S3-compatible):

```bash
dvc remote add -d origin s3://dagshub/<username>/<repo>.dvc
dvc remote modify origin endpointurl https://dagshub.com
dvc remote modify origin access_key_id <DAGSHUB_USER_TOKEN>
dvc remote modify origin secret_access_key <DAGSHUB_TOKEN>
dvc push
```

### Stage 3: Preprocessing

```bash
python src/preprocess.py
```

Output:

- `data/processed/products_clean.csv`

### Stage 4: Representation

```bash
python src/representation.py
```

Outputs:

- `data/features/vocab.json`
- `data/features/bow_matrix.npy`
- `data/features/frequencies.json`

### Stage 5: Linguistic Intelligence

```bash
python src/statistics.py
```

Output:

- `reports/trend_summary.txt`

### Stage 6: Orchestration (Airflow)

Airflow DAG file:

- `dags/nlp_trend_dag.py`

Tasks:

1. `scrape_data`
2. `preprocess_data`
3. `generate_features`
4. `compute_statistics`
5. `dvc_push`

The DAG is manually triggerable (`schedule=None`) and includes retries + logging.

## Reproducible Pipeline with DVC

```bash
dvc repro
```

## Notes

- Scraper includes rate limiting, retry logic, and missing value handling.
- NLP cleaning includes Unicode normalization, lowercasing, HTML/URL removal,
  punctuation removal, tokenization, stopword removal, stemming, and token filters.
- Duplicate detection uses Minimum Edit Distance (Levenshtein).
