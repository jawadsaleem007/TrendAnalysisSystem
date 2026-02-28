from __future__ import annotations

import logging
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = os.environ.get("PIPELINE_PYTHON", "python")


def run_cmd(command: list[str]) -> None:
    logging.info("Running command: %s", " ".join(command))
    result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    logging.info("stdout:\n%s", result.stdout)
    if result.stderr:
        logging.warning("stderr:\n%s", result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}")


def task_scrape_data() -> None:
    run_cmd([PYTHON_BIN, "src/scraper.py", "--limit", "500"])


def task_preprocess_data() -> None:
    run_cmd([PYTHON_BIN, "src/preprocess.py"])


def task_generate_features() -> None:
    run_cmd([PYTHON_BIN, "src/representation.py"])


def task_compute_statistics() -> None:
    run_cmd([PYTHON_BIN, "src/statistics.py"])


def task_dvc_push() -> None:
    run_cmd(["dvc", "add", "data/raw/products_raw.json"])
    run_cmd(["dvc", "add", "data/processed/products_clean.csv"])
    run_cmd(["dvc", "add", "data/features/vocab.json"])
    run_cmd(["dvc", "add", "data/features/bow_matrix.npy"])
    run_cmd(["dvc", "push"])


default_args = {
    "owner": "trendscope-nlp",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=3),
}


with DAG(
    dag_id="nlp_trend_dag",
    default_args=default_args,
    description="TrendScope NLP trend intelligence pipeline",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["nlp", "dvc", "trendscope"],
) as dag:
    scrape_data = PythonOperator(task_id="scrape_data", python_callable=task_scrape_data)
    preprocess_data = PythonOperator(task_id="preprocess_data", python_callable=task_preprocess_data)
    generate_features = PythonOperator(task_id="generate_features", python_callable=task_generate_features)
    compute_statistics = PythonOperator(task_id="compute_statistics", python_callable=task_compute_statistics)
    dvc_push = PythonOperator(task_id="dvc_push", python_callable=task_dvc_push)

    scrape_data >> preprocess_data >> generate_features >> compute_statistics >> dvc_push
