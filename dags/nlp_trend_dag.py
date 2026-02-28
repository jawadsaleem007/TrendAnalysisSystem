from __future__ import annotations

import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = os.environ.get("PIPELINE_PYTHON", sys.executable)
DVC_BIN = os.environ.get("PIPELINE_DVC")
COMMAND_TIMEOUT_SECONDS = int(os.environ.get("PIPELINE_CMD_TIMEOUT_SECONDS", "300"))

if not DVC_BIN:
    python_path = Path(PYTHON_BIN)
    candidate = python_path.with_name("dvc.exe")
    DVC_BIN = str(candidate) if candidate.exists() else "dvc"


def run_cmd(command: list[str], timeout_seconds: int | None = None) -> None:
    logging.info("Running command: %s", " ".join(command))
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds or COMMAND_TIMEOUT_SECONDS,
    )
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
    if os.environ.get("SKIP_DVC_PUSH", "0") == "1":
        logging.info("Skipping dvc_push because SKIP_DVC_PUSH=1")
        return
    run_cmd([DVC_BIN, "push"])


default_args = {
    "owner": "trendscope-nlp",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(seconds=10),
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
