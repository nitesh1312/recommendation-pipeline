"""
Apache Airflow DAG Definition for Recommendation System Pipeline.
Orchestrates task execution order, retries, alerts, and operational monitoring.
"""

from datetime import datetime, timedelta
import os
import sys

# Add project root to python path for DAG execution
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False


# Task callable wrappers
def task_ingestion(**kwargs):
    from src.ingestion.ingest import run_ingestion
    return run_ingestion("configs/config.yaml")

def task_validation(**kwargs):
    from src.validation.validate import run_validation
    return run_validation("configs/config.yaml")

def task_preparation(**kwargs):
    from src.preparation.prepare import run_preparation
    return run_preparation("configs/config.yaml")

def task_transformation(**kwargs):
    from src.transformation.transform import run_transformation
    return run_transformation("configs/config.yaml")

def task_feature_engineering(**kwargs):
    from src.features.feature_store import run_feature_engineering
    return run_feature_engineering("configs/config.yaml")

def task_model_training(**kwargs):
    from src.training.train import run_training
    return run_training("configs/config.yaml")

def task_model_evaluation(**kwargs):
    from src.evaluation.evaluate import run_evaluation
    return run_evaluation("configs/config.yaml")


default_args = {
    "owner": "mlops-platform-team",
    "depends_on_past": False,
    "email": ["mlops-alerts@dummy-domain.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    "start_date": datetime(2024, 1, 1),
}

if AIRFLOW_AVAILABLE:
    dag = DAG(
        dag_id="recommendation_system_pipeline_dag",
        default_args=default_args,
        description="End-to-End Recommendation Data Management and Model Training Workflow",
        schedule_interval="0 2 * * *",  # Daily at 02:00 AM UTC
        catchup=False,
        tags=["recommendations", "dmml", "machine-learning", "mlops"],
    )

    with dag:
        ingest_task = PythonOperator(
            task_id="data_ingestion",
            python_callable=task_ingestion,
            provide_context=True,
        )

        validate_task = PythonOperator(
            task_id="data_validation",
            python_callable=task_validation,
            provide_context=True,
        )

        prepare_task = PythonOperator(
            task_id="data_preparation",
            python_callable=task_preparation,
            provide_context=True,
        )

        transform_task = PythonOperator(
            task_id="data_transformation",
            python_callable=task_transformation,
            provide_context=True,
        )

        feature_task = PythonOperator(
            task_id="feature_engineering",
            python_callable=task_feature_engineering,
            provide_context=True,
        )

        train_task = PythonOperator(
            task_id="model_training",
            python_callable=task_model_training,
            provide_context=True,
        )

        eval_task = PythonOperator(
            task_id="model_evaluation",
            python_callable=task_model_evaluation,
            provide_context=True,
        )

        # Pipeline Task Dependency Flow
        ingest_task >> validate_task >> prepare_task >> transform_task >> feature_task >> train_task >> eval_task
