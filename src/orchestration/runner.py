"""
Pipeline Orchestration Runner.
Executes the full end-to-end recommendation workflow with strict dependency handling,
execution timing, structured logging, and health checking.
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.ingestion.ingest import run_ingestion
from src.validation.validate import run_validation
from src.preparation.prepare import run_preparation
from src.transformation.transform import run_transformation
from src.features.feature_store import run_feature_engineering
from src.training.train import run_training
from src.evaluation.evaluate import run_evaluation
from src.utils.config_loader import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline_execution.log", mode="a", encoding="utf-8")
    ]
)
logger = logging.getLogger("PipelineOrchestrator")


class PipelineOrchestrator:
    """Manages sequential execution of end-to-end recommendation pipeline stages."""

    STAGES = [
        ("data_ingestion", run_ingestion),
        ("data_validation", run_validation),
        ("data_preparation", run_preparation),
        ("data_transformation", run_transformation),
        ("feature_engineering", run_feature_engineering),
        ("model_training", run_training),
        ("model_evaluation", run_evaluation)
    ]

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = config_path
        self.config = load_config(config_path)
        self.execution_summary: Dict[str, Any] = {
            "pipeline_name": self.config["pipeline"]["name"],
            "version": self.config["pipeline"]["version"],
            "execution_start": datetime.utcnow().isoformat(),
            "status": "RUNNING",
            "stages": {}
        }

    def run_stage(self, stage_name: str) -> Dict[str, Any]:
        """Runs an individual stage by name."""
        stage_map = dict(self.STAGES)
        if stage_name not in stage_map:
            raise ValueError(f"Unknown stage: {stage_name}. Valid stages: {list(stage_map.keys())}")

        func = stage_map[stage_name]
        logger.info(f"========== STARTING STAGE: {stage_name.upper()} ==========")
        start_t = time.time()
        
        try:
            result = func(self.config_path)
            duration = round(time.time() - start_t, 3)
            logger.info(f"========== COMPLETED STAGE: {stage_name.upper()} in {duration}s ==========")
            return {"status": "SUCCESS", "duration_seconds": duration, "output": result}
        except Exception as e:
            duration = round(time.time() - start_t, 3)
            logger.error(f"========== FAILED STAGE: {stage_name.upper()} after {duration}s: {str(e)} ==========", exc_info=True)
            return {"status": "FAILED", "duration_seconds": duration, "error": str(e)}

    def run_all(self) -> Dict[str, Any]:
        """Executes the entire end-to-end DAG workflow."""
        logger.info("================================================================================")
        logger.info("🚀 INITIATING END-TO-END RECOMMENDATION DATA MANAGEMENT PIPELINE")
        logger.info("================================================================================")
        overall_start = time.time()

        for stage_name, _ in self.STAGES:
            stage_result = self.run_stage(stage_name)
            self.execution_summary["stages"][stage_name] = stage_result
            
            if stage_result["status"] == "FAILED":
                self.execution_summary["status"] = "FAILED"
                self.execution_summary["failed_stage"] = stage_name
                logger.error(f"Pipeline execution halted due to failure in stage '{stage_name}'.")
                break

        if self.execution_summary["status"] != "FAILED":
            self.execution_summary["status"] = "SUCCESS"

        total_duration = round(time.time() - overall_start, 3)
        self.execution_summary["execution_end"] = datetime.utcnow().isoformat()
        self.execution_summary["total_duration_seconds"] = total_duration

        # Save run summary report
        reports_dir = Path(self.config["paths"]["reports_dir"])
        reports_dir.mkdir(parents=True, exist_ok=True)
        summary_path = reports_dir / "pipeline_run_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(self.execution_summary, f, indent=2)

        logger.info("================================================================================")
        logger.info(f"🏁 PIPELINE RUN FINISHED: Status={self.execution_summary['status']}, Total Time={total_duration}s")
        logger.info(f"Summary persisted to {summary_path}")
        logger.info("================================================================================")

        return self.execution_summary


def run_pipeline(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Helper entry point."""
    orchestrator = PipelineOrchestrator(config_path)
    return orchestrator.run_all()


if __name__ == "__main__":
    run_pipeline()
