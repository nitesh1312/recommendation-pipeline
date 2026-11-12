"""
Experiment Tracker Module.
Provides lightweight, production-grade experiment tracking for model parameters,
metrics, and artifact provenance without heavy server dependencies.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class ExperimentTracker:
    """Tracks model training runs, parameters, metrics, and artifact references."""
    
    def __init__(self, tracking_dir: str = "reports/mlflow_runs"):
        self.tracking_dir = Path(tracking_dir)
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        self.active_run: Optional[Dict[str, Any]] = None

    def start_run(self, run_name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """Starts a new experiment run session."""
        run_id = f"{run_name}_{int(time.time())}"
        self.active_run = {
            "run_id": run_id,
            "run_name": run_name,
            "start_time": datetime.utcnow().isoformat(),
            "status": "RUNNING",
            "tags": tags or {},
            "parameters": {},
            "metrics": {},
            "artifacts": []
        }
        return run_id

    def log_param(self, key: str, value: Any) -> None:
        """Logs a single parameter to the active run."""
        if self.active_run:
            self.active_run["parameters"][key] = value

    def log_params(self, params: Dict[str, Any]) -> None:
        """Logs a dictionary of parameters to the active run."""
        if self.active_run:
            self.active_run["parameters"].update(params)

    def log_metric(self, key: str, value: float) -> None:
        """Logs a single numerical metric to the active run."""
        if self.active_run:
            self.active_run["metrics"][key] = float(value)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Logs a dictionary of metrics to the active run."""
        if self.active_run:
            for k, v in metrics.items():
                self.active_run["metrics"][k] = float(v)

    def log_artifact(self, artifact_path: str) -> None:
        """Logs an artifact filepath associated with the run."""
        if self.active_run:
            self.active_run["artifacts"].append(str(artifact_path))

    def end_run(self, status: str = "COMPLETED") -> Dict[str, Any]:
        """Finalizes the active experiment run and persists metadata."""
        if not self.active_run:
            raise RuntimeError("No active experiment run to end.")
            
        self.active_run["status"] = status
        self.active_run["end_time"] = datetime.utcnow().isoformat()
        
        run_file = self.tracking_dir / f"{self.active_run['run_id']}.json"
        with open(run_file, "w", encoding="utf-8") as f:
            json.dump(self.active_run, f, indent=2)
            
        completed_run = self.active_run
        self.active_run = None
        return completed_run

    def get_all_runs(self) -> Dict[str, Dict[str, Any]]:
        """Retrieves summary of all recorded runs."""
        runs = {}
        for f in self.tracking_dir.glob("*.json"):
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                runs[data["run_id"]] = data
        return runs
