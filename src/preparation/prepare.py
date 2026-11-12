"""
Data Preparation Module.
Cleans validated datasets, standardizes datetime and types, and performs
chronological time-based train/test splitting to prevent future data leakage.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.config_loader import load_config
from src.validation.validate import DataValidator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DataPreparator:
    """Prepares validated data for downstream feature engineering and model training."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        self.raw_dir = Path(self.config["paths"]["raw_data_dir"])
        self.processed_dir = Path(self.config["paths"]["processed_data_dir"])
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def prepare_and_split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Runs validation extraction, standardizes schemas, and splits interactions
        chronologically by timestamp.
        """
        logger.info("Executing data preparation pipeline...")
        
        # Load validated clean datasets directly from validator
        validator = DataValidator(self.config)
        df_users_raw = pd.read_csv(self.raw_dir / "users.csv")
        df_items_raw = pd.read_csv(self.raw_dir / "items.csv")
        df_interactions_raw = pd.read_csv(self.raw_dir / "interactions.csv")

        clean_users, _ = validator.validate_users(df_users_raw)
        clean_items, _ = validator.validate_items(df_items_raw)
        clean_interactions, _ = validator.validate_interactions(df_interactions_raw, clean_users, clean_items)

        # Standardize datetimes and numeric fields
        clean_users["signup_date"] = pd.to_datetime(clean_users["signup_date"])
        clean_interactions["timestamp"] = pd.to_datetime(clean_interactions["timestamp"])
        clean_interactions["rating"] = clean_interactions["rating"].astype(np.float32)

        # Chronological Time-based Train/Test Split
        test_size = float(self.config["preparation"].get("test_size", 0.20))
        split_strategy = self.config["preparation"].get("split_strategy", "time_based")

        logger.info(f"Applying '{split_strategy}' split with test_size={test_size}...")
        
        # Sort chronologically
        clean_interactions = clean_interactions.sort_values(by="timestamp").reset_index(drop=True)
        split_idx = int(len(clean_interactions) * (1.0 - test_size))
        
        train_interactions = clean_interactions.iloc[:split_idx].copy().reset_index(drop=True)
        test_interactions = clean_interactions.iloc[split_idx:].copy().reset_index(drop=True)

        split_cutoff_time = train_interactions["timestamp"].max()
        logger.info(f"Train/Test split completed. Split timestamp cutoff: {split_cutoff_time}")
        logger.info(f"Train size: {len(train_interactions)} interactions, Test size: {len(test_interactions)} interactions")

        # Persist prepared datasets
        clean_users.to_parquet(self.processed_dir / "clean_users.parquet", index=False)
        clean_items.to_parquet(self.processed_dir / "clean_items.parquet", index=False)
        train_interactions.to_parquet(self.processed_dir / "train_interactions.parquet", index=False)
        test_interactions.to_parquet(self.processed_dir / "test_interactions.parquet", index=False)

        # Also write CSV copies for quick manual inspection
        clean_users.to_csv(self.processed_dir / "clean_users.csv", index=False)
        clean_items.to_csv(self.processed_dir / "clean_items.csv", index=False)
        train_interactions.to_csv(self.processed_dir / "train_interactions.csv", index=False)
        test_interactions.to_csv(self.processed_dir / "test_interactions.csv", index=False)

        prep_summary = {
            "total_clean_users": len(clean_users),
            "total_clean_items": len(clean_items),
            "total_clean_interactions": len(clean_interactions),
            "train_interactions_count": len(train_interactions),
            "test_interactions_count": len(test_interactions),
            "split_strategy": split_strategy,
            "test_ratio": test_size,
            "train_time_min": str(train_interactions["timestamp"].min()),
            "train_time_max": str(train_interactions["timestamp"].max()),
            "test_time_min": str(test_interactions["timestamp"].min()),
            "test_time_max": str(test_interactions["timestamp"].max())
        }

        with open(self.processed_dir / "preparation_summary.json", "w", encoding="utf-8") as f:
            json.dump(prep_summary, f, indent=2)

        return clean_users, clean_items, train_interactions, test_interactions


def run_preparation(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """CLI/Dag entrypoint for data preparation."""
    cfg = load_config(config_path)
    prep = DataPreparator(cfg)
    clean_u, clean_i, train_int, test_int = prep.prepare_and_split_data()
    return {
        "status": "SUCCESS",
        "num_users": len(clean_u),
        "num_items": len(clean_i),
        "train_interactions": len(train_int),
        "test_interactions": len(test_int)
    }


if __name__ == "__main__":
    run_preparation()
