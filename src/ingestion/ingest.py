"""
Data Ingestion Module.
Generates and ingests rich user profiles, item catalogs, and interaction logs
following MovieLens/streaming platform schemas with support for synthetic anomaly injection.
"""

import json
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DataIngestor:
    """Handles raw data generation, ingestion, and initial staging."""

    GENRES = [
        "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
        "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
        "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]

    OCCUPATIONS = [
        "engineer", "educator", "student", "artist", "executive", "doctor",
        "scientist", "writer", "technician", "marketer", "lawyer", "other"
    ]

    INTERACTION_TYPES = ["rating", "view", "purchase", "bookmark"]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        self.raw_dir = Path(self.config["paths"]["raw_data_dir"])
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        self.random_seed = self.config["pipeline"].get("random_seed", 42)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

    def generate_users(self, num_users: int) -> pd.DataFrame:
        """Generates realistic user demographic data."""
        logger.info(f"Generating {num_users} raw user records...")
        user_ids = [f"U{i:05d}" for i in range(1, num_users + 1)]
        ages = np.random.choice(
            [18, 22, 25, 30, 35, 42, 50, 60],
            size=num_users,
            p=[0.10, 0.20, 0.25, 0.20, 0.12, 0.08, 0.03, 0.02]
        )
        genders = np.random.choice(["M", "F", "Other"], size=num_users, p=[0.52, 0.45, 0.03])
        occupations = np.random.choice(self.OCCUPATIONS, size=num_users)
        zip_codes = [f"{np.random.randint(10000, 99999)}" for _ in range(num_users)]
        
        start_date = datetime.strptime(self.config["ingestion"]["start_date"], "%Y-%m-%d")
        signup_dates = [
            (start_date + timedelta(days=int(np.random.randint(0, 180)))).strftime("%Y-%m-%d %H:%M:%S")
            for _ in range(num_users)
        ]

        df_users = pd.DataFrame({
            "user_id": user_ids,
            "age": ages,
            "gender": genders,
            "occupation": occupations,
            "zip_code": zip_codes,
            "signup_date": signup_dates
        })
        return df_users

    def generate_items(self, num_items: int) -> pd.DataFrame:
        """Generates rich item catalog data with multi-genre labels."""
        logger.info(f"Generating {num_items} raw catalog item records...")
        item_ids = [f"I{i:05d}" for i in range(1, num_items + 1)]
        
        adjectives = ["The Dark", "Interstellar", "Cosmic", "Silent", "Midnight", "Quantum", "Lost", "Golden", "Eternal", "Neon"]
        nouns = ["Knight", "Journey", "Horizon", "Odyssey", "Dreamer", "Protocol", "Chronicles", "Echoes", "Matrix", "Rebellion"]
        titles = [f"{random.choice(adjectives)} {random.choice(nouns)} Part {i%10 + 1}" for i in range(num_items)]

        genres_list = []
        for _ in range(num_items):
            k = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
            sampled_genres = np.random.choice(self.GENRES, size=k, replace=False)
            genres_list.append("|".join(sampled_genres))

        release_years = np.random.choice(range(1990, 2025), size=num_items)
        base_prices = np.round(np.random.uniform(4.99, 29.99, size=num_items), 2)

        df_items = pd.DataFrame({
            "item_id": item_ids,
            "title": titles,
            "genres": genres_list,
            "release_year": release_years,
            "base_price": base_prices
        })
        return df_items

    def generate_interactions(
        self,
        df_users: pd.DataFrame,
        df_items: pd.DataFrame,
        num_interactions: int,
        inject_anomalies: bool = True
    ) -> pd.DataFrame:
        """Generates realistic user-item interaction streams with power-law distributions."""
        logger.info(f"Generating {num_interactions} raw interaction records...")
        user_list = df_users["user_id"].values
        item_list = df_items["item_id"].values

        # Power law distribution for items (popular items get more interactions)
        item_weights = 1.0 / np.power(np.arange(1, len(item_list) + 1), 0.7)
        item_weights /= item_weights.sum()

        sampled_users = np.random.choice(user_list, size=num_interactions)
        sampled_items = np.random.choice(item_list, size=num_interactions, p=item_weights)

        # Ratings skewed towards 3.5 - 4.5
        ratings = np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0], size=num_interactions, p=[0.05, 0.10, 0.25, 0.35, 0.25])
        interaction_types = np.random.choice(self.INTERACTION_TYPES, size=num_interactions, p=[0.60, 0.25, 0.10, 0.05])

        start_ts = datetime.strptime(self.config["ingestion"]["start_date"], "%Y-%m-%d")
        end_ts = datetime.strptime(self.config["ingestion"]["end_date"], "%Y-%m-%d")
        total_seconds = int((end_ts - start_ts).total_seconds())

        timestamps = [
            (start_ts + timedelta(seconds=int(np.random.randint(0, total_seconds)))).strftime("%Y-%m-%d %H:%M:%S")
            for _ in range(num_interactions)
        ]

        df_interactions = pd.DataFrame({
            "interaction_id": [f"INT_{i:07d}" for i in range(1, num_interactions + 1)],
            "user_id": sampled_users,
            "item_id": sampled_items,
            "rating": ratings,
            "interaction_type": interaction_types,
            "timestamp": timestamps
        })

        if inject_anomalies:
            df_interactions = self._inject_anomalies(df_interactions, df_users, df_items)

        return df_interactions

    def _inject_anomalies(
        self,
        df_interactions: pd.DataFrame,
        df_users: pd.DataFrame,
        df_items: pd.DataFrame
    ) -> pd.DataFrame:
        """Injects controlled data quality violations to test validation layer."""
        rate = self.config["ingestion"].get("anomaly_rate", 0.03)
        n_anomalies = max(10, int(len(df_interactions) * rate))
        logger.info(f"Injecting {n_anomalies} synthetic anomalies for validation engine...")

        df_mod = df_interactions.copy()
        
        # 1. Invalid Ratings (e.g. 99.0 or -1.0)
        idx_invalid_ratings = np.random.choice(df_mod.index, size=n_anomalies // 4, replace=False)
        df_mod.loc[idx_invalid_ratings, "rating"] = np.random.choice([-1.0, 0.0, 99.0, 10.5], size=len(idx_invalid_ratings))

        # 2. Dangling Foreign Keys (User not in users or Item not in items)
        idx_dangling_user = np.random.choice(df_mod.index, size=n_anomalies // 6, replace=False)
        df_mod.loc[idx_dangling_user, "user_id"] = "U_NON_EXISTENT_9999"

        idx_dangling_item = np.random.choice(df_mod.index, size=n_anomalies // 6, replace=False)
        df_mod.loc[idx_dangling_item, "item_id"] = "I_NON_EXISTENT_9999"

        # 3. Missing values in mandatory fields
        idx_missing_ts = np.random.choice(df_mod.index, size=n_anomalies // 8, replace=False)
        df_mod.loc[idx_missing_ts, "timestamp"] = np.nan

        # 4. Duplicate rows
        idx_dups = np.random.choice(df_mod.index, size=n_anomalies // 6, replace=False)
        dup_rows = df_mod.loc[idx_dups].copy()
        df_mod = pd.concat([df_mod, dup_rows], ignore_index=True)

        return df_mod

    def ingest_data(self) -> Tuple[Path, Path, Path]:
        """Runs end-to-end data generation/ingestion and writes raw CSVs."""
        n_users = self.config["ingestion"]["num_users"]
        n_items = self.config["ingestion"]["num_items"]
        n_interactions = self.config["ingestion"]["num_interactions"]
        inject_anomalies = self.config["ingestion"].get("include_anomalies", True)

        df_users = self.generate_users(n_users)
        df_items = self.generate_items(n_items)
        df_interactions = self.generate_interactions(df_users, df_items, n_interactions, inject_anomalies)

        users_path = self.raw_dir / "users.csv"
        items_path = self.raw_dir / "items.csv"
        interactions_path = self.raw_dir / "interactions.csv"

        df_users.to_csv(users_path, index=False)
        df_items.to_csv(items_path, index=False)
        df_interactions.to_csv(interactions_path, index=False)

        metadata = {
            "ingestion_timestamp": datetime.utcnow().isoformat(),
            "num_users_raw": len(df_users),
            "num_items_raw": len(df_items),
            "num_interactions_raw": len(df_interactions),
            "anomalies_injected": inject_anomalies,
            "files": {
                "users": str(users_path),
                "items": str(items_path),
                "interactions": str(interactions_path)
            }
        }

        with open(self.raw_dir / "ingestion_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Ingestion completed successfully. Output files saved in {self.raw_dir}")
        return users_path, items_path, interactions_path


def run_ingestion(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """CLI/Dag entrypoint for ingestion."""
    cfg = load_config(config_path)
    ingestor = DataIngestor(cfg)
    u_path, i_path, int_path = ingestor.ingest_data()
    return {
        "status": "SUCCESS",
        "users_path": str(u_path),
        "items_path": str(i_path),
        "interactions_path": str(int_path)
    }


if __name__ == "__main__":
    run_ingestion()
