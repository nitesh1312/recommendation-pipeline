"""
Data Transformation Module.
Computes domain aggregations and creates derived datasets:
1. User activity summary dataset
2. Item popularity & quality summary dataset
3. User-Item interaction matrix (sparse and tabular)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz

from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DataTransformer:
    """Transforms prepared data into derived analytical structures and interaction matrices."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        self.processed_dir = Path(self.config["paths"]["processed_data_dir"])

    def create_user_activity_summary(
        self,
        df_users: pd.DataFrame,
        df_train_interactions: pd.DataFrame
    ) -> pd.DataFrame:
        """Derives aggregate behavioral statistics for each user."""
        logger.info("Generating user activity summaries...")
        user_grp = df_train_interactions.groupby("user_id")

        agg_df = user_grp.agg(
            total_interactions=("rating", "count"),
            avg_rating=("rating", "mean"),
            std_rating=("rating", lambda x: float(np.std(x)) if len(x) > 1 else 0.0),
            first_interaction_ts=("timestamp", "min"),
            last_interaction_ts=("timestamp", "max")
        ).reset_index()

        # Merge with user base
        user_summary = pd.merge(df_users, agg_df, on="user_id", how="left")
        user_summary["total_interactions"] = user_summary["total_interactions"].fillna(0).astype(int)
        user_summary["avg_rating"] = user_summary["avg_rating"].fillna(df_train_interactions["rating"].mean())
        user_summary["std_rating"] = user_summary["std_rating"].fillna(0.0)

        # Categorize activity tier
        user_summary["activity_tier"] = pd.qcut(
            user_summary["total_interactions"].rank(method="first"),
            q=4,
            labels=["Low", "Medium-Low", "Medium-High", "High"]
        )

        output_path = self.processed_dir / "user_activity_summary.parquet"
        user_summary.to_parquet(output_path, index=False)
        user_summary.to_csv(self.processed_dir / "user_activity_summary.csv", index=False)
        logger.info(f"User activity summary saved to {output_path}")
        return user_summary

    def create_item_popularity_summary(
        self,
        df_items: pd.DataFrame,
        df_train_interactions: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Derives item popularity, Bayesian smoothed rating, and demand metrics.
        Bayesian formula: (v / (v + m)) * R + (m / (v + m)) * C
        where:
          v = number of ratings for item
          m = minimum rating threshold (e.g. 10)
          R = average rating of item
          C = global mean rating across all items
        """
        logger.info("Generating item popularity and quality metrics...")
        item_grp = df_train_interactions.groupby("item_id")

        global_mean_rating = float(df_train_interactions["rating"].mean())
        m = float(self.config["training"]["models"][0].get("bayesian_shrinkage_m", 10.0))

        agg_df = item_grp.agg(
            rating_count=("rating", "count"),
            avg_rating=("rating", "mean"),
            std_rating=("rating", lambda x: float(np.std(x)) if len(x) > 1 else 0.0),
            positive_ratings_count=("rating", lambda x: int(sum(x >= 4.0)))
        ).reset_index()

        # Compute Bayesian Weighted Rating
        v = agg_df["rating_count"]
        R = agg_df["avg_rating"]
        agg_df["bayesian_weighted_score"] = ((v / (v + m)) * R) + ((m / (v + m)) * global_mean_rating)

        item_summary = pd.merge(df_items, agg_df, on="item_id", how="left")
        item_summary["rating_count"] = item_summary["rating_count"].fillna(0).astype(int)
        item_summary["avg_rating"] = item_summary["avg_rating"].fillna(global_mean_rating)
        item_summary["std_rating"] = item_summary["std_rating"].fillna(0.0)
        item_summary["positive_ratings_count"] = item_summary["positive_ratings_count"].fillna(0).astype(int)
        item_summary["bayesian_weighted_score"] = item_summary["bayesian_weighted_score"].fillna(global_mean_rating)

        # Categorize popularity tier
        item_summary["popularity_tier"] = pd.qcut(
            item_summary["rating_count"].rank(method="first"),
            q=4,
            labels=["Niche", "Moderate", "Popular", "Blockbuster"]
        )

        output_path = self.processed_dir / "item_popularity.parquet"
        item_summary.to_parquet(output_path, index=False)
        item_summary.to_csv(self.processed_dir / "item_popularity.csv", index=False)
        logger.info(f"Item popularity summary saved to {output_path}")
        return item_summary

    def build_user_item_matrix(
        self,
        df_train_interactions: pd.DataFrame,
        df_users: pd.DataFrame,
        df_items: pd.DataFrame
    ) -> Tuple[csr_matrix, Dict[str, int], Dict[str, int]]:
        """Constructs a sparse CSR matrix of user-item ratings."""
        logger.info("Building sparse user-item interaction matrix...")
        
        # Consistent mapping
        user_to_idx = {uid: idx for idx, uid in enumerate(df_users["user_id"].unique())}
        item_to_idx = {iid: idx for idx, iid in enumerate(df_items["item_id"].unique())}

        # Filter interactions to indexed entities
        valid_int = df_train_interactions[
            df_train_interactions["user_id"].isin(user_to_idx) &
            df_train_interactions["item_id"].isin(item_to_idx)
        ].copy()

        rows = valid_int["user_id"].map(user_to_idx).values
        cols = valid_int["item_id"].map(item_to_idx).values
        data = valid_int["rating"].values

        n_users = len(user_to_idx)
        n_items = len(item_to_idx)

        sparse_matrix = csr_matrix((data, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)

        # Save matrix and mappings
        save_npz(self.processed_dir / "user_item_matrix.npz", sparse_matrix)
        
        mappings = {
            "user_to_idx": user_to_idx,
            "item_to_idx": item_to_idx,
            "matrix_shape": [n_users, n_items],
            "non_zero_entries": int(sparse_matrix.nnz),
            "sparsity_pct": round((1.0 - (sparse_matrix.nnz / (n_users * n_items))) * 100, 3)
        }

        with open(self.processed_dir / "matrix_mappings.json", "w", encoding="utf-8") as f:
            json.dump(mappings, f, indent=2)

        logger.info(f"Interaction matrix built with shape {n_users}x{n_items}, sparsity: {mappings['sparsity_pct']}%")
        return sparse_matrix, user_to_idx, item_to_idx

    def transform_all(self) -> Dict[str, Any]:
        """Runs all transformation stages."""
        df_users = pd.read_parquet(self.processed_dir / "clean_users.parquet")
        df_items = pd.read_parquet(self.processed_dir / "clean_items.parquet")
        df_train_int = pd.read_parquet(self.processed_dir / "train_interactions.parquet")

        user_summary = self.create_user_activity_summary(df_users, df_train_int)
        item_summary = self.create_item_popularity_summary(df_items, df_train_int)
        _, u_map, i_map = self.build_user_item_matrix(df_train_int, df_users, df_items)

        return {
            "status": "SUCCESS",
            "users_transformed": len(user_summary),
            "items_transformed": len(item_summary),
            "users_indexed": len(u_map),
            "items_indexed": len(i_map)
        }


def run_transformation(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """CLI/Dag entrypoint for transformation."""
    cfg = load_config(config_path)
    transformer = DataTransformer(cfg)
    return transformer.transform_all()


if __name__ == "__main__":
    run_transformation()
