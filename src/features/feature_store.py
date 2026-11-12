"""
Feature Management & Feature Store Module.
Engineers reusable user, item, and contextual interaction features.
Maintains an offline Feature Store supporting point-in-time entity retrieval and catalog documentation.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FeatureStore:
    """Production-ready offline feature store and feature engineering engine."""

    GENRES = [
        "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
        "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
        "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        self.processed_dir = Path(self.config["paths"]["processed_data_dir"])
        self.features_dir = Path(self.config["paths"]["features_data_dir"])
        self.reports_dir = Path(self.config["paths"]["reports_dir"])
        
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.user_features_df: Optional[pd.DataFrame] = None
        self.item_features_df: Optional[pd.DataFrame] = None

    def compute_user_features(
        self,
        df_users: pd.DataFrame,
        df_train_int: pd.DataFrame,
        df_items: pd.DataFrame
    ) -> pd.DataFrame:
        """Computes statistical, categorical, and behavioral features for users."""
        logger.info("Computing reusable user feature tables...")
        
        # Merge train interactions with item genres to discover genre preferences
        int_with_genres = pd.merge(df_train_int, df_items[["item_id", "genres"]], on="item_id", how="left")
        
        # Explode genres to find top preferred genre per user
        int_with_genres["genre_list"] = int_with_genres["genres"].fillna("Other").apply(lambda x: str(x).split("|"))
        exploded = int_with_genres.explode("genre_list")
        user_top_genre = exploded.groupby(["user_id", "genre_list"]).size().reset_index(name="count")
        user_top_genre = user_top_genre.sort_values(["user_id", "count"], ascending=[True, False]).drop_duplicates(subset=["user_id"])
        user_top_genre = user_top_genre.rename(columns={"genre_list": "user_preferred_genre"})[["user_id", "user_preferred_genre"]]

        # Aggregations
        ref_time = df_train_int["timestamp"].max()
        user_stats = df_train_int.groupby("user_id").agg(
            user_avg_rating=("rating", "mean"),
            user_rating_count=("rating", "count"),
            user_rating_std=("rating", lambda x: float(np.std(x)) if len(x) > 1 else 0.0),
            first_ts=("timestamp", "min"),
            last_ts=("timestamp", "max")
        ).reset_index()

        user_stats["user_days_since_first_interaction"] = (ref_time - user_stats["first_ts"]).dt.total_seconds() / 86400.0
        user_stats["user_days_since_last_interaction"] = (ref_time - user_stats["last_ts"]).dt.total_seconds() / 86400.0
        user_stats["user_activity_ratio"] = user_stats["user_rating_count"] / (user_stats["user_days_since_first_interaction"] + 1.0)

        # Merge with demographic metadata
        features = pd.merge(df_users, user_stats, on="user_id", how="left")
        features = pd.merge(features, user_top_genre, on="user_id", how="left")

        # Fill defaults for cold users
        global_avg = df_train_int["rating"].mean()
        features["user_avg_rating"] = features["user_avg_rating"].fillna(global_avg)
        features["user_rating_count"] = features["user_rating_count"].fillna(0).astype(int)
        features["user_rating_std"] = features["user_rating_std"].fillna(0.0)
        features["user_preferred_genre"] = features["user_preferred_genre"].fillna("Drama")
        features["user_days_since_first_interaction"] = features["user_days_since_first_interaction"].fillna(0.0)
        features["user_days_since_last_interaction"] = features["user_days_since_last_interaction"].fillna(999.0)
        features["user_activity_ratio"] = features["user_activity_ratio"].fillna(0.0)

        cols_to_keep = [
            "user_id", "age", "gender", "occupation",
            "user_avg_rating", "user_rating_count", "user_rating_std",
            "user_preferred_genre", "user_activity_ratio",
            "user_days_since_first_interaction", "user_days_since_last_interaction"
        ]
        self.user_features_df = features[cols_to_keep].copy()
        
        # Save to feature store
        self.user_features_df.to_parquet(self.features_dir / "user_features.parquet", index=False)
        self.user_features_df.to_csv(self.features_dir / "user_features.csv", index=False)
        return self.user_features_df

    def compute_item_features(
        self,
        df_items: pd.DataFrame,
        df_train_int: pd.DataFrame
    ) -> pd.DataFrame:
        """Computes statistical, popularity, and multi-hot genre features for items."""
        logger.info("Computing reusable item feature tables...")
        
        current_year = 2024
        item_stats = df_train_int.groupby("item_id").agg(
            item_avg_rating=("rating", "mean"),
            item_popularity_count=("rating", "count"),
            item_rating_std=("rating", lambda x: float(np.std(x)) if len(x) > 1 else 0.0)
        ).reset_index()

        features = pd.merge(df_items, item_stats, on="item_id", how="left")
        global_avg = df_train_int["rating"].mean()

        features["item_avg_rating"] = features["item_avg_rating"].fillna(global_avg)
        features["item_popularity_count"] = features["item_popularity_count"].fillna(0).astype(int)
        features["item_rating_std"] = features["item_rating_std"].fillna(0.0)
        features["item_recency_years"] = current_year - features["release_year"]

        # Multi-hot genre encoding
        for g in self.GENRES:
            features[f"genre_{g.lower().replace('-', '_')}"] = features["genres"].apply(
                lambda x: 1.0 if g in str(x).split("|") else 0.0
            )

        self.item_features_df = features.copy()
        self.item_features_df.to_parquet(self.features_dir / "item_features.parquet", index=False)
        self.item_features_df.to_csv(self.features_dir / "item_features.csv", index=False)
        return self.item_features_df

    def compute_interaction_features(
        self,
        df_interactions: pd.DataFrame,
        user_feat: pd.DataFrame,
        item_feat: pd.DataFrame
    ) -> pd.DataFrame:
        """Extracts interaction-level temporal and user-item category affinity features."""
        logger.info("Computing interaction-level contextual features...")
        df = df_interactions.copy()
        
        # Temporal features
        df["interaction_hour"] = df["timestamp"].dt.hour
        df["interaction_weekday"] = df["timestamp"].dt.weekday
        df["is_weekend"] = df["interaction_weekday"].isin([5, 6]).astype(int)

        # Merge user preference and item genre
        df = pd.merge(df, user_feat[["user_id", "user_preferred_genre"]], on="user_id", how="left")
        df = pd.merge(df, item_feat[["item_id", "genres"]], on="item_id", how="left")

        # Category affinity: 1 if item genres contain user's preferred genre, else 0
        df["user_item_category_affinity"] = df.apply(
            lambda r: 1.0 if pd.notna(r["user_preferred_genre"]) and r["user_preferred_genre"] in str(r["genres"]).split("|") else 0.0,
            axis=1
        )

        cols_to_keep = [
            "interaction_id", "user_id", "item_id", "rating", "timestamp",
            "interaction_hour", "interaction_weekday", "is_weekend",
            "user_item_category_affinity"
        ]
        return df[cols_to_keep].copy()

    def generate_feature_catalog(self) -> pd.DataFrame:
        """Documents metadata, entity types, update frequency, and descriptions for all features."""
        catalog_records = [
            {"feature_name": "user_avg_rating", "entity": "User", "data_type": "Float", "source": "Train Interactions Aggregation", "update_frequency": "Batch / Daily", "description": "Historical average rating given by the user across all rated items."},
            {"feature_name": "user_rating_count", "entity": "User", "data_type": "Integer", "source": "Train Interactions Aggregation", "update_frequency": "Batch / Daily", "description": "Total number of items rated/interacted with by the user."},
            {"feature_name": "user_rating_std", "entity": "User", "data_type": "Float", "source": "Train Interactions Aggregation", "update_frequency": "Batch / Daily", "description": "Standard deviation of user ratings indicating rating variance."},
            {"feature_name": "user_preferred_genre", "entity": "User", "data_type": "Categorical", "source": "Interactions x Item Catalog", "update_frequency": "Batch / Weekly", "description": "Dominant movie genre the user interacts with most frequently."},
            {"feature_name": "user_activity_ratio", "entity": "User", "data_type": "Float", "source": "Interactions Aggregation", "update_frequency": "Batch / Daily", "description": "Interaction count per day of platform tenure."},
            {"feature_name": "user_days_since_first_interaction", "entity": "User", "data_type": "Float", "source": "Interactions Timestamp", "update_frequency": "Batch / Daily", "description": "Days since the user's initial interaction on the platform."},
            {"feature_name": "item_popularity_count", "entity": "Item", "data_type": "Integer", "source": "Train Interactions Aggregation", "update_frequency": "Batch / Hourly", "description": "Total interaction volume received by the item."},
            {"feature_name": "item_avg_rating", "entity": "Item", "data_type": "Float", "source": "Train Interactions Aggregation", "update_frequency": "Batch / Hourly", "description": "Mean rating received by the item across all consumers."},
            {"feature_name": "item_rating_std", "entity": "Item", "data_type": "Float", "source": "Train Interactions Aggregation", "update_frequency": "Batch / Hourly", "description": "Standard deviation of ratings reflecting review consensus."},
            {"feature_name": "item_recency_years", "entity": "Item", "data_type": "Integer", "source": "Item Catalog Metadata", "update_frequency": "Static / Monthly", "description": "Age of the item relative to current year."},
            {"feature_name": "genre_* (18 vectors)", "entity": "Item", "data_type": "Binary Vector", "source": "Item Catalog Genres", "update_frequency": "Static", "description": "Multi-hot binary indicator vectors for 18 distinct genres."},
            {"feature_name": "interaction_hour", "entity": "Interaction", "data_type": "Integer (0-23)", "source": "Interaction Timestamp", "update_frequency": "Real-time / Point-in-time", "description": "Hour of the day when interaction event transpired."},
            {"feature_name": "interaction_weekday", "entity": "Interaction", "data_type": "Integer (0-6)", "source": "Interaction Timestamp", "update_frequency": "Real-time / Point-in-time", "description": "Day of the week of the interaction event."},
            {"feature_name": "user_item_category_affinity", "entity": "Interaction", "data_type": "Binary / Float", "source": "User-Item Cross Feature", "update_frequency": "Point-in-time", "description": "Indicator of whether the target item matches the user's primary preferred genre."}
        ]

        df_catalog = pd.DataFrame(catalog_records)
        df_catalog.to_csv(self.reports_dir / "feature_catalog.csv", index=False)
        
        # Save markdown table
        md_lines = [
            "# Feature Store Catalog & Data Dictionary",
            "",
            "| Feature Name | Entity | Data Type | Source | Update Frequency | Description |",
            "| :--- | :---: | :---: | :--- | :---: | :--- |"
        ]
        for _, r in df_catalog.iterrows():
            md_lines.append(f"| `{r['feature_name']}` | `{r['entity']}` | `{r['data_type']}` | {r['source']} | {r['update_frequency']} | {r['description']} |")

        with open(self.reports_dir / "feature_catalog.md", "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        logger.info(f"Feature catalog generated at {self.reports_dir / 'feature_catalog.md'}")
        return df_catalog

    def build_feature_store(self) -> Dict[str, Any]:
        """Runs end-to-end feature engineering and persists tables and catalog."""
        df_users = pd.read_parquet(self.processed_dir / "clean_users.parquet")
        df_items = pd.read_parquet(self.processed_dir / "clean_items.parquet")
        df_train_int = pd.read_parquet(self.processed_dir / "train_interactions.parquet")
        df_test_int = pd.read_parquet(self.processed_dir / "test_interactions.parquet")

        u_feat = self.compute_user_features(df_users, df_train_int, df_items)
        i_feat = self.compute_item_features(df_items, df_train_int)

        train_int_feat = self.compute_interaction_features(df_train_int, u_feat, i_feat)
        test_int_feat = self.compute_interaction_features(df_test_int, u_feat, i_feat)

        train_int_feat.to_parquet(self.features_dir / "train_interaction_features.parquet", index=False)
        test_int_feat.to_parquet(self.features_dir / "test_interaction_features.parquet", index=False)

        self.generate_feature_catalog()

        return {
            "status": "SUCCESS",
            "user_features_count": len(u_feat),
            "item_features_count": len(i_feat),
            "train_interaction_features_count": len(train_int_feat),
            "test_interaction_features_count": len(test_int_feat)
        }


def run_feature_engineering(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """CLI/Dag entrypoint for feature engineering."""
    cfg = load_config(config_path)
    store = FeatureStore(cfg)
    return store.build_feature_store()


if __name__ == "__main__":
    run_feature_engineering()
