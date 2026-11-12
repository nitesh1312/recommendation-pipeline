"""
Model Training & Experiment Tracking Module.
Implements:
1. Baseline Bayesian Popularity Recommender
2. Personalized SVD Matrix Factorization (Latent Factors + Biases)
3. Hybrid Recommendation Engine
Includes automated model serialization and experiment artifact logging.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.config_loader import load_config
from src.utils.tracker import ExperimentTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class BaseRecommender:
    """Abstract interface for recommendation algorithms."""
    def fit(self, train_df: pd.DataFrame, items_df: pd.DataFrame) -> "BaseRecommender":
        raise NotImplementedError

    def predict(self, user_id: str, item_id: str) -> float:
        raise NotImplementedError

    def recommend(self, user_id: str, top_k: int = 10, exclude_known: bool = True) -> List[Tuple[str, float]]:
        raise NotImplementedError


class PopularityRecommender(BaseRecommender):
    """
    Baseline Recommender ranking items by Bayesian Shrinkage Score:
    Score = (v / (v + m)) * R + (m / (v + m)) * C
    """
    def __init__(self, m: float = 10.0):
        self.m = m
        self.global_mean: float = 3.5
        self.item_scores: Dict[str, float] = {}
        self.ranked_items: List[Tuple[str, float]] = []
        self.user_history: Dict[str, set] = {}

    def fit(self, train_df: pd.DataFrame, items_df: pd.DataFrame) -> "PopularityRecommender":
        logger.info(f"Fitting Popularity Recommender with Bayesian shrinkage m={self.m}...")
        self.global_mean = float(train_df["rating"].mean())
        
        # User history
        self.user_history = train_df.groupby("user_id")["item_id"].apply(set).to_dict()

        # Item statistics
        grp = train_df.groupby("item_id")["rating"].agg(["count", "mean"])
        
        all_item_ids = list(items_df["item_id"].unique())
        for item_id in all_item_ids:
            if item_id in grp.index:
                v = grp.loc[item_id, "count"]
                R = grp.loc[item_id, "mean"]
                score = ((v / (v + self.m)) * R) + ((self.m / (v + self.m)) * self.global_mean)
            else:
                score = self.global_mean
            self.item_scores[item_id] = float(score)

        self.ranked_items = sorted(self.item_scores.items(), key=lambda x: x[1], reverse=True)
        return self

    def predict(self, user_id: str, item_id: str) -> float:
        return self.item_scores.get(item_id, self.global_mean)

    def recommend(self, user_id: str, top_k: int = 10, exclude_known: bool = True) -> List[Tuple[str, float]]:
        known = self.user_history.get(user_id, set()) if exclude_known else set()
        recs = []
        for item_id, score in self.ranked_items:
            if item_id not in known:
                recs.append((item_id, score))
            if len(recs) >= top_k:
                break
        return recs


class SVDCollaborativeFiltering(BaseRecommender):
    """
    Personalized Matrix Factorization with Regularized Latent Factors and Biases:
    r_hat(u, i) = mu + b_u + b_i + p_u^T . q_i
    Optimized via Stochastic Gradient Descent (SGD) with L2 regularization.
    """
    def __init__(
        self,
        n_factors: int = 32,
        n_epochs: int = 25,
        lr: float = 0.007,
        reg: float = 0.05,
        random_seed: int = 42
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.random_seed = random_seed

        self.global_mean: float = 0.0
        self.user_to_idx: Dict[str, int] = {}
        self.item_to_idx: Dict[str, int] = {}
        self.idx_to_item: Dict[int, str] = {}

        self.bu: Optional[np.ndarray] = None
        self.bi: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None
        self.Q: Optional[np.ndarray] = None

        self.user_history: Dict[str, set] = {}

    def fit(self, train_df: pd.DataFrame, items_df: pd.DataFrame) -> "SVDCollaborativeFiltering":
        logger.info(f"Fitting SVD Matrix Factorization (k={self.n_factors}, epochs={self.n_epochs}, lr={self.lr}, reg={self.reg})...")
        np.random.seed(self.random_seed)

        self.global_mean = float(train_df["rating"].mean())
        self.user_history = train_df.groupby("user_id")["item_id"].apply(set).to_dict()

        unique_users = train_df["user_id"].unique()
        unique_items = items_df["item_id"].unique()

        self.user_to_idx = {u: i for i, u in enumerate(unique_users)}
        self.item_to_idx = {it: i for i, it in enumerate(unique_items)}
        self.idx_to_item = {i: it for it, i in self.item_to_idx.items()}

        n_u = len(self.user_to_idx)
        n_i = len(self.item_to_idx)

        # Initialize biases and latent factors
        self.bu = np.zeros(n_u, dtype=np.float32)
        self.bi = np.zeros(n_i, dtype=np.float32)
        self.P = np.random.normal(0, 0.08, (n_u, self.n_factors)).astype(np.float32)
        self.Q = np.random.normal(0, 0.08, (n_i, self.n_factors)).astype(np.float32)

        u_indices = train_df["user_id"].map(self.user_to_idx).values
        i_indices = train_df["item_id"].map(self.item_to_idx).values
        ratings = train_df["rating"].values.astype(np.float32)

        # SGD Optimization
        for epoch in range(1, self.n_epochs + 1):
            indices = np.random.permutation(len(ratings))
            epoch_loss = 0.0

            for idx in indices:
                u = u_indices[idx]
                i = i_indices[idx]
                r = ratings[idx]

                # Prediction
                pred = self.global_mean + self.bu[u] + self.bi[i] + float(np.dot(self.P[u], self.Q[i]))
                err = r - pred
                epoch_loss += err ** 2

                # Gradient updates with L2 penalty
                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])
                
                P_u_old = self.P[u].copy()
                self.P[u] += self.lr * (err * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (err * P_u_old - self.reg * self.Q[i])

            if epoch % 5 == 0 or epoch == self.n_epochs:
                rmse_loss = np.sqrt(epoch_loss / len(ratings))
                logger.info(f"Epoch {epoch:02d}/{self.n_epochs} - Train RMSE: {rmse_loss:.4f}")

        return self

    def predict(self, user_id: str, item_id: str) -> float:
        """Predicts explicit rating bounded within [1.0, 5.0]."""
        u_idx = self.user_to_idx.get(user_id)
        i_idx = self.item_to_idx.get(item_id)

        pred = self.global_mean
        if u_idx is not None and i_idx is not None:
            pred += self.bu[u_idx] + self.bi[i_idx] + float(np.dot(self.P[u_idx], self.Q[i_idx]))
        elif u_idx is not None:
            pred += self.bu[u_idx]
        elif i_idx is not None:
            pred += self.bi[i_idx]

        return float(np.clip(pred, 1.0, 5.0))

    def recommend(self, user_id: str, top_k: int = 10, exclude_known: bool = True) -> List[Tuple[str, float]]:
        """Generates Top-K personalized recommendations."""
        u_idx = self.user_to_idx.get(user_id)
        known = self.user_history.get(user_id, set()) if exclude_known else set()

        if u_idx is None:
            # Fallback for completely cold users
            scores = [(item_id, self.global_mean + self.bi[self.item_to_idx[item_id]]) for item_id in self.item_to_idx]
        else:
            # Vectorized dot product scoring across all catalog items
            user_vec = self.P[u_idx]
            base_bias = self.global_mean + self.bu[u_idx]
            scores_array = base_bias + self.bi + np.dot(self.Q, user_vec)
            scores_array = np.clip(scores_array, 1.0, 5.0)
            
            scores = [(self.idx_to_item[i], float(scores_array[i])) for i in range(len(scores_array))]

        filtered = [s for s in scores if s[0] not in known]
        filtered.sort(key=lambda x: x[1], reverse=True)
        return filtered[:top_k]


class HybridRecommender(BaseRecommender):
    """
    Ensemble Recommender combining Personalized SVD predictions
    with Bayesian Popularity priors to balance discovery and accuracy.
    """
    def __init__(self, cf_model: SVDCollaborativeFiltering, pop_model: PopularityRecommender, cf_weight: float = 0.75):
        self.cf_model = cf_model
        self.pop_model = pop_model
        self.cf_weight = cf_weight
        self.pop_weight = 1.0 - cf_weight

    def predict(self, user_id: str, item_id: str) -> float:
        cf_pred = self.cf_model.predict(user_id, item_id)
        pop_pred = self.pop_model.predict(user_id, item_id)
        return float(self.cf_weight * cf_pred + self.pop_weight * pop_pred)

    def recommend(self, user_id: str, top_k: int = 10, exclude_known: bool = True) -> List[Tuple[str, float]]:
        cf_recs = dict(self.cf_model.recommend(user_id, top_k=top_k * 3, exclude_known=exclude_known))
        pop_recs = dict(self.pop_model.recommend(user_id, top_k=top_k * 3, exclude_known=exclude_known))

        candidate_items = set(cf_recs.keys()).union(set(pop_recs.keys()))
        hybrid_scores = []

        for item_id in candidate_items:
            cf_score = cf_recs.get(item_id, self.cf_model.predict(user_id, item_id))
            pop_score = pop_recs.get(item_id, self.pop_model.predict(user_id, item_id))
            combined = float(self.cf_weight * cf_score + self.pop_weight * pop_score)
            hybrid_scores.append((item_id, combined))

        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        return hybrid_scores[:top_k]


class ModelTrainer:
    """Orchestrates model fitting, experiment tracking, and artifact persistence."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        self.processed_dir = Path(self.config["paths"]["processed_data_dir"])
        self.models_dir = Path(self.config["paths"]["models_dir"])
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = ExperimentTracker(self.config["paths"].get("tracking_dir", "reports/mlflow_runs"))

    def train_models(self) -> Dict[str, Any]:
        """Trains baseline, personalized SVD, and hybrid models."""
        df_train = pd.read_parquet(self.processed_dir / "train_interactions.parquet")
        df_items = pd.read_parquet(self.processed_dir / "clean_items.parquet")

        # 1. Train Baseline Popularity
        run_id_pop = self.tracker.start_run("PopularityBaseline_Run", {"model_family": "heuristic_baseline"})
        pop_cfg = self.config["training"]["models"][0]
        pop_model = PopularityRecommender(m=float(pop_cfg.get("bayesian_shrinkage_m", 10.0)))
        pop_model.fit(df_train, df_items)

        pop_model_path = self.models_dir / "popularity_model.pkl"
        with open(pop_model_path, "wb") as f:
            pickle.dump(pop_model, f)

        self.tracker.log_params({"model_type": "PopularityRecommender", "m": pop_model.m})
        self.tracker.log_artifact(str(pop_model_path))
        self.tracker.end_run()

        # 2. Train SVD Matrix Factorization
        run_id_svd = self.tracker.start_run("SVD_CollaborativeFiltering_Run", {"model_family": "matrix_factorization"})
        svd_cfg = self.config["training"]["models"][1]
        svd_model = SVDCollaborativeFiltering(
            n_factors=int(svd_cfg.get("n_factors", 32)),
            n_epochs=int(svd_cfg.get("n_epochs", 25)),
            lr=float(svd_cfg.get("lr", 0.007)),
            reg=float(svd_cfg.get("reg", 0.05)),
            random_seed=int(self.config["pipeline"].get("random_seed", 42))
        )
        svd_model.fit(df_train, df_items)

        svd_model_path = self.models_dir / "svd_cf_model.pkl"
        with open(svd_model_path, "wb") as f:
            pickle.dump(svd_model, f)

        self.tracker.log_params({
            "model_type": "SVDCollaborativeFiltering",
            "n_factors": svd_model.n_factors,
            "n_epochs": svd_model.n_epochs,
            "lr": svd_model.lr,
            "reg": svd_model.reg
        })
        self.tracker.log_artifact(str(svd_model_path))
        self.tracker.end_run()

        # 3. Hybrid Recommender
        hybrid_cfg = self.config["training"]["models"][2] if len(self.config["training"]["models"]) > 2 else {"cf_weight": 0.75}
        hybrid_model = HybridRecommender(
            cf_model=svd_model,
            pop_model=pop_model,
            cf_weight=float(hybrid_cfg.get("cf_weight", 0.75))
        )
        hybrid_model_path = self.models_dir / "hybrid_model.pkl"
        with open(hybrid_model_path, "wb") as f:
            pickle.dump(hybrid_model, f)

        return {
            "status": "SUCCESS",
            "popularity_model": str(pop_model_path),
            "svd_cf_model": str(svd_model_path),
            "hybrid_model": str(hybrid_model_path)
        }


def run_training(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """CLI/Dag entrypoint for model training."""
    cfg = load_config(config_path)
    trainer = ModelTrainer(cfg)
    return trainer.train_models()


if __name__ == "__main__":
    run_training()
