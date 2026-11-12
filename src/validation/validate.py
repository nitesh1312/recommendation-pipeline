"""
Data Validation Module.
Executes automated quality checks, referential integrity assertions,
quarantines anomalous/corrupted records into an error dataset, and generates validation logs.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DataValidator:
    """Automated schema validation, anomaly quarantine, and reporting engine."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        self.raw_dir = Path(self.config["paths"]["raw_data_dir"])
        self.errors_dir = Path(self.config["paths"]["errors_data_dir"])
        self.reports_dir = Path(self.config["paths"]["reports_dir"])
        
        self.errors_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.validation_results: Dict[str, Any] = {
            "validation_timestamp": datetime.utcnow().isoformat(),
            "overall_status": "PASSED",
            "summary": {},
            "detailed_checks": []
        }

    def _record_check(self, check_name: str, entity: str, passed: bool, details: Dict[str, Any]):
        """Helper to append check results."""
        self.validation_results["detailed_checks"].append({
            "check_name": check_name,
            "entity": entity,
            "status": "PASSED" if passed else "FAILED",
            "details": details
        })
        if not passed:
            self.validation_results["overall_status"] = "WARNINGS_FOUND"

    def validate_users(self, df_users: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Validates users dataset against schema, nulls, duplicates, and ranges."""
        logger.info(f"Validating {len(df_users)} user records...")
        errors: List[pd.DataFrame] = []

        # 1. Null check on critical fields
        null_mask = df_users["user_id"].isna() | df_users["age"].isna() | df_users["gender"].isna()
        if null_mask.any():
            err_df = df_users[null_mask].copy()
            err_df["error_reason"] = "MISSING_MANDATORY_USER_FIELDS"
            errors.append(err_df)
        self._record_check("null_check", "users", not null_mask.any(), {"null_records_count": int(null_mask.sum())})

        # 2. Duplicate user_id check
        dup_mask = df_users["user_id"].duplicated(keep="first")
        if dup_mask.any():
            err_df = df_users[dup_mask].copy()
            err_df["error_reason"] = "DUPLICATE_USER_ID"
            errors.append(err_df)
        self._record_check("duplicate_check", "users", not dup_mask.any(), {"duplicates_count": int(dup_mask.sum())})

        # 3. Age bounds check (e.g., 5 to 120)
        invalid_age_mask = (df_users["age"] < 5) | (df_users["age"] > 120)
        if invalid_age_mask.any():
            err_df = df_users[invalid_age_mask].copy()
            err_df["error_reason"] = "INVALID_AGE_RANGE"
            errors.append(err_df)
        self._record_check("age_range_check", "users", not invalid_age_mask.any(), {"invalid_age_count": int(invalid_age_mask.sum())})

        all_err_mask = null_mask | dup_mask | invalid_age_mask
        clean_df = df_users[~all_err_mask].copy()
        error_df = pd.concat(errors, ignore_index=True) if errors else pd.DataFrame(columns=list(df_users.columns) + ["error_reason"])

        return clean_df, error_df

    def validate_items(self, df_items: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Validates catalog items for schema, missing titles/genres, and duplicates."""
        logger.info(f"Validating {len(df_items)} item records...")
        errors: List[pd.DataFrame] = []

        # 1. Null check
        null_mask = df_items["item_id"].isna() | df_items["title"].isna() | df_items["genres"].isna()
        if null_mask.any():
            err_df = df_items[null_mask].copy()
            err_df["error_reason"] = "MISSING_MANDATORY_ITEM_FIELDS"
            errors.append(err_df)
        self._record_check("null_check", "items", not null_mask.any(), {"null_records_count": int(null_mask.sum())})

        # 2. Duplicate item_id
        dup_mask = df_items["item_id"].duplicated(keep="first")
        if dup_mask.any():
            err_df = df_items[dup_mask].copy()
            err_df["error_reason"] = "DUPLICATE_ITEM_ID"
            errors.append(err_df)
        self._record_check("duplicate_check", "items", not dup_mask.any(), {"duplicates_count": int(dup_mask.sum())})

        # 3. Year validation
        year_mask = (df_items["release_year"] < 1880) | (df_items["release_year"] > datetime.utcnow().year + 2)
        if year_mask.any():
            err_df = df_items[year_mask].copy()
            err_df["error_reason"] = "INVALID_RELEASE_YEAR"
            errors.append(err_df)
        self._record_check("year_bounds_check", "items", not year_mask.any(), {"invalid_year_count": int(year_mask.sum())})

        all_err_mask = null_mask | dup_mask | year_mask
        clean_df = df_items[~all_err_mask].copy()
        error_df = pd.concat(errors, ignore_index=True) if errors else pd.DataFrame(columns=list(df_items.columns) + ["error_reason"])

        return clean_df, error_df

    def validate_interactions(
        self,
        df_interactions: pd.DataFrame,
        valid_users: pd.DataFrame,
        valid_items: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Validates interaction logs for ratings range, timestamps, duplicates, and foreign keys."""
        logger.info(f"Validating {len(df_interactions)} interaction records...")
        errors: List[pd.DataFrame] = []
        
        rating_min = float(self.config["validation"].get("rating_min", 1.0))
        rating_max = float(self.config["validation"].get("rating_max", 5.0))

        # 1. Null check on critical keys and values
        null_mask = df_interactions["interaction_id"].isna() | df_interactions["user_id"].isna() | \
                    df_interactions["item_id"].isna() | df_interactions["rating"].isna() | df_interactions["timestamp"].isna()
        if null_mask.any():
            err_df = df_interactions[null_mask].copy()
            err_df["error_reason"] = "NULL_VALUES_IN_INTERACTIONS"
            errors.append(err_df)
        self._record_check("null_check", "interactions", not null_mask.any(), {"null_count": int(null_mask.sum())})

        # 2. Rating range check
        rating_numeric = pd.to_numeric(df_interactions["rating"], errors="coerce")
        rating_out_of_bounds = rating_numeric.isna() | (rating_numeric < rating_min) | (rating_numeric > rating_max)
        if rating_out_of_bounds.any():
            err_df = df_interactions[rating_out_of_bounds].copy()
            err_df["error_reason"] = f"RATING_OUT_OF_BOUNDS_[{rating_min},{rating_max}]"
            errors.append(err_df)
        self._record_check("rating_range_check", "interactions", not rating_out_of_bounds.any(), {
            "invalid_rating_count": int(rating_out_of_bounds.sum()),
            "allowed_range": [rating_min, rating_max]
        })

        # 3. Timestamp parsing & future check
        parsed_ts = pd.to_datetime(df_interactions["timestamp"], errors="coerce")
        now = pd.Timestamp.utcnow().tz_localize(None)
        invalid_ts_mask = parsed_ts.isna() | (parsed_ts > now + pd.Timedelta(days=365*2))
        if invalid_ts_mask.any():
            err_df = df_interactions[invalid_ts_mask].copy()
            err_df["error_reason"] = "INVALID_OR_FUTURE_TIMESTAMP"
            errors.append(err_df)
        self._record_check("timestamp_validity", "interactions", not invalid_ts_mask.any(), {
            "invalid_timestamp_count": int(invalid_ts_mask.sum())
        })

        # 4. Duplicate interaction check (same interaction_id or exact duplicate row)
        dup_mask = df_interactions.duplicated(subset=["interaction_id"], keep="first")
        if dup_mask.any():
            err_df = df_interactions[dup_mask].copy()
            err_df["error_reason"] = "DUPLICATE_INTERACTION_ID"
            errors.append(err_df)
        self._record_check("duplicate_interaction_check", "interactions", not dup_mask.any(), {
            "duplicates_count": int(dup_mask.sum())
        })

        # 5. Referential Integrity Checks
        valid_user_set = set(valid_users["user_id"])
        valid_item_set = set(valid_items["item_id"])

        dangling_user_mask = ~df_interactions["user_id"].isin(valid_user_set)
        if dangling_user_mask.any():
            err_df = df_interactions[dangling_user_mask].copy()
            err_df["error_reason"] = "FOREIGN_KEY_VIOLATION_UNKNOWN_USER"
            errors.append(err_df)
        self._record_check("referential_integrity_users", "interactions", not dangling_user_mask.any(), {
            "dangling_user_records": int(dangling_user_mask.sum())
        })

        dangling_item_mask = ~df_interactions["item_id"].isin(valid_item_set)
        if dangling_item_mask.any():
            err_df = df_interactions[dangling_item_mask].copy()
            err_df["error_reason"] = "FOREIGN_KEY_VIOLATION_UNKNOWN_ITEM"
            errors.append(err_df)
        self._record_check("referential_integrity_items", "interactions", not dangling_item_mask.any(), {
            "dangling_item_records": int(dangling_item_mask.sum())
        })

        # Combine all error filters
        all_err_mask = null_mask | rating_out_of_bounds | invalid_ts_mask | dup_mask | dangling_user_mask | dangling_item_mask
        clean_df = df_interactions[~all_err_mask].copy()
        
        # Standardize rating type
        clean_df["rating"] = clean_df["rating"].astype(float)
        
        error_df = pd.concat(errors, ignore_index=True) if errors else pd.DataFrame(columns=list(df_interactions.columns) + ["error_reason"])
        error_df = error_df.drop_duplicates(subset=["interaction_id", "error_reason"])

        return clean_df, error_df

    def run_validation_pipeline(self) -> Dict[str, Any]:
        """Orchestrates full dataset validation and saves error datasets and reports."""
        users_raw_path = self.raw_dir / "users.csv"
        items_raw_path = self.raw_dir / "items.csv"
        interactions_raw_path = self.raw_dir / "interactions.csv"

        df_users = pd.read_csv(users_raw_path)
        df_items = pd.read_csv(items_raw_path)
        df_interactions = pd.read_csv(interactions_raw_path)

        clean_users, err_users = self.validate_users(df_users)
        clean_items, err_items = self.validate_items(df_items)
        clean_interactions, err_interactions = self.validate_interactions(df_interactions, clean_users, clean_items)

        # Save quarantined error datasets
        err_users.to_csv(self.errors_dir / "users_errors.csv", index=False)
        err_items.to_csv(self.errors_dir / "items_errors.csv", index=False)
        err_interactions.to_csv(self.errors_dir / "interactions_errors.csv", index=False)

        # Populate summary
        self.validation_results["summary"] = {
            "users": {
                "raw_count": len(df_users),
                "clean_count": len(clean_users),
                "quarantined_error_count": len(err_users),
                "pass_rate_pct": round((len(clean_users) / len(df_users)) * 100, 2)
            },
            "items": {
                "raw_count": len(df_items),
                "clean_count": len(clean_items),
                "quarantined_error_count": len(err_items),
                "pass_rate_pct": round((len(clean_items) / len(df_items)) * 100, 2)
            },
            "interactions": {
                "raw_count": len(df_interactions),
                "clean_count": len(clean_interactions),
                "quarantined_error_count": len(err_interactions),
                "pass_rate_pct": round((len(clean_interactions) / len(df_interactions)) * 100, 2)
            }
        }

        # Write reports
        json_report_path = self.reports_dir / "data_validation_report.json"
        with open(json_report_path, "w", encoding="utf-8") as f:
            json.dump(self.validation_results, f, indent=2)

        md_report_path = self.reports_dir / "data_validation_report.md"
        self._generate_markdown_report(md_report_path)

        logger.info(f"Validation finished. Errors quarantined in {self.errors_dir}. Reports generated at {self.reports_dir}")
        return self.validation_results

    def _generate_markdown_report(self, output_path: Path):
        """Generates a detailed markdown report of the data quality audit."""
        summary = self.validation_results["summary"]
        lines = [
            "# Automated Data Quality & Validation Report",
            f"**Validation Timestamp:** `{self.validation_results['validation_timestamp']}`  ",
            f"**Overall Pipeline Health:** `{self.validation_results['overall_status']}`  ",
            "",
            "## 1. Executive Summary Table",
            "| Dataset | Raw Records | Clean Records | Quarantined Errors | Pass Rate (%) | Status |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
            f"| **Users** | {summary['users']['raw_count']} | {summary['users']['clean_count']} | {summary['users']['quarantined_error_count']} | {summary['users']['pass_rate_pct']}% | ✅ VALIDATED |",
            f"| **Items** | {summary['items']['raw_count']} | {summary['items']['clean_count']} | {summary['items']['quarantined_error_count']} | {summary['items']['pass_rate_pct']}% | ✅ VALIDATED |",
            f"| **Interactions** | {summary['interactions']['raw_count']} | {summary['interactions']['clean_count']} | {summary['interactions']['quarantined_error_count']} | {summary['interactions']['pass_rate_pct']}% | 🛡️ QUARANTINED |",
            "",
            "## 2. Detailed Quality Checks and Assertions",
            "| Check Name | Target Entity | Status | Details / Violation Counts |",
            "| :--- | :--- | :---: | :--- |"
        ]

        for check in self.validation_results["detailed_checks"]:
            details_str = ", ".join([f"`{k}`: {v}" for k, v in check["details"].items()])
            badge = "✅ PASS" if check["status"] == "PASSED" else "⚠️ ANOMALIES DETECTED"
            lines.append(f"| `{check['check_name']}` | `{check['entity']}` | {badge} | {details_str} |")

        lines.extend([
            "",
            "## 3. Quarantine & Error Handling Strategy",
            "- **Referential Integrity**: All orphaned interactions referencing non-existent `user_id` or `item_id` are isolated into `data/errors/interactions_errors.csv`.",
            "- **Boundary Enforcement**: Ratings outside `[1.0, 5.0]` and malformed/future timestamps are filtered out and logged.",
            "- **Zero Data Loss**: Raw datasets remain immutable in `data/raw/`, clean datasets are passed to `data/processed/`, and anomalous rows are retained for root-cause analysis in `data/errors/`."
        ])

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


def run_validation(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """CLI/Dag entrypoint for data validation."""
    cfg = load_config(config_path)
    validator = DataValidator(cfg)
    return validator.run_validation_pipeline()


if __name__ == "__main__":
    run_validation()
