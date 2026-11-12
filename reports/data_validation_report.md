# Automated Data Quality & Validation Report
**Validation Timestamp:** `2026-11-12T19:38:07.935215`  
**Overall Pipeline Health:** `WARNINGS_FOUND`  

## 1. Executive Summary Table
| Dataset | Raw Records | Clean Records | Quarantined Errors | Pass Rate (%) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Users** | 1000 | 1000 | 0 | 100.0% | ✅ VALIDATED |
| **Items** | 300 | 300 | 0 | 100.0% | ✅ VALIDATED |
| **Interactions** | 35175 | 34259 | 1049 | 97.4% | 🛡️ QUARANTINED |

## 2. Detailed Quality Checks and Assertions
| Check Name | Target Entity | Status | Details / Violation Counts |
| :--- | :--- | :---: | :--- |
| `null_check` | `users` | ✅ PASS | `null_records_count`: 0 |
| `duplicate_check` | `users` | ✅ PASS | `duplicates_count`: 0 |
| `age_range_check` | `users` | ✅ PASS | `invalid_age_count`: 0 |
| `null_check` | `items` | ✅ PASS | `null_records_count`: 0 |
| `duplicate_check` | `items` | ✅ PASS | `duplicates_count`: 0 |
| `year_bounds_check` | `items` | ✅ PASS | `invalid_year_count`: 0 |
| `null_check` | `interactions` | ⚠️ ANOMALIES DETECTED | `null_count`: 131 |
| `rating_range_check` | `interactions` | ⚠️ ANOMALIES DETECTED | `invalid_rating_count`: 262, `allowed_range`: [1.0, 5.0] |
| `timestamp_validity` | `interactions` | ⚠️ ANOMALIES DETECTED | `invalid_timestamp_count`: 131 |
| `duplicate_interaction_check` | `interactions` | ⚠️ ANOMALIES DETECTED | `duplicates_count`: 175 |
| `referential_integrity_users` | `interactions` | ⚠️ ANOMALIES DETECTED | `dangling_user_records`: 176 |
| `referential_integrity_items` | `interactions` | ⚠️ ANOMALIES DETECTED | `dangling_item_records`: 176 |

## 3. Quarantine & Error Handling Strategy
- **Referential Integrity**: All orphaned interactions referencing non-existent `user_id` or `item_id` are isolated into `data/errors/interactions_errors.csv`.
- **Boundary Enforcement**: Ratings outside `[1.0, 5.0]` and malformed/future timestamps are filtered out and logged.
- **Zero Data Loss**: Raw datasets remain immutable in `data/raw/`, clean datasets are passed to `data/processed/`, and anomalous rows are retained for root-cause analysis in `data/errors/`.