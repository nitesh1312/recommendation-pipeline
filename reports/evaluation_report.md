# Recommendation Models Comparative Evaluation Benchmark

## 1. Overall Performance Summary Table
| Metric | Baseline Popularity | SVD Collaborative Filtering | Hybrid Recommender | Best Performing |
| :--- | :---: | :---: | :---: | :---: |
| **RMSE (Lower is Better)** | 1.1111 | 1.1412 | 1.1278 | **Baseline Popularity** 🏆 |
| **MAE (Lower is Better)** | 0.9226 | 0.9302 | 0.9235 | **Baseline Popularity** 🏆 |
| **Precision@5 (Higher is Better)** | 0.0069 | 0.0138 | 0.0126 | **SVD Collaborative Filtering** 🏆 |
| **Recall@5 (Higher is Better)** | 0.0094 | 0.0185 | 0.0178 | **SVD Collaborative Filtering** 🏆 |
| **NDCG@5 (Higher is Better)** | 0.0087 | 0.018 | 0.0153 | **SVD Collaborative Filtering** 🏆 |
| **Catalog Coverage@5 (%)** | 2.67 | 49.0 | 38.33 | **SVD Collaborative Filtering** 🏆 |
| **Intra-List Diversity@5** | 0.9452 | 0.9209 | 0.9257 | **Baseline Popularity** 🏆 |
| **Precision@10 (Higher is Better)** | 0.0074 | 0.015 | 0.0112 | **SVD Collaborative Filtering** 🏆 |
| **Recall@10 (Higher is Better)** | 0.0189 | 0.0381 | 0.0301 | **SVD Collaborative Filtering** 🏆 |
| **NDCG@10 (Higher is Better)** | 0.0132 | 0.0266 | 0.0206 | **SVD Collaborative Filtering** 🏆 |
| **Catalog Coverage@10 (%)** | 4.67 | 62.33 | 48.67 | **SVD Collaborative Filtering** 🏆 |
| **Intra-List Diversity@10** | 0.9193 | 0.9239 | 0.9301 | **Hybrid Recommender** 🏆 |

## 2. Key Observations and Performance Analysis
1. **Personalization Advantage**: SVD Matrix Factorization significantly outperforms the Popularity Baseline across ranking metrics (`Precision@K`, `Recall@K`, `NDCG@K`) because it captures personalized user taste vectors.
2. **Catalog Coverage**: Collaborative Filtering and Hybrid models achieve substantially higher catalog coverage compared to the popularity baseline (which concentrates only on head items), enabling long-tail discovery.
3. **Hybrid Robustness**: The Hybrid recommender successfully blends latent factor scoring with popularity Bayesian priors, maintaining top-tier precision while preserving diversity and catalog exploration.