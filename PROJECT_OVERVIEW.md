# Project Overview

**Problem:** Generic movie lists do not reflect an individual viewer's taste.

**Solution:** A hybrid recommender combines movie metadata similarity, collaborative signals from user-rating patterns, and a popularity prior.

**No-data-first workflow:** The repository contains no dataset. The first app launch downloads MovieLens Latest Small from GroupLens, transforms it into the project's normalized schema, and stores the files locally.

**Pipeline:** Public dataset ingestion → preprocessing → TF-IDF feature engineering → item-item collaborative similarity → hybrid Top-K ranking → Streamlit interface → CSV export.

**Primary technologies:** Python, Pandas, NumPy, scikit-learn, Streamlit.

**Dataset source:** https://grouplens.org/datasets/movielens/latest/
