# Personalized Movie Recommendation System

An end-to-end movie recommender built with **Python, Pandas, NumPy, scikit-learn, and Streamlit**. The repository starts **without a bundled dataset**. On first launch, it downloads the public MovieLens Latest Small dataset from GroupLens, preprocesses it, and stores normalized CSV files locally.

## Features

- **Zero-data setup:** no dataset is committed to GitHub; data is downloaded on first run.
- **Content-based filtering:** TF-IDF over movie titles and genres, plus any available metadata.
- **Collaborative filtering:** item-item cosine similarity from user-rating histories.
- **Hybrid ranking:** blends content affinity, collaborative signals, and a popularity prior.
- **Personalization:** recommendations for a selected MovieLens user or from favorite movies.
- **Streamlit UI:** recommendation dashboard, catalog search, similar-movie lookup, and CSV export.
- **Evaluation utilities:** Precision@K, Recall@K, and NDCG@K.
- **Repeatable ingestion:** rerunning the downloader skips an existing normalized dataset.

## Quick start

Python 3.10+ recommended. An internet connection is required for the initial dataset download.

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

macOS/Linux:
```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On first launch, the app downloads MovieLens Latest Small and writes the normalized files to `data/movielens/`. Later launches reuse those files.

To download the data manually instead:
```bash
python scripts/download_movielens.py
streamlit run app.py
```

## Data source and schema

The project uses the public **MovieLens Latest Small** dataset from GroupLens:
https://grouplens.org/datasets/movielens/latest/

The download script converts it to:
- `data/movielens/movies.csv`: `movie_id,title,genres,overview,keywords,cast,director`
- `data/movielens/ratings.csv`: `user_id,movie_id,rating,timestamp`

MovieLens Latest Small does not provide plot summaries, keywords, cast, or director in its core files. Those columns are left empty, so content recommendations are driven primarily by title and genre until additional metadata is sourced from a properly licensed provider.

## Project structure

```text
personalized_movie_recommender/
├── app.py
├── requirements.txt
├── README.md
├── PROJECT_OVERVIEW.md
├── data/                         # Empty in the Git repository
├── src/
│   └── recommender.py
├── scripts/
│   └── download_movielens.py
└── tests/
    └── test_recommender.py
```

## Pipeline

1. Download raw MovieLens archive from GroupLens.
2. Normalize movie and rating column names and save local CSV files.
3. Clean missing values, duplicate movie IDs, invalid ratings, and orphan rating rows.
4. Generate TF-IDF metadata features.
5. Fit item-item collaborative similarities from the user-item rating matrix.
6. Blend content, collaborative, and popularity scores into Top-K recommendations.
7. Display recommendations in Streamlit and optionally export them to CSV.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests use small in-memory fixtures; they do not need the dataset or internet access.

## GitHub publishing

```bash
git init
git add .
git commit -m "Build personalized movie recommendation system"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/personalized-movie-recommender.git
git push -u origin main
```

The `.gitignore` excludes downloaded data and archives so the repository stays lightweight.

## Limitations and next steps

- First launch requires internet access to download the dataset.
- The dataset is a public benchmark dataset, not live streaming or commercial catalog data.
- MovieLens Latest Small has ratings and genres but no plot/cast metadata in its core release.
- A production system should add robust offline evaluation, cold-start handling, caching, model monitoring, and privacy-aware user history management.
