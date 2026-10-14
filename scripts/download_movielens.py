"""Download and normalize the public MovieLens Latest Small dataset."""
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile
import shutil

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "movielens"
ARCHIVE = ROOT / "data" / "ml-latest-small.zip"
URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def download_and_prepare():
    """Download MovieLens and write normalized CSVs. Safe to rerun."""
    (ROOT / "data").mkdir(exist_ok=True)
    if not (DATA_DIR / "movies.csv").exists() or not (DATA_DIR / "ratings.csv").exists():
        print(f"Downloading {URL}")
        urlretrieve(URL, ARCHIVE)
        extract_dir = ROOT / "data" / "_movielens_extract"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True)
        try:
            with ZipFile(ARCHIVE) as archive:
                archive.extractall(extract_dir)
            source = extract_dir / "ml-latest-small"
            DATA_DIR.mkdir(parents=True, exist_ok=True)

            movies = pd.read_csv(source / "movies.csv")
            ratings = pd.read_csv(source / "ratings.csv")
            normalized_movies = pd.DataFrame({
                "movie_id": movies["movieId"],
                "title": movies["title"],
                "genres": movies["genres"].replace("(no genres listed)", ""),
                "overview": "",
                "keywords": "",
                "cast": "",
                "director": "",
            })
            normalized_ratings = ratings.rename(columns={
                "userId": "user_id", "movieId": "movie_id"
            })[["user_id", "movie_id", "rating", "timestamp"]]
            normalized_movies.to_csv(DATA_DIR / "movies.csv", index=False)
            normalized_ratings.to_csv(DATA_DIR / "ratings.csv", index=False)
        finally:
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
        print(f"Normalized data written to {DATA_DIR}")
    else:
        print(f"Dataset already exists in {DATA_DIR}; skipping download.")


def main():
    download_and_prepare()
    print("Data setup complete. Run `streamlit run app.py` to launch the app.")


if __name__ == "__main__":
    main()
