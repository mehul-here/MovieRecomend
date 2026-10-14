"""Data preparation and hybrid recommendation models."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


MOVIE_COLUMNS = ["movie_id", "title", "genres", "overview", "keywords", "cast", "director"]
RATING_COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]


def load_data(data_dir: str | Path = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load normalized movie and rating CSVs from a directory."""
    data_dir = Path(data_dir)
    movies = pd.read_csv(data_dir / "movies.csv")
    ratings = pd.read_csv(data_dir / "ratings.csv")
    return prepare_data(movies, ratings)


def prepare_data(
    movies: pd.DataFrame, ratings: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean and validate movie metadata and user ratings."""
    movies = movies.copy()
    ratings = ratings.copy()

    for col in MOVIE_COLUMNS:
        if col not in movies.columns:
            movies[col] = ""
    for col in RATING_COLUMNS:
        if col not in ratings.columns:
            ratings[col] = np.nan

    movies["movie_id"] = pd.to_numeric(movies["movie_id"], errors="coerce")
    movies = movies.dropna(subset=["movie_id"]).drop_duplicates("movie_id")
    movies["movie_id"] = movies["movie_id"].astype(int)
    for col in ["title", "genres", "overview", "keywords", "cast", "director"]:
        movies[col] = movies[col].fillna("").astype(str).str.strip()
    movies["title"] = movies["title"].replace("", "Untitled")

    ratings["user_id"] = pd.to_numeric(ratings["user_id"], errors="coerce")
    ratings["movie_id"] = pd.to_numeric(ratings["movie_id"], errors="coerce")
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = ratings.dropna(subset=["user_id", "movie_id", "rating"])
    ratings = ratings[ratings["rating"].between(0.5, 5.0)]
    ratings["user_id"] = ratings["user_id"].astype(int)
    ratings["movie_id"] = ratings["movie_id"].astype(int)
    ratings = ratings[ratings["movie_id"].isin(movies["movie_id"])]
    ratings = ratings.drop_duplicates(["user_id", "movie_id"], keep="last")
    if "timestamp" not in ratings.columns:
        ratings["timestamp"] = 0
    ratings["timestamp"] = pd.to_numeric(ratings["timestamp"], errors="coerce").fillna(0)
    return movies.reset_index(drop=True), ratings.reset_index(drop=True)


@dataclass
class Recommendation:
    movie_id: int
    title: str
    score: float
    genres: str
    reason: str = ""


class HybridRecommender:
    """Content-based + item-item collaborative hybrid recommender."""

    def __init__(
        self,
        content_weight: float = 0.45,
        collaborative_weight: float = 0.45,
        popularity_weight: float = 0.10,
    ):
        weights = np.array([content_weight, collaborative_weight, popularity_weight], dtype=float)
        if np.any(weights < 0) or weights.sum() == 0:
            raise ValueError("Weights must be non-negative and sum to a positive value.")
        self.weights = weights / weights.sum()
        self.movies: Optional[pd.DataFrame] = None
        self.ratings: Optional[pd.DataFrame] = None
        self.movie_index: dict[int, int] = {}
        self.content_matrix = None
        self.content_similarity = None
        self.collab_similarity = None
        self.user_item = None
        self.popularity = None
        self.tfidf = None

    def fit(self, movies: pd.DataFrame, ratings: pd.DataFrame) -> "HybridRecommender":
        self.movies, self.ratings = prepare_data(movies, ratings)
        self.movie_index = {int(mid): i for i, mid in enumerate(self.movies["movie_id"])}
        text_fields = ["title", "genres", "overview", "keywords", "cast", "director"]
        metadata = self.movies[text_fields].fillna("").agg(" ".join, axis=1)
        metadata = metadata.str.replace(r"[^A-Za-z0-9 ]+", " ", regex=True).str.lower()
        self.tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        self.content_matrix = self.tfidf.fit_transform(metadata)
        self.content_similarity = cosine_similarity(self.content_matrix, dense_output=True)

        if len(self.ratings):
            self.user_item = self.ratings.pivot_table(
                index="user_id", columns="movie_id", values="rating", aggfunc="mean"
            ).reindex(columns=self.movies["movie_id"])
            user_means = self.user_item.mean(axis=1)
            centered = self.user_item.sub(user_means, axis=0).fillna(0.0)
            # Cosine similarity between item-rating vectors; zero vectors remain safe.
            item_vectors = centered.to_numpy(dtype=float).T
            self.collab_similarity = cosine_similarity(item_vectors)
            counts = self.ratings.groupby("movie_id")["rating"].count()
            means = self.ratings.groupby("movie_id")["rating"].mean()
            popularity = means * np.log1p(counts)
        else:
            self.user_item = pd.DataFrame(columns=self.movies["movie_id"])
            self.collab_similarity = np.zeros((len(self.movies), len(self.movies)))
            popularity = pd.Series(dtype=float)

        pop = self.movies["movie_id"].map(popularity).fillna(0.0).to_numpy(dtype=float)
        if len(pop) and pop.max() > pop.min():
            self.popularity = (pop - pop.min()) / (pop.max() - pop.min())
        else:
            self.popularity = np.zeros(len(self.movies), dtype=float)
        return self

    def _movie_position(self, movie_id: int) -> Optional[int]:
        return self.movie_index.get(int(movie_id))

    def similar_movies(self, movie_id: int, k: int = 10) -> pd.DataFrame:
        """Return metadata-similar movies, excluding the input movie."""
        self._require_fit()
        pos = self._movie_position(movie_id)
        if pos is None:
            return pd.DataFrame(columns=["movie_id", "title", "genres", "score"])
        scores = self.content_similarity[pos].copy()
        scores[pos] = -1
        indices = np.argsort(scores)[::-1][:max(0, k)]
        result = self.movies.iloc[indices][["movie_id", "title", "genres"]].copy()
        result["score"] = scores[indices]
        return result.reset_index(drop=True)

    def recommend_for_user(
        self, user_id: int, k: int = 10, exclude_rated: bool = True
    ) -> pd.DataFrame:
        """Generate Top-K recommendations for an existing user."""
        self._require_fit()
        user_id = int(user_id)
        user_ratings = self.ratings[self.ratings["user_id"] == user_id]
        liked = user_ratings[user_ratings["rating"] >= 3.5]["movie_id"].tolist()
        candidate_content = self._content_affinity(liked)
        candidate_collab = self._collaborative_affinity(user_id)
        scores = self._combine(candidate_content, candidate_collab)
        if exclude_rated:
            rated = set(user_ratings["movie_id"].astype(int))
            for i, mid in enumerate(self.movies["movie_id"]):
                if int(mid) in rated:
                    scores[i] = -1
        return self._format_recommendations(scores, k, "Based on your rating history and similar movies")

    def recommend_from_likes(
        self, liked_movie_ids: Iterable[int], k: int = 10
    ) -> pd.DataFrame:
        """Generate recommendations from a user's explicitly selected favorite movies."""
        self._require_fit()
        liked = [int(mid) for mid in liked_movie_ids if int(mid) in self.movie_index]
        scores = self._combine(self._content_affinity(liked), self._collaborative_affinity(None, liked))
        for i, mid in enumerate(self.movies["movie_id"]):
            if int(mid) in set(liked):
                scores[i] = -1
        return self._format_recommendations(scores, k, "Based on your selected favorites")

    def _content_affinity(self, liked_ids: Iterable[int]) -> np.ndarray:
        n = len(self.movies)
        positions = [self.movie_index[mid] for mid in liked_ids if mid in self.movie_index]
        if not positions:
            return self.popularity.copy()
        return self.content_similarity[positions].mean(axis=0)

    def _collaborative_affinity(
        self, user_id: Optional[int], liked_ids: Optional[Iterable[int]] = None
    ) -> np.ndarray:
        n = len(self.movies)
        if self.collab_similarity is None or self.collab_similarity.shape != (n, n):
            return np.zeros(n)
        seed_ids = []
        if user_id is not None and self.user_item is not None and user_id in self.user_item.index:
            row = self.user_item.loc[user_id]
            seed_ids.extend(row[row >= 3.5].dropna().index.astype(int).tolist())
        if liked_ids is not None:
            seed_ids.extend(int(mid) for mid in liked_ids)
        positions = [self.movie_index[mid] for mid in set(seed_ids) if mid in self.movie_index]
        if not positions:
            return np.zeros(n)
        return self.collab_similarity[positions].mean(axis=0)

    def _combine(self, content: np.ndarray, collaborative: np.ndarray) -> np.ndarray:
        content = np.asarray(content, dtype=float)
        collaborative = np.asarray(collaborative, dtype=float)
        # Cosine similarities can be negative; clip before scaling to keep scores interpretable.
        content = np.clip(content, 0, None)
        collaborative = np.clip(collaborative, 0, None)
        def scale(x):
            if len(x) == 0 or np.max(x) == np.min(x):
                return np.zeros_like(x)
            return (x - np.min(x)) / (np.max(x) - np.min(x))
        wc, wcf, wp = self.weights
        return wc * scale(content) + wcf * scale(collaborative) + wp * self.popularity

    def _format_recommendations(self, scores: np.ndarray, k: int, reason: str) -> pd.DataFrame:
        valid = np.where(scores >= 0)[0]
        order = valid[np.argsort(scores[valid])[::-1]][:max(0, int(k))]
        result = self.movies.iloc[order][["movie_id", "title", "genres"]].copy()
        result["score"] = np.round(scores[order], 4)
        result["reason"] = reason
        return result.reset_index(drop=True)

    def _require_fit(self):
        if self.movies is None:
            raise RuntimeError("Call fit(movies, ratings) before requesting recommendations.")


def precision_recall_at_k(recommended: Iterable[int], relevant: Iterable[int], k: int = 10) -> tuple[float, float]:
    """Calculate Precision@K and Recall@K for a single recommendation list."""
    recs = list(recommended)[:k]
    relevant_set = set(relevant)
    hits = sum(1 for item in recs if item in relevant_set)
    precision = hits / len(recs) if recs else 0.0
    recall = hits / len(relevant_set) if relevant_set else 0.0
    return precision, recall


def ndcg_at_k(recommended: Iterable[int], relevant: Iterable[int], k: int = 10) -> float:
    """Binary relevance NDCG@K."""
    relevant_set = set(relevant)
    recs = list(recommended)[:k]
    dcg = sum(1 / np.log2(i + 2) for i, item in enumerate(recs) if item in relevant_set)
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1 / np.log2(i + 2) for i in range(ideal_hits))
    return float(dcg / idcg) if idcg else 0.0
