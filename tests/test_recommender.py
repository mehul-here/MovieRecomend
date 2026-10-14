import unittest
import pandas as pd

from src.recommender import HybridRecommender, precision_recall_at_k, ndcg_at_k, prepare_data


class RecommenderTests(unittest.TestCase):
    def setUp(self):
        self.movies = pd.DataFrame([
            {"movie_id": 1, "title": "Space Voyage", "genres": "Sci-Fi Adventure", "overview": "A journey through space", "keywords": "space galaxy", "cast": "Actor A", "director": "Director A"},
            {"movie_id": 2, "title": "Galaxy Mission", "genres": "Sci-Fi Adventure", "overview": "A mission among the stars", "keywords": "space stars", "cast": "Actor B", "director": "Director A"},
            {"movie_id": 3, "title": "Kitchen Stories", "genres": "Comedy Drama", "overview": "Friends open a restaurant", "keywords": "food friends", "cast": "Actor C", "director": "Director B"},
            {"movie_id": 4, "title": "Laughing Chef", "genres": "Comedy", "overview": "A chef gets into trouble", "keywords": "food comedy", "cast": "Actor D", "director": "Director C"},
        ])
        self.ratings = pd.DataFrame([
            {"user_id": 1, "movie_id": 1, "rating": 5.0, "timestamp": 1},
            {"user_id": 1, "movie_id": 2, "rating": 4.5, "timestamp": 2},
            {"user_id": 1, "movie_id": 3, "rating": 2.0, "timestamp": 3},
            {"user_id": 2, "movie_id": 1, "rating": 4.5, "timestamp": 1},
            {"user_id": 2, "movie_id": 2, "rating": 4.0, "timestamp": 2},
            {"user_id": 2, "movie_id": 4, "rating": 3.5, "timestamp": 3},
        ])
        self.model = HybridRecommender().fit(self.movies, self.ratings)

    def test_similar_movies_excludes_input(self):
        result = self.model.similar_movies(1, k=3)
        self.assertNotIn(1, result["movie_id"].tolist())
        self.assertGreater(len(result), 0)

    def test_user_recommendations_exclude_rated(self):
        result = self.model.recommend_for_user(1, k=5)
        self.assertNotIn(1, result["movie_id"].tolist())
        self.assertNotIn(2, result["movie_id"].tolist())

    def test_recommend_from_likes(self):
        result = self.model.recommend_from_likes([1], k=3)
        self.assertNotIn(1, result["movie_id"].tolist())
        self.assertGreater(len(result), 0)

    def test_metrics(self):
        precision, recall = precision_recall_at_k([1, 2, 3], [2, 3], k=2)
        self.assertAlmostEqual(precision, 0.5)
        self.assertAlmostEqual(recall, 0.5)
        self.assertGreater(ndcg_at_k([2, 1, 3], [2, 3], k=3), 0)

    def test_prepare_data_filters_bad_ratings(self):
        ratings = pd.concat([self.ratings, pd.DataFrame([{"user_id": 9, "movie_id": 1, "rating": 9, "timestamp": 0}])])
        _, clean = prepare_data(self.movies, ratings)
        self.assertTrue(clean["rating"].between(0.5, 5.0).all())


if __name__ == "__main__":
    unittest.main()
