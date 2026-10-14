from pathlib import Path
import os

import pandas as pd
import streamlit as st

from src.recommender import HybridRecommender, load_data

st.set_page_config(page_title="CineMatch | Movie Recommender", page_icon="🎬", layout="wide")

st.title("CineMatch")
st.caption("Personalized movie recommendations powered by content-based filtering and collaborative signals.")

data_dir = Path(os.getenv("MOVIE_DATA_DIR", "data/movielens"))
required_files = [data_dir / "movies.csv", data_dir / "ratings.csv"]

if not all(path.exists() for path in required_files):
    st.info("No dataset found yet. Downloading the public MovieLens Latest Small dataset for the first run...")
    try:
        from scripts.download_movielens import download_and_prepare
        with st.spinner("Downloading and preparing MovieLens data..."):
            download_and_prepare()
    except Exception as exc:
        st.error(f"Dataset setup failed: {exc}")
        st.markdown(
            "Check your internet connection and try again, or run "
            "`python scripts/download_movielens.py` from the project directory."
        )
        st.stop()

try:
    movies, ratings = load_data(data_dir)
except Exception as exc:
    st.error(f"Could not load data from `{data_dir}`: {exc}")
    st.info("Try running `python scripts/download_movielens.py` from the project directory.")
    st.stop()

@st.cache_resource(show_spinner="Training recommendation models...")
def train_model(movie_frame, rating_frame):
    model = HybridRecommender().fit(movie_frame, rating_frame)
    return model

# Cache on data content to avoid stale model after changing dataset.
model = train_model(movies.to_json(), ratings.to_json()) if False else None
@st.cache_resource
def get_model(data_path, movie_signature, rating_signature):
    m, r = load_data(data_path)
    return HybridRecommender().fit(m, r)

model = get_model(str(data_dir), str(len(movies)) + str(movies["title"].sum()), str(len(ratings)) + str(ratings["rating"].sum()))

with st.sidebar:
    st.header("Recommendation settings")
    method = st.radio("Personalize using", ["Demo user profile", "Movies I like"])
    top_k = st.slider("Number of recommendations", min_value=5, max_value=20, value=10, step=5)
    st.divider()
    st.metric("Movies", f"{len(movies):,}")
    st.metric("Ratings", f"{len(ratings):,}")
    st.metric("Users", f"{ratings['user_id'].nunique():,}")

tab1, tab2, tab3 = st.tabs(["For you", "Explore movies", "Similar movies"])

with tab1:
    st.subheader("Your recommendations")
    if method == "Demo user profile":
        users = sorted(ratings["user_id"].unique().tolist())
        if users:
            chosen_user = st.selectbox("Choose a demo user", users, format_func=lambda x: f"User {x}")
            history = ratings[ratings["user_id"] == chosen_user].merge(movies[["movie_id", "title"]], on="movie_id")
            if not history.empty:
                st.write("Rating history")
                display_history = history.sort_values("rating", ascending=False)[["title", "rating"]]
                st.dataframe(display_history, use_container_width=True, hide_index=True)
            recs = model.recommend_for_user(int(chosen_user), k=top_k)
        else:
            st.warning("No user ratings available. Switch to Movies I like.")
            recs = pd.DataFrame()
    else:
        options = movies.sort_values("title")
        selected_titles = st.multiselect(
            "Select a few movies you enjoy",
            options=options["title"].tolist(),
            default=options["title"].head(2).tolist(),
            help="Pick at least one movie to personalize the list."
        )
        ids = options[options["title"].isin(selected_titles)]["movie_id"].tolist()
        recs = model.recommend_from_likes(ids, k=top_k) if ids else pd.DataFrame()
    if not recs.empty:
        st.dataframe(recs[["title", "genres", "score", "reason"]], use_container_width=True, hide_index=True)
        st.download_button("Download recommendations as CSV", recs.to_csv(index=False), "recommendations.csv", "text/csv")
    else:
        st.info("Choose a user with rating history or select favorite movies.")

with tab2:
    st.subheader("Explore the catalog")
    search = st.text_input("Search by title or genre")
    catalog = movies.copy()
    if search:
        mask = catalog["title"].str.contains(search, case=False, na=False) | catalog["genres"].str.contains(search, case=False, na=False)
        catalog = catalog[mask]
    catalog = catalog.merge(
        ratings.groupby("movie_id").agg(rating_count=("rating", "count"), average_rating=("rating", "mean")).reset_index(),
        on="movie_id", how="left"
    )
    catalog["rating_count"] = catalog["rating_count"].fillna(0).astype(int)
    catalog["average_rating"] = catalog["average_rating"].round(2)
    st.dataframe(catalog[["title", "genres", "average_rating", "rating_count"]], use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Find movies similar to a favorite")
    title = st.selectbox("Choose a movie", sorted(movies["title"].tolist()), key="similar_movie")
    movie_id = int(movies.loc[movies["title"] == title, "movie_id"].iloc[0])
    similar = model.similar_movies(movie_id, k=top_k)
    st.dataframe(similar[["title", "genres", "score"]], use_container_width=True, hide_index=True)

st.divider()
st.caption("Dataset: MovieLens Latest Small, downloaded from GroupLens on first run. Ratings and movie metadata are stored locally in data/movielens/.")
