import streamlit as st
import pandas as pd
import requests
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender System")

# Read API key from Streamlit secrets / environment
API_KEY = os.getenv("TMDB_API_KEY")

# ---------------------- DATA PREPARATION ----------------------

@st.cache_data(show_spinner=True)
def prepare_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")

    movies = movies.merge(credits, on="title")
    movies = movies[['movie_id', 'title', 'overview']].dropna()

    cv = CountVectorizer(stop_words="english", max_features=5000)
    vectors = cv.fit_transform(movies['overview']).toarray()

    similarity = cosine_similarity(vectors)
    return movies, similarity


movies, similarity = prepare_data()

# ---------------------- POSTER FETCH ----------------------

def fetch_poster(movie_id):
    if not API_KEY:
        return "https://via.placeholder.com/500x750?text=No+API+Key"

    url = f"https://api.themoviedb.org/3/movie/{int(movie_id)}"
    params = {"api_key": API_KEY, "language": "en-US"}

    try:
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        data = r.json()

        if data.get("poster_path"):
            return "https://image.tmdb.org/t/p/w500" + data["poster_path"]

    except Exception:
        pass

    return "https://via.placeholder.com/500x750?text=No+Poster"

# ---------------------- RECOMMENDER ----------------------

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, reverse=True, key=lambda x: x[1])[1:6]

    names, posters = [], []
    for i in scores:
        row = movies.iloc[i[0]]
        names.append(row.title)
        posters.append(fetch_poster(row.movie_id))

    return names, posters

# ---------------------- UI ----------------------

movie_list = movies['title'].values
selected_movie = st.selectbox("Select a movie", movie_list)

if st.button("🎯 Show Recommendation"):
    names, posters = recommend(selected_movie)
    cols = st.columns(5)

    for i in range(5):
        with cols[i]:
            st.caption(names[i])
            st.image(posters[i], use_container_width=True)
