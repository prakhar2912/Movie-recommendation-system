import streamlit as st
import pandas as pd
import requests
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ---------------- CSS ----------------
st.markdown("""
<style>
.movie-card {
    background-color: #1e1e1e;
    padding: 12px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 6px 15px rgba(0,0,0,0.4);
}
.movie-title {
    font-size: 15px;
    font-weight: 600;
    margin-top: 8px;
}
.subtitle {
    text-align:center;
    font-size:18px;
    color:#b3b3b3;
}
.footer {
    text-align:center;
    color:gray;
    font-size:13px;
    margin-top:40px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<h1 style='text-align:center;'>🎬 Movie Recommender System</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Find movies similar to your favorites using Machine Learning</p>", unsafe_allow_html=True)
st.divider()

# ---------------- API KEY ----------------
API_KEY = os.getenv("TMDB_API_KEY")

# ---------------- DATA PREP ----------------
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

# ---------------- POSTER ----------------
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

    except:
        pass

    return "https://via.placeholder.com/500x750?text=No+Poster"

# ---------------- RECOMMENDER ----------------
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

# ---------------- UI ----------------
movie_list = movies['title'].values
selected_movie = st.selectbox("🎥 Select a movie", movie_list)

if st.button("✨ Recommend Movies"):
    with st.spinner("Finding the best recommendations..."):
        names, posters = recommend(selected_movie)

    st.subheader("🍿 Recommended for you")
    cols = st.columns(5)

    for i in range(5):
        with cols[i]:
            st.markdown("<div class='movie-card'>", unsafe_allow_html=True)
            st.image(posters[i], use_container_width=True)
            st.markdown(f"<div class='movie-title'>{names[i]}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
Built with ❤️ using Streamlit & TMDB API<br>
Content-Based Movie Recommendation System
</div>
""", unsafe_allow_html=True)
