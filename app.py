import streamlit as st
import pandas as pd
import requests
import os
import base64
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ---------------- BACKGROUND IMAGE ----------------
def add_bg_from_local(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.65);
            z-index: -1;
        }}

        .subtitle {{
            text-align: center;
            color: #cccccc;
            font-size: 18px;
            margin-bottom: 10px;
        }}

        .movie-card {{
            text-align: center;
            font-weight: 600;
            margin-top: 8px;
            color: #ffffff;
        }}

        .poster-img img {{
            border-radius: 16px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.6);
        }}

        .footer {{
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background: rgba(0,0,0,0.6);
            text-align: center;
            color: #b3b3b3;
            font-size: 14px;
            padding: 10px 0;
            z-index: 999;
        }}

        .footer span {{
            display: block;
            font-size: 13px;
            opacity: 0.85;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("Background_image.jpg")

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    Built with ❤️ using <b>Streamlit</b> & <b>TMDB API</b>
    <span>Content-Based Movie Recommendation System</span>
    <span>© Developed by <b>Prakhar Pandey</b></span>
</div>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<h1 style='text-align:center;'>🎬 Movie Recommender System</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Find movies similar to your favorites using Machine Learning</p>", unsafe_allow_html=True)
st.divider()

# ---------------- API KEY ----------------
API_KEY = os.getenv("TMDB_API_KEY")

# ---------------- DATA PREPARATION ----------------
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

# ---------------- POSTER FETCH ----------------
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
    names, posters = recommend(selected_movie)

    st.subheader("🍿 Recommended for you")

    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.markdown(
                f"""
                <div class="poster-img">
                    <img src="{posters[i]}" width="100%">
                </div>
                <div class="movie-card">
                    {names[i]}
                </div>
                """,
                unsafe_allow_html=True
            )
