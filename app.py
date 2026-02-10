import streamlit as st
import pickle
import requests
import os

st.title("🎬 Movie Recommender System")

# Read API key from environment
API_Key = os.getenv("TMDB_API_KEY")

def fetch_poster(movie_id):
    if not API_Key:
        return "https://via.placeholder.com/500x750?text=No+API+Key"

    url = f"https://api.themoviedb.org/3/movie/{int(movie_id)}?api_key={API_Key}&language=en-US"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("poster_path"):
            return "https://image.tmdb.org/t/p/w500" + data["poster_path"]
        else:
            return "https://via.placeholder.com/500x750?text=No+Poster"

    except requests.exceptions.RequestException:
        return "https://via.placeholder.com/500x750?text=API+Error"


def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(
        list(enumerate(similarity[index])),
        reverse=True,
        key=lambda x: x[1]
    )

    names = []
    posters = []

    for i in distances[1:6]:
        movie_id = movies.iloc[i[0]].movie_id
        names.append(movies.iloc[i[0]].title)
        posters.append(fetch_poster(movie_id))

    return names, posters


# Load data
movies = pickle.load(open('movie_list.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

movie_list = movies['title'].values

selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

if st.button("Show Recommendation"):
    names, posters = recommend(selected_movie)

    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.text(names[i])
            st.image(posters[i])
