# 🎬 Movie Recommender System

A **Content-Based Movie Recommendation System** built using **Streamlit**, **Machine Learning**, and the **TMDB API**.  
This web app recommends movies similar to your favorite one by analyzing movie overviews using **cosine similarity**.

---

## 🚀 Live Demo
👉 https://<your-streamlit-app-link>

---

## 📌 Features

- 🎯 Content-based movie recommendations
- 🧠 Machine Learning using cosine similarity
- 🎥 Real-time movie posters via TMDB API
- ⚡ Fast & optimized with Streamlit caching
- 🌙 Clean dark-themed UI
- 📱 Fully responsive layout
- 🔐 Secure API key handling using environment variables

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit  
- **Backend / ML:** Python, Scikit-learn  
- **API:** TMDB (The Movie Database)  
- **Data Processing:** Pandas, CountVectorizer  
- **Deployment:** Streamlit Community Cloud  

---

## 🧠 How It Works

1. Movie metadata is loaded from the TMDB dataset.
2. Movie overviews are vectorized using **Bag of Words**.
3. **Cosine similarity** is calculated between movies.
4. When a user selects a movie, the system:
   - Finds similar movies
   - Fetches posters using TMDB API
   - Displays top 5 recommendations

---

## 📂 Project Structure


