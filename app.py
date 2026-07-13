import pickle
import streamlit as st
import requests
import pandas as pd

# Set page configuration
st.set_page_config(page_title="Movie Recommender", page_icon="🎬")

# ----------------------------------------------------
# 1. Poster Fetching Logic (TMDb API)
# ----------------------------------------------------
def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
    try:
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            data = response.json()
            poster_path = data.get('poster_path')
            if poster_path:
                return "https://image.tmdb.org/t/p/w500/" + poster_path
    except Exception:
        pass
    # Fallback default cinema image
    return "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?q=80&w=300&auto=format&fit=crop"


# ----------------------------------------------------
# 2. Recommendation Logic (Using loaded similarity matrix)
# ----------------------------------------------------
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        # fetch the movie poster
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)

    return recommended_movie_names, recommended_movie_posters


# ----------------------------------------------------
# 3. Model & Data Loading (movies2.pkl & similarity.pkl)
# ----------------------------------------------------
st.header('Movie Recommender System')

# Load movie list from movies2.pkl directly
movies = None
try:
    with open('movies2.pkl', 'rb') as f:
        data = pickle.load(f)
        if isinstance(data, dict):
            movies = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            movies = data
        else:
            movies = pd.DataFrame(data)
except Exception as e:
    st.error(f"Error loading 'movies2.pkl'. Please make sure it is in this folder. Error: {e}")
    st.stop()

# Load similarity matrix directly from similarity.pkl
similarity = None
try:
    with open('similarity.pkl', 'rb') as f:
        similarity = pickle.load(f)
except Exception as e:
    st.error(f"Error loading 'similarity.pkl'. Please make sure it is in this folder. Error: {e}")
    st.stop()


# ----------------------------------------------------
# 4. User Interface
# ----------------------------------------------------
movie_list = movies['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

if st.button('Show Recommendation'):
    recommended_movie_names, recommended_movie_posters = recommend(selected_movie)
    
    # Render recommendations in 5 columns
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.text(recommended_movie_names[i])
            st.image(recommended_movie_posters[i], use_container_width=True)