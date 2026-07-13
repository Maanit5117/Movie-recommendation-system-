import streamlit as st
import pickle
import pandas as pd
import requests
import io
import hashlib
import os
from PIL import Image, ImageDraw, ImageFont
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page configuration
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

st.title('Movie Recommender System')

# ----------------------------------------------------
# 1. Model / Data Loading
# ----------------------------------------------------
movies_filenames = ['movies2.pkl', 'movie_dict.pkl', 'movies.pkl', 'movies_dict.pkl', 'movie_list.pkl']
movies = None

for fname in movies_filenames:
    if os.path.exists(fname):
        try:
            with open(fname, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    movies = pd.DataFrame(data)
                elif isinstance(data, pd.DataFrame):
                    movies = data
                else:
                    movies = pd.DataFrame(data)
                break
        except Exception:
            pass

if movies is None:
    st.error("Please place your movie list pickle file (e.g., movies2.pkl) in this folder.")
    st.stop()

# Auto-detect ID column
movie_id_col = 'movie_id' if 'movie_id' in movies.columns else 'id'

# Setup vectorizer and compute similarity on-the-fly (zero-config, lightweight)
@st.cache_resource
def get_movie_vectors(df):
    tags_col = 'tags' if 'tags' in df.columns else 'title'
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(df[tags_col].fillna('').apply(lambda x: str(x).lower()))
    return vectors

vectors = get_movie_vectors(movies)


# ----------------------------------------------------
# 2. Poster Fetching & Fallback Image Generator
# ----------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_poster(movie_id, api_key):
    """Fetch poster path from TMDB API."""
    if not api_key or not movie_id:
        return None
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
    try:
        response = requests.get(url, timeout=3).json()
        poster_path = response.get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500/{poster_path}"
    except Exception:
        pass
    return None

def generate_fallback_poster(title):
    """Draw a beautiful placeholder poster using PIL."""
    width, height = 300, 450
    # Create unique linear gradient colors based on title hash
    h = int(hashlib.md5(title.encode('utf-8')).hexdigest(), 16)
    gradients = [
        ((15, 32, 39), (44, 83, 100)),
        ((31, 28, 44), (146, 141, 171)),
        ((49, 16, 16), (92, 29, 29)),
        ((13, 58, 36), (30, 107, 69)),
        ((24, 24, 27), (63, 63, 70))
    ]
    color1, color2 = gradients[h % len(gradients)]
    img = Image.new('RGB', (width, height), color1)
    draw = ImageDraw.Draw(img)
    for y in range(height):
        r = int(color1[0] + (color2[0] - color1[0]) * (y / height))
        g = int(color1[1] + (color2[1] - color1[1]) * (y / height))
        b = int(color1[2] + (color2[2] - color1[2]) * (y / height))
        draw.line((0, y, width, y), fill=(r, g, b))
        
    border_color = (245, 158, 11) if h % 2 == 0 else (161, 161, 170)
    draw.rectangle([10, 10, width-10, height-10], outline=border_color, width=2)
    
    try:
        font_title = ImageFont.truetype('/System/Library/Fonts/Supplemental/Georgia.ttf', 20)
    except Exception:
        font_title = ImageFont.load_default()
        
    # Wrap text
    max_text_width = width - 40
    words = title.split()
    lines = []
    current_line = []
    for word in words:
        current_line.append(word)
        try:
            line_width = draw.textlength(" ".join(current_line), font=font_title)
        except AttributeError:
            line_width = len(" ".join(current_line)) * 10
        if line_width > max_text_width:
            if len(current_line) == 1:
                lines.append(current_line[0])
                current_line = []
            else:
                current_line.pop()
                lines.append(" ".join(current_line))
                current_line = [word]
    if current_line:
        lines.append(" ".join(current_line))
        
    title_height = len(lines) * 28
    start_y = 180
    for i, line in enumerate(lines):
        try:
            line_width = draw.textlength(line, font=font_title)
        except AttributeError:
            line_width = len(line) * 10
        draw.text(((width - line_width)/2, start_y + i*28), line, fill=(255, 255, 255), font=font_title)
        
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


# ----------------------------------------------------
# 3. Recommendation Logic
# ----------------------------------------------------
def recommend(movie):
    try:
        movie_idx = movies[movies['title'] == movie].index[0]
        # Cosine similarity for the selected movie
        similarity_scores = cosine_similarity(vectors[movie_idx], vectors)[0]
        distances = sorted(list(enumerate(similarity_scores)), reverse=True, key=lambda x: x[1])
        
        recommended_movie_names = []
        recommended_movie_posters = []
        
        count = 0
        for i in distances:
            # Skip the selected movie itself
            if i[0] == movie_idx:
                continue
            
            idx = i[0]
            recommended_movie_names.append(movies.iloc[idx]['title'])
            
            movie_id = movies.iloc[idx][movie_id_col]
            poster_url = fetch_poster(movie_id, api_key)
            recommended_movie_posters.append(poster_url if poster_url else movies.iloc[idx]['title'])
            
            count += 1
            if count == 5:
                break
                
        return recommended_movie_names, recommended_movie_posters
    except Exception as e:
        st.error(f"Could not calculate recommendations: {e}")
        return [], []


# ----------------------------------------------------
# 4. User Interface
# ----------------------------------------------------
api_key = st.sidebar.text_input("TMDB API Key (Optional)", type="password")
st.sidebar.caption("Provide an API Key from themoviedb.org to view actual movie posters.")

selected_movie_name = st.selectbox(
    'Select a movie to get recommendations:',
    movies['title'].values
)

if st.button('Recommend'):
    names, posters = recommend(selected_movie_name)
    
    if names:
        cols = st.columns(5)
        for i in range(5):
            with cols[i]:
                # If poster is URL, show TMDB image. Otherwise generate graphic
                if isinstance(posters[i], str) and posters[i].startswith("http"):
                    st.image(posters[i], use_container_width=True)
                else:
                    fallback_bytes = generate_fallback_poster(names[i])
                    st.image(fallback_bytes, use_container_width=True)
                st.write(f"**{names[i]}**")