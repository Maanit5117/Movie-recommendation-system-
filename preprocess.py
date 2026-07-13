import pandas as pd
import numpy as np
import ast
import pickle
import os
from nltk.stem.porter import PorterStemmer

print("Starting preprocessing...")

# Paths
movies_path = '/Users/maanitmittra/Downloads/archive/tmdb_5000_movies.csv'
credits_path = '/Users/maanitmittra/Downloads/archive/tmdb_5000_credits.csv'

# Check files exist
if not os.path.exists(movies_path) or not os.path.exists(credits_path):
    raise FileNotFoundError("Raw CSV datasets not found in Downloads folder.")

movies = pd.read_csv(movies_path)
credits = pd.read_csv(credits_path)

print(f"Loaded movies shape: {movies.shape}")
print(f"Loaded credits shape: {credits.shape}")

# Merge datasets
movies = movies.merge(credits, on='title')
print(f"Merged movies shape: {movies.shape}")

# Helper functions to convert stringified lists of dicts
def convert_genres_keywords(obj):
    if pd.isna(obj):
        return []
    L = []
    try:
        for i in ast.literal_eval(obj):
            L.append(i['name'])
    except Exception as e:
        pass
    return L

def convert_cast(obj):
    if pd.isna(obj):
        return []
    L = []
    try:
        counter = 0
        for i in ast.literal_eval(obj):
            if counter != 3:
                L.append(i['name'])
                counter += 1
            else:
                break
    except Exception as e:
        pass
    return L

def fetch_director(obj):
    if pd.isna(obj):
        return []
    L = []
    try:
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                L.append(i['name'])
                break
    except Exception as e:
        pass
    return L

# Apply conversions
movies['genres_list'] = movies['genres'].apply(convert_genres_keywords)
movies['keywords_list'] = movies['keywords'].apply(convert_genres_keywords)
movies['cast_list'] = movies['cast'].apply(convert_cast)
movies['director_list'] = movies['crew'].apply(fetch_director)

# Preprocess overview: handle NaN and convert to list of words
movies['overview'] = movies['overview'].fillna('')
movies['overview_list'] = movies['overview'].apply(lambda x: x.split())

# Clean spaces: remove space to merge tokens (e.g. "Johnny Depp" -> "JohnnyDepp")
movies['genres_clean'] = movies['genres_list'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords_clean'] = movies['keywords_list'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast_clean'] = movies['cast_list'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['director_clean'] = movies['director_list'].apply(lambda x: [i.replace(" ", "") for i in x])

# Create combined tags
movies['tags_list'] = movies['overview_list'] + movies['genres_clean'] + movies['keywords_clean'] + movies['cast_clean'] + movies['director_clean']
movies['tags'] = movies['tags_list'].apply(lambda x: " ".join(x).lower())

# Apply stemming
ps = PorterStemmer()
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

movies['tags'] = movies['tags'].apply(stem)

# Keep only necessary columns for the application:
# id, title, tags, genres_list, overview, tagline, release_date, vote_average, cast_list, director_list
# We will use these in the Streamlit UI to show movie details.
output_cols = [
    'id', 
    'title', 
    'tags', 
    'genres_list', 
    'overview', 
    'tagline', 
    'release_date', 
    'vote_average', 
    'cast_list', 
    'director_list'
]
processed_movies = movies[output_cols].copy()

# Print some statistics
print(f"Preprocessed data shape: {processed_movies.shape}")
print("Columns in processed data:", processed_movies.columns.tolist())

# Save to a pickle file
output_pickle_path = '/Users/maanitmittra/Documents/Movie Recommender system/movies.pkl'
with open(output_pickle_path, 'wb') as f:
    pickle.dump(processed_movies, f)

print(f"Successfully saved preprocessed movies to {output_pickle_path}")
