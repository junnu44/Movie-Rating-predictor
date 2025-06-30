import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

# Load data
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

# Merge on title
movies = movies.merge(credits, on='title')

# Select relevant columns
movies = movies[['movie_id', 'title', 'genres', 'overview', 'cast', 'crew', 'vote_average']]

# Functions to extract names
def extract_names(text):
    return [i['name'] for i in ast.literal_eval(text)]

def extract_director(text):
    crew = ast.literal_eval(text)
    for i in crew:
        if i['job'] == 'Director':
            return i['name']
    return ''

# Apply
movies.dropna(inplace=True)
movies['genres'] = movies['genres'].apply(extract_names)
movies['cast'] = movies['cast'].apply(lambda x: extract_names(x)[:3])
movies['crew'] = movies['crew'].apply(extract_director)
movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies['tags'] = movies['overview'] + movies['genres'] + movies['cast'] + movies['crew'].apply(lambda x: [x])
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x).lower())

# Vectorization
cv = CountVectorizer(max_features=5000, stop_words='english')
X = cv.fit_transform(movies['tags']).toarray()
y = movies['vote_average']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model and vectorizer
pickle.dump(model, open("model/movie_model.pkl", "wb"))
pickle.dump(cv, open("model/vectorizer.pkl", "wb"))
