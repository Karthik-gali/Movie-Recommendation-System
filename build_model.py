import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset (download from Kaggle: tmdb_5000_movies.csv)
movies = pd.read_csv("tmdb_5000_movies.csv")

# Select required columns
movies = movies[['title', 'overview']]
movies.dropna(inplace=True)

# Convert overview text into feature vectors
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['overview']).toarray()

# Compute similarity matrix
similarity = cosine_similarity(vectors)

# Save for Flask app
pickle.dump(movies, open('movies.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

print("✅ Model built successfully and saved as movies.pkl & similarity.pkl")
