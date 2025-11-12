from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load data and similarity matrix
movies = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Function to recommend similar movies
def recommend(movie):
    movie = movie.strip().title()
    if movie not in movies['title'].values:
        return []
    idx = movies[movies['title'] == movie].index[0]
    distances = similarity[idx]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return [movies.iloc[i[0]].title for i in movie_list]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    movie_name = request.form['movie']
    recommendations = recommend(movie_name)
    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
    app.run(debug=True)
