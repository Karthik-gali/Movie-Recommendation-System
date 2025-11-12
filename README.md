#  Movie Recommendation System

A simple yet powerful **Movie Recommendation System** that suggests similar movies based on content similarity.  
Built with **Flask (Python)** on the backend and a **clean HTML + CSS** frontend interface.


## 🚀 Features
- 🎬 Search for any movie  
- 🤖 Get top 5 similar movies using cosine similarity  
- 💅 Clean and responsive HTML + CSS design  
- ⚙️ Simple Flask backend — lightweight and fast  

---

## 🧠 How It Works
1. The dataset (`tmdb_5000_movies.csv`) is processed using **CountVectorizer**.  
2. Cosine similarity measures the closeness between movie descriptions.  
3. Flask serves the top 5 similar movie titles to the frontend.  
4. The user sees the recommended results instantly.

## 🧰 Technologies Used

| Category | Technologies |
|-----------|--------------|
| **Frontend** | HTML5, CSS3 |
| **Backend** | Python (Flask Framework) |
| **Machine Learning** | scikit-learn (CountVectorizer, Cosine Similarity) |
| **Data Handling** | pandas, NumPy |
| **Model Serialization** | pickle |
| **Dataset** | TMDB 5000 Movie Dataset (from Kaggle) |
| **Tools / IDE** | VS Code, Jupyter Notebook |
| **Version Control** | Git & GitHub |



