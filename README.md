# Movie-Rating-Predictor
# 🎬 Movie Rating Predictor

A machine learning-powered Flask web app that predicts the **user rating** of a movie based on its **metadata** — such as genre, director, and top cast.

This project uses data from TMDB (The Movie Database) and applies natural language processing and regression techniques to estimate movie ratings.

---

## 📌 Features

- 🔍 Accepts movie metadata (genre, director, actors)
- 🧠 Uses a trained machine learning model for prediction
- 💡 Built using Flask, Scikit-learn, Pandas
- 🌐 Deployed on Render with a beautiful UI and background image

---

## 📊 Dataset Used

- [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- Files:
  - `tmdb_5000_movies.csv`
  - `tmdb_5000_credits.csv`

---

## 📁 Project Structure

movie-rating-predictor/
├── app.py # Flask backend
├── preprocess.py # Data cleaning & model training
├── requirements.txt
├── README.md
├── tmdb_5000_movies.csv # Dataset
├── tmdb_5000_credits.csv # Dataset
├── model/
│ ├── movie_model.pkl # Trained model
│ └── vectorizer.pkl # CountVectorizer
├── static/
│ ├── style.css
│ └── background.jpg
├── templates/
│ └── index.html

