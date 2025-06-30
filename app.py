from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open('model/movie_model.pkl', 'rb'))
cv = pickle.load(open('model/vectorizer.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    metadata = request.form['metadata']
    vector = cv.transform([metadata.lower()]).toarray()
    prediction = model.predict(vector)[0]
    return render_template('index.html', prediction_text=f"Predicted Movie Rating: {round(prediction, 2)} / 10")

if __name__ == "__main__":
    app.run(debug=True)
