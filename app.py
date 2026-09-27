from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re

app = Flask(__name__)

# Load model and tokenizer once, at startup
model = load_model('ai_detector_model.h5')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    user_text = request.form['essay']

    cleaned = clean_text(user_text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=421, padding='post', truncating='post')

    prediction = model.predict(padded)[0][0]
    label = "AI-Generated" if prediction > 0.5 else "Human-Written"
    confidence = round(float(prediction if prediction > 0.5 else 1 - prediction) * 100, 2)

    return render_template('index.html', prediction=label, confidence=confidence)

@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)