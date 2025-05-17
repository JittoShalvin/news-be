from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import re
import nltk
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import requests
from bs4 import BeautifulSoup

nltk.download('punkt')

app = Flask(__name__)
CORS(app)

# Load tokenizer and model
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

model = tf.keras.models.load_model('fake_news_model.h5')

MAX_TITLE_LEN = 20
MAX_TEXT_LEN = 300

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', str(text), flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    return text.lower()

def web_scrape_find_source(content):
    """
    Search Google (or Bing) with a snippet of the content and return the first found news source URL.
    NOTE: Google scraping is complex due to bot detection, so here is a simple Bing search example.
    """
    search_query = '+'.join(content.split()[:10])  # first 10 words
    search_url = f"https://www.bing.com/news/search?q={search_query}"

    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(search_url, headers=headers, timeout=5)
        soup = BeautifulSoup(resp.text, 'html.parser')
        # Bing news search results have <a> tags with 'title' class inside
        results = soup.find_all('a', attrs={'href': True})
        for link in results:
            href = link['href']
            if href.startswith('http'):
                # Simple heuristic: return first link that looks like news article
                if any(domain in href for domain in ['.com', '.in', '.org', '.news', '.co']):
                    return href
    except Exception as e:
        print("Web scraping error:", e)

    return "Source not found"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    title = clean_text(data.get('title', ''))
    content = clean_text(data.get('content', ''))

    title_seq = tokenizer.texts_to_sequences([title])
    text_seq = tokenizer.texts_to_sequences([content])

    title_pad = pad_sequences(title_seq, maxlen=MAX_TITLE_LEN)
    text_pad = pad_sequences(text_seq, maxlen=MAX_TEXT_LEN)

    prediction = model.predict([title_pad, text_pad])[0][0]
    label = "Real" if prediction > 0.5 else "Fake"

    return jsonify({
        'prediction': label,
        'confidence': float(prediction)
    })

@app.route('/predict_from_content', methods=['POST'])
def predict_from_content():
    data = request.json
    content = clean_text(data.get('content', ''))
    content_seq = tokenizer.texts_to_sequences([content])
    content_pad = pad_sequences(content_seq, maxlen=MAX_TEXT_LEN)

    # Empty title padding as model expects 2 inputs
    empty_title_pad = np.zeros((1, MAX_TITLE_LEN), dtype=int)

    prediction = model.predict([empty_title_pad, content_pad])[0][0]
    label = "Real" if prediction > 0.5 else "Fake"

    # Find source URL by scraping
    source_url = web_scrape_find_source(content)

    return jsonify({
        'prediction': label,
        'confidence': float(prediction),
        'source': source_url
    })

if __name__ == '__main__':
    app.run(debug=True)
