from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from keras.models import load_model

from feature import *

app = Flask(__name__)

model = load_model('fakenewspred.h5')


@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():

    news_title = request.form['news_title']
    author_name = request.form['author_name']
    text = request.form['text']
    total = get_all_query(news_title, author_name, text)
    query = remove_punctuation_stopwords_lemma(total)
    user_input = {'query': query}
    pred = model.predict(query)
    pred = pred[:, 0]
    dic = {1: 'real', 0: 'fake'}
    return f'<html><body><h1>{dic[pred[0]]}</h1> <form action="/"> <button type="submit">back </button> </form></body></html>'


if __name__ == "__main__":
    app.run(port=8080, debug=True)
