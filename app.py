from flask import Flask, render_template, request,redirect
from helper import preprocess_text
from logger import logging

app = Flask(__name__)

logging.info('Starting the app...')

data = dict()

type = '' 
NBAcuuracy = 0
LRAccuracy = 0

@app.route('/')
def index():
    data['reviews'] = reviews
    data['positive'] = positive
    data['negative'] = negative

    logging.info('.................Open Home Page.................')
    return render_template('index.html', data=data)

@app.route("/",methods=['POST'])
def get_data():
    text = request.form['text']
    logging.info(f'Text: {text}')
    preprocessed_text = preprocess_text(text)
    logging.info(f'Preprocessed Text: {preprocessed_text}')
    
    return redirect(request.url)
    


if(__name__ == '__main__'):
    app.run()