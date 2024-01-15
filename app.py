from flask import Flask, render_template, request,redirect
from helper import preprocess_text,vectorizer,get_prediction_from_NB,get_prediction_from_LR
from logger import logging

app = Flask(__name__)

logging.info('Starting the app...')

data = dict()

 
NBOutput = ''
LROutput = ''

@app.route('/')
def index():
    logging.info('.................Open Home Page.................')
    data['NBOutput'] = NBOutput
    data['LROutput'] = LROutput
    return render_template('index.html', data=data)

@app.route("/",methods=['POST'])
def get_data():
    text = request.form['text']
    logging.info(f'Text: {text}')
    preprocessed_text = preprocess_text(text)
    logging.info(f'Preprocessed Text: {preprocessed_text}')
    vectorized_input = vectorizer([preprocessed_text])
    logging.info(f'Vectorized Input: {vectorized_input}')
    predictionNB = get_prediction_from_NB(vectorized_input)
    if(predictionNB[0]==1):
        global NBOutput
        NBOutput = 'Not Spam'
    else:
        NBOutput = 'Spam'
    logging.info(f'NB Prediction: {NB_prediction}')
    predictionLR = get_prediction_from_LR(vectorized_input)
    if(predictionLR[0]==1):
        global LROutput
        LROutput = 'Not Spam'
    else:
        LROutput = 'Spam'
    logging.info(f'LR Prediction: {LR_prediction}')
    return redirect(request.url)
    


if(__name__ == '__main__'):
    app.run()