import numpy as np
import pandas as pd
import string
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer,PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

#load model
with open('static/logistic_model.pickle','rb') as file:
    logistic_model = pickle.load(file)

with open('static/naivebayes_model.pickle','rb') as file:
    naivebayes_model = pickle.load(file)

#load vocabulary
vocab = pd.read_csv('static/vocabulary.txt',header=None)
tokens = vocab[0].tolist()

#preprocess
def preprocess_text(text):
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [word for word in tokens if word.lower() not in set(stopwords.words('english'))]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    return ' '.join(tokens)


#Vectorization
def vectorizer(ds):
    vectorized_list=[]
    for sentence in ds:
        sentence_list = np.zeros(len(tokens))
        for i in range(len(tokens)):
            if tokens[i] in sentence.split():
                sentence_list[i] = 1
        vectorized_list.append(sentence_list)
    vectorized_list_new = np.asarray(vectorized_list,dtype='float32')
    return vectorized_list_new

#predict
def get_prediction_from_NB(vectorized_input):
    prediction = naivebayes_model.predict(vectorized_input)
    return prediction
    
def get_prediction_from_LR(vectorized_input):
    prediction = logistic_model.predict(vectorized_input)
    return prediction       

