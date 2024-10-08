"""
The above Python code defines functions to analyze sentiment of text data using TextBlob, perform
sentiment analysis using a custom API, and display the analysis results on a web page.

:param request: The code you provided is a Django view function that performs sentiment analysis on
text data using various machine learning models and APIs. Here's a breakdown of the main
functionalities:
:return: The `analytic` function returns a rendered HTML template with various data passed as
context variables. The context variables include:
- "pos": Percentage of positive tweets
- "neu": Percentage of neutral tweets
- "neg": Percentage of negative tweets
- "acc": Accuracy score
- "result_sc": Result of sentiment analysis using SVM classifier
- "messages": Thai comments/messages
"""
from django.shortcuts import render
from django.http import JsonResponse
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn import tree
from google_trans_new import google_translator
from sklearn import metrics
import requests
import pyrebase
import tweepy
import pandas as pd
import numpy as np
import emoji
import csv
import re
import nltk
from nltk.corpus import stopwords

# The code snippet you provided is setting up the configuration for Firebase services in a Python
# Django application. Here's what each part of the code is doing:
firebaseConfig = {
    'apiKey': "AIzaSyCErkdR0G1y05dq5Ea2pavPbC-gTHeyssY",
    'authDomain': "webssru-87cc4.firebaseapp.com",
    'databaseURL': "https://webssru-87cc4.firebaseio.com",
    'projectId': "webssru-87cc4",
    'storageBucket': "webssru-87cc4.appspot.com",
    'messagingSenderId': "231310531528",
    'appId': "1:231310531528:web:4f48608234c255b70d3efd",
    'measurementId': "G-1SNRWYBJD0"
}

firebase = pyrebase.initialize_app(firebaseConfig)

db = firebase.database()

storage = firebase.storage()

def analytic(request):
    """
    The `analytic` function retrieves data from a database, performs sentiment analysis on the comments
    in Thai and English, and then renders the analysis results on an HTML template.
    
    :param request: The `analytic` function takes a `request` parameter, which is typically an HTTP
    request object that Django passes to the view function when a user accesses a web page. The function
    processes the request and returns a rendered HTML template with the analysis results and messages to
    be displayed on the webpage
    :return: The function `analytic(request)` is returning a rendered HTML template called
    'analytic.html' with the following context data:
    - "pos": pos
    - "neu": neu
    - "neg": neg
    - "acc": acc_sc
    - "result_sc": result_sc
    - "messages": messages_th
    - "score": sc
    - "result_ss": result_ss
    """
    messages_th = ""
    messages_en = ""
    acc = []
    result = []
    result_ssense = []
    score = []
    listP = []
    listSs = []
    requirement = db.child("Requirement").get()
    for require in requirement.each():
        message = require.val()
    messages_th = message["comments_th"]
    messages_en = message["comments_en"]
    textBlob_clf(messages_en, acc, result)
    ssense(messages_th, score, result_ssense, listSs)
    showAnalysis(listP)
    pos = listP[0]
    neu = listP[1]
    neg = listP[2]
    acc_sc = acc[0]
    sc = score[0]
    result_sc = result[0][0]
    result_ss = result_ssense[0]
    preneg = listSs[0]
    prepos = listSs[1]
    preseg = listSs[2]
    prekey = listSs[3]
    return render(request, 'analytic.html', {"pos": pos, "neu": neu, "neg": neg, "acc": acc_sc, "result_sc": result_sc,
                                            "messages": messages_th, "score": sc, 
                                            "result_ss":result_ss, "preneg":preneg, 
                                            "prepos":prepos, "preseg":preseg, "prekey":prekey})

def textBlob_clf(messages_en, acc, result):
    """
    This Python function performs sentiment analysis on Twitter data using TextBlob and a Support Vector
    Machine classifier, handling a MemoryError exception if necessary.
    
    :param messages_en: The `messages_en` parameter in the `textBlob_clf` function is used to pass a
    list of English text messages for sentiment analysis. These messages will be processed and
    classified as either positive or negative using a Support Vector Machine (SVM) classifier trained on
    a dataset of Twitter data
    :param acc: The `acc` parameter in the `textBlob_clf` function is a list that stores the accuracy
    scores of the model predictions. The function appends the accuracy score to this list after each
    iteration
    :param result: The `result` parameter in the `textBlob_clf` function seems to be a list that stores
    the results of sentiment analysis predictions for the input messages. The function appends the
    sentiment analysis prediction for a given message to this list. Each prediction is based on the
    input message and the trained SVM
    """
    # download Data
    download_csv()

    try:
        # Set Data
        df_Clean = pd.read_csv('twitterCrawler_clean.csv')
        df_Clean['Text'] = df_Clean['text']
        df_Clean['Polarity'] = df_Clean['Text'].apply(getPolarity)
        df_Clean['Analysis'] = df_Clean['Polarity'].apply(getAnalysis)
        dataSet = df_Clean[df_Clean.Polarity != 0]
        dataSet['Analysis'] = dataSet['Analysis'].replace('Positive', 1)
        dataSet['Analysis'] = dataSet['Analysis'].replace('Negative', 0)
        dataSet = dataSet.drop(['Unnamed: 0', 'text', 'Polarity'], axis=1)
        df_Sentiment = pd.read_csv('twitterCrawler_Sentiment_final.csv')
        df_Sentiment = df_Sentiment.drop(['Unnamed: 0'], axis=1)
        df_Sentiment = df_Sentiment.append(dataSet, ignore_index = True)

        # Analysis_Data
        stopset = set(stopwords.words('english'))
        vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset)
        y = df_Sentiment["Analysis"]
        X = vectorizer.fit_transform(df_Sentiment["Text"])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        svm_clf = SVC(probability=True, kernel='linear')
        svm_clf.fit(X_train, y_train)
        pred = svm_clf.predict(X_test)
        message_input = np.array([messages_en])
        message_vector = vectorizer.transform(message_input)
        result.append(svm_clf.predict(message_vector))
        acc_sc = "%.2f"%(metrics.accuracy_score(y_test, pred))
        acc.append(acc_sc)
    except MemoryError:
        # Set Data
        df_Sentiment = pd.read_csv('twitterCrawler_Sentiment_final.csv')
        df_Sentiment = df_Sentiment.drop(['Unnamed: 0'], axis=1)

        # Analysis_Data
        stopset = set(stopwords.words('english'))
        vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset)
        y = df_Sentiment["Analysis"]
        X = vectorizer.fit_transform(df_Sentiment["Text"])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        svm_clf = SVC(probability=True, kernel='linear')
        svm_clf.fit(X_train, y_train)
        pred = svm_clf.predict(X_test)
        message_input = np.array([messages_en])
        message_vector = vectorizer.transform(message_input)
        result.append(svm_clf.predict(message_vector))
        acc_sc = "%.2f"%(metrics.accuracy_score(y_test, pred))
        acc.append(acc_sc)

def ssense(messages, score, result_ssense, listSs):
    """
    This Python function sends a text message to the Ssense API to analyze sentiment and preprocess the
    text, storing the results in specified lists.
    
    :param messages: The `messages` parameter in the `ssense` function is used to pass the text data
    that you want to analyze for sentiment using the Ssense API. This text can be any message, review,
    comment, or other textual data that you want to evaluate for sentiment analysis
    :param score: The `score` parameter in the `ssense` function is a list that stores the sentiment
    scores of the messages processed by the API. The sentiment score indicates the sentiment intensity
    of the text, with positive values representing positive sentiment and negative values representing
    negative sentiment
    :param result_ssense: The `result_ssense` parameter in the `ssense` function is a list that stores
    the polarity of the sentiment analysis result for each message processed. The polarity indicates
    whether the sentiment of the text is positive, negative, or neutral. The function appends the
    polarity value for each message to this
    :param listSs: `listSs` is a list that stores various elements extracted from the response of the
    Ssense API. These elements include the negative sentiment, positive sentiment, segmented text, and
    keywords extracted from the input text provided in the `messages` parameter
    """
    url = "https://api.aiforthai.in.th/ssense"

    text = messages

    params = {'text': text}

    headers = {
        'Apikey': "ARVYukGnRlOej6pT7BIxKd993BVxaf37"
    }

    response = requests.get(url, headers=headers, params=params)

    response_dict = response.json()
    scores = response_dict['sentiment']['score']
    polarity = response_dict['sentiment']['polarity']
    listSs.append(response_dict['preprocess']['neg'])
    listSs.append(response_dict['preprocess']['pos'])
    listSs.append(response_dict['preprocess']['segmented'])
    listSs.append(response_dict['preprocess']['keyword'])
    score.append(scores)
    result_ssense.append(polarity)

def showAnalysis(listP):
    """
    The function `showAnalysis` reads and analyzes Twitter data to calculate the percentage of positive,
    neutral, and negative tweets.
    
    :param listP: The `listP` parameter is a list that is used to store the percentage of positive,
    neutral, and negative tweets in the analysis. The function `showAnalysis` calculates these
    percentages and appends them to the `listP` list
    """
    # Set Data
    df_Clean = pd.read_csv('twitterCrawler_clean.csv')
    df_Clean['Text'] = df_Clean['text']
    df_Clean['Polarity'] = df_Clean['Text'].apply(getPolarity)
    df_Clean['Analysis'] = df_Clean['Polarity'].apply(getAnalysis)
    dataSet = df_Clean[df_Clean.Polarity != 0]
    dataSet['Analysis'] = dataSet['Analysis'].replace('Positive', 1)
    dataSet['Analysis'] = dataSet['Analysis'].replace('Negative', 0)
    dataSet = dataSet.drop(['Unnamed: 0', 'text', 'Polarity'], axis=1)
    anlysis_df = pd.read_csv('twitterCrawler_Sentiment_final.csv')
    anlysis_df = anlysis_df.drop(['Unnamed: 0'], axis=1)
    anlysis_df = anlysis_df.append(dataSet, ignore_index = True)
    anlysis_df['Polarity'] = anlysis_df['Text'].apply(getPolarity)
    anlysis_df['Analysis'] = anlysis_df['Polarity'].apply(getAnalysis)
    # Get the percentage of positive tweets
    postweets = anlysis_df[anlysis_df.Analysis == 'Positive']
    postweets = postweets['Text']
    posPercen = round((postweets.shape[0] / anlysis_df.shape[0]) * 100, 1)
    listP.append(posPercen)

    # Get the percentage of neutral tweets
    neutweets = anlysis_df[anlysis_df.Analysis == 'Neutral']
    neutweets = neutweets['Text']
    neuPercen = round((neutweets.shape[0] / anlysis_df.shape[0]) * 100, 1)
    listP.append(neuPercen)

    # Get the percentage of negative tweets
    negtweets = anlysis_df[anlysis_df.Analysis == 'Negative']
    negtweets = negtweets['Text']
    negPercen = round((negtweets.shape[0] / anlysis_df.shape[0]) * 100, 1)
    listP.append(negPercen)
    # return render(request, 'analytic.html', {"listP": listP})

def download_csv():
    """
    The code includes functions to download a CSV file, calculate the polarity of text using TextBlob,
    and analyze the sentiment as negative, neutral, or positive based on the polarity score.
    """
    filename_new = "twitterCrawler_Sentiment_final.csv"
    cloud = "twitterCrawler_Sentiment_final.csv"
    storage.child(cloud).download("", filename_new)

# Create a function th get the polarity
    """
    The function `getPolarity(text)` uses TextBlob to analyze the sentiment of the input text and
    returns the polarity score.
    
    :param text: The function you provided is almost correct, but it seems like you have a typo in the
    function definition. Here is the corrected version:
    :return: The function `getPolarity(text)` returns the polarity of the input text using TextBlob's
    sentiment analysis.
    """
def getPolarity(text):
    return TextBlob(text).sentiment.polarity

# Create a function to computer the negative, neutral and positive analysis
def getAnalysis(score):
    """
    The function `getAnalysis` categorizes a given score as 'Negative', 'Neutral', or 'Positive' based
    on its value.
    
    :param score: The `getAnalysis` function takes a parameter `score` as input. This function analyzes
    the score and returns a corresponding analysis result - 'Negative' if the score is less than 0,
    'Neutral' if the score is equal to 0, and 'Positive' if the score is greater
    :return: The function `getAnalysis` returns a string indicating whether the input `score` is
    'Negative' if it is less than 0, 'Neutral' if it is equal to 0, or 'Positive' if it is greater than
    0.
    """
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'