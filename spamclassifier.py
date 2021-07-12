import pandas as pd
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

messages = pd.read_csv("smsspamcollection/SMSSpamCollection", sep='\t', names = ["label","messages"])

# Data Cleaning
sb = SnowballStemmer('english')
corpus = []
for i in range(0,len(messages)):
    message = re.sub('[^a-zA-Z]]', ' ', messages['messages'][i])
    message = message.lower()
    message = message.split()

    message = [sb.stem(word) for word in message if not word in stopwords.words('english')]
    message = ' '.join(message)
    corpus.append(message)

# Creating bag of words
cv = CountVectorizer(max_features=5000)
x = cv.fit_transform(corpus)
y = pd.get_dummies(messages['label'])
y = y.iloc[:,1]

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.20, random_state=255)

spam_detect = MultinomialNB()
spam_detect.fit(X_train, y_train)

st.title("Spam Detection System")

def detect():
    inp = st.text_area("Enter Email to detect as Spam or Not: ")
    if len(inp) < 1:
        st.write("Enter something: ")
    else:
        msg = inp
        msg = cv.transform([msg]).toarray()
        pred = spam_detect.predict(msg)
        if pred == 1:
            st.title("Spam")
        else:
            st.title("Not Spam")

detect()













