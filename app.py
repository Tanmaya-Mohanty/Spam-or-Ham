import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import os
import gdown

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()

vectorizer_id = st.secrets["VECTORIZER_ID"]
model_id = st.secrets["MODEL_ID"]

vectorizer_url = f"https://drive.google.com/uc?id={vectorizer_id}"
spam_classifier_url = f"https://drive.google.com/uc?id={model_id}"

# Download vectorizer and model
if not os.path.exists("vectorizer.pkl"):
    gdown.download(vectorizer_url, "vectorizer.pkl", quiet=False)

if not os.path.exists("spam_classifier.pkl"):
    gdown.download(spam_classifier_url, "spam_classifier.pkl", quiet=False)

# Load vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
mnb = pickle.load(open('spam_classifier.pkl', 'rb'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return ' '.join(y)

st.title('Spam or Ham')

input_text = st.text_area('Enter the SMS')

if st.button('Predict'):
    transformed_text = transform_text(input_text) # preprocess
    vector_input = tfidf.transform([transformed_text]) # vectorize
    pred = mnb.predict(vector_input)[0] # predict

    if pred == 0:
        st.header('Not spam')
    else:
        st.header('Spam')
