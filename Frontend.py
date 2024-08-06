import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

nltk.download('punkt')
nltk.download('stopwords')

def transform_text(text):

    text = text.lower()

    tokens = nltk.word_tokenize(text)

    tokens = [token for token in tokens if token.isalnum()]

    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]


    ps = PorterStemmer()
    tokens = [ps.stem(token) for token in tokens]
    return " ".join(tokens)


tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title("Email/SMS classifier")
input_sms=st.text_area("Enter the message")
if st.button('Predict'):

    transform_sms=transform_text(input_sms)
    vector_input=tfidf.transform([transform_sms])
    result=model.predict(vector_input)[0]

    if result==1:
        st.header('SPAM')
    else:
        st.header('NOT SPAM')

