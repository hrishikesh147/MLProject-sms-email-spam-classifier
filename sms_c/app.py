import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()

tfidf= pickle.load(open('vectorizer1.pkl','rb'))
model= pickle.load(open('model1.pkl','rb'))


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

    return " ".join(y)

st.title("SMS/Email Classifier")

input_sms=st.text_area("enter the sms or email")

if st.button('predict'):

    #steps
    #preprocess
    transformed_sms=transform_text(input_sms)
    #vectorise
    vector_input=tfidf.transform([transformed_sms])
    #predict
    result=model.predict(vector_input)[0]
    #Display
    if result==1:
        st.header("Spam")
    else:
        st.header("not Spam")




