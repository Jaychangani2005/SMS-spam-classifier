import streamlit as st
import pickle
import nltk
from nltk.stem import  PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
import string


def ensure_nltk_data():
    # Download required tokenization/corpus resources if unavailable.
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]
    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name, quiet=True)


ensure_nltk_data()
STOP_WORDS = set(stopwords.words('english'))

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    try:
        text = nltk.word_tokenize(text)
    except LookupError:
        # Newer NLTK releases may require punkt_tab at runtime.
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        try:
            text = nltk.word_tokenize(text)
        except LookupError:
            # Keep app functional if tokenizer data cannot be downloaded.
            text = wordpunct_tokenize(text)
    y= []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in STOP_WORDS and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


# st.title("Email/SMS Classifier")
st.title(':red[Email/SMS] Classifier')
input_sms = st.text_area("Enter the massage")


if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    # vectorize
    vector_input = tfidf.transform([transformed_sms])
    # predict
    result = model.predict(vector_input)[0]
    # display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
