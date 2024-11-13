import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
import ssl

# Create an unverified HTTPS context to bypass SSL verification issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


# Function to download NLTK resources if not already available
def download_nltk_resources():
    try:
        # Check if 'punkt' is downloaded
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        # Check if 'stopwords' is downloaded
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')


# Ensure required NLTK resources are downloaded
download_nltk_resources()

# Initialize PorterStemmer
ps = PorterStemmer()


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


# Load the model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit UI
st.title("Spam Shield SMS Sentinel")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1. Preprocess the input SMS
    transformed_sms = transform_text(input_sms)

    # 2. Vectorize the preprocessed SMS
    vector_input = tfidf.transform([transformed_sms])

    # 3. Predict if the message is spam or not
    result = model.predict(vector_input)[0]

    # 4. Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")