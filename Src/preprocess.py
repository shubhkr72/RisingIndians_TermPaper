# src/preprocess.py

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download once (safe if already installed)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

STOP = set(stopwords.words('english'))
lem = WordNetLemmatizer()


def clean_text(text: str) -> str:
    text = str(text).lower()

    # remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # remove numbers
    text = re.sub(r'\d+', '', text)

    # tokenize + lemmatize + remove stopwords
    tokens = [
        lem.lemmatize(word)
        for word in text.split()
        if word not in STOP and len(word) > 2
    ]

    return " ".join(tokens)