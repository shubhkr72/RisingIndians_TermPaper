import re
import string
from typing import Optional

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

STOP = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def combine_title_text(title: Optional[str], text: Optional[str]) -> str:
    title = "" if title is None else str(title)
    text = "" if text is None else str(text)
    return f"{title.strip()} {text.strip()}".strip()


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = [
        LEMMATIZER.lemmatize(token)
        for token in text.split()
        if token not in STOP and len(token) > 2
    ]
    return " ".join(tokens)