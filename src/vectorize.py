from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer():
    return TfidfVectorizer(stop_words="english")


def fit_transform(vectorizer, corpus):
    return vectorizer.fit_transform(corpus)
