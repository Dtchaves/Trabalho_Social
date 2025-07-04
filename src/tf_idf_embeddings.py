import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class TfidfEmbedder:
    def __init__(self, max_features=1000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts).toarray()

    def transform(self, texts):
        return self.vectorizer.transform(texts).toarray()