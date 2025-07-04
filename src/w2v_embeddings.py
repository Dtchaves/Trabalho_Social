import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

def train_word2vec(texts, vector_size=100, min_count=2, window=5, sg=1):
    tokenized_texts = [simple_preprocess(text) for text in texts]
    w2v_model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        sg=sg
    )
    return w2v_model, tokenized_texts

def get_sentence_embedding(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

def embed_texts(tokenized_texts, w2v_model):
    return np.array([get_sentence_embedding(tokens, w2v_model) for tokens in tokenized_texts])