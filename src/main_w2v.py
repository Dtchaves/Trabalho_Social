import os

import numpy as np

from utils import load_predator_ids
from preprocessing import load_conversations, prepare_data, prepare_predator_texts
from w2v_embeddings import train_word2vec, embed_texts
from analysis import plot_variance, train_classifiers, latent_space_analysis

import logging

logging.basicConfig(
    level=logging.INFO,  # Mostra mensagens de nível INFO ou superior
    format='%(message)s'  # Só mostra a mensagem (sem timestamp ou nível)
)

DATA_DIR = '../data/training'
XML_FILE = os.path.join(DATA_DIR, 'pan12-sexual-predator-identification-training-corpus-2012-05-01.xml')
PREDATOR_FILE = os.path.join(DATA_DIR, 'pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt')

DATA_DIR_TEST = '../data/test'
XML_FILE_TEST = os.path.join(DATA_DIR_TEST, 'pan12-sexual-predator-identification-test-corpus-2012-05-17.xml')
PREDATOR_FILE_TEST = os.path.join(DATA_DIR_TEST, 'pan12-sexual-predator-identification-groundtruth-problem1.txt')

predator_ids = load_predator_ids(PREDATOR_FILE)
conversations = load_conversations(XML_FILE, predator_ids)

predator_ids_test = load_predator_ids(PREDATOR_FILE_TEST)
conversations_test = load_conversations(XML_FILE_TEST, predator_ids_test)

# --- Conversa completa ---
texts, labels, ids = prepare_data(conversations)
logging.info(f"Total de conversas nos dados de treino: {len(texts)}")

texts_test, labels_test, ids_test = prepare_data(conversations_test)
logging.info(f"Total de conversas nos dados de teste: {len(texts_test)}")

# === TF-IDF Embeddings: Conversa completa ===
# === Embeddings Word2Vec: Conversa completa ===
print("Treinando Word2Vec para conversa completa do treino...")
w2v_model, tokenized_texts = train_word2vec(texts)
X_w2v = embed_texts(tokenized_texts, w2v_model)

print("Treinando Word2Vec para conversa completa do teste...")
w2v_model_test, tokenized_texts_test = train_word2vec(texts_test)
X_w2v_test = embed_texts(tokenized_texts_test, w2v_model_test)


logging.info("Treinando classificadores...")
train_classifiers(X_w2v, labels, X_w2v_test, labels_test, ids_test)


logging.info("Caracteristicas do espaço latente da base de treino...")
latent_space_analysis(X=X_w2v,y=labels, texts=texts,ids=ids, n_clusters=2, dir="../results/w2v/latent_space/training")

logging.info("Caracteristicas do espaço latente da base de teste...")
latent_space_analysis(X=X_w2v_test,y=labels_test, texts=texts_test,ids=ids_test, n_clusters=2, dir="../results/w2v/latent_space/test")

#logging.info("Caracteristicas do espaço latente da base completa (treino + teste)...")
#latent_space_analysis(
#    X=np.concatenate([X_w2v, X_w2v_test]),
#    y=np.concatenate([labels, labels_test]),
#    texts=texts + texts_test,
#    ids=ids + ids_test,
#    n_clusters=2,
#    dir="../results/BERT/latent_space/both"
#)



# === Paths ===
#DATA_DIR = '../training'
#XML_FILE = os.path.join(DATA_DIR, 'pan12-sexual-predator-identification-training-corpus-2012-05-01.xml')
#PREDATOR_FILE = os.path.join(DATA_DIR, 'pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt')

# === Load data ===
#predator_ids = load_predator_ids(PREDATOR_FILE)
#conversations = load_conversations(XML_FILE, predator_ids)

# --- Conversa completa ---
#texts, labels = prepare_data(conversations)
#print(f"Total de conversas: {len(texts)}")

# --- Só mensagens dos predadores ---
#texts_pred, labels_pred = prepare_predator_texts(conversations, predator_ids)
#print(f"Total de conversas com mensagem de predador: {len(texts_pred)}")

# === Embeddings Word2Vec: Conversa completa ===
#print("Treinando Word2Vec para conversa completa...")
#w2v_model, tokenized_texts = train_word2vec(texts)
#X_w2v = embed_texts(tokenized_texts, w2v_model)

# PCA e análise
#pca_w2v = plot_variance(X_w2v, n_components=30, title='PCA - Word2Vec (conversa completa)')
#X_w2v_pca = pca_w2v.transform(X_w2v)
#train_classifiers(X_w2v_pca, labels)

# Clusterização só das conversas de predadores
#indices_predadores = [i for i, label in enumerate(labels) if label == 1]
#X_pred = X_w2v_pca[indices_predadores]
#texts_pred_full = [texts[i] for i in indices_predadores]
#cluster_analysis(X_pred, texts=texts_pred_full, n_clusters=2)

# === Embeddings Word2Vec: Só mensagens do predador ===
#print("Treinando Word2Vec para mensagens só do predador...")
#w2v_pred, tokenized_pred = train_word2vec(texts_pred)
#X_w2v_pred = embed_texts(tokenized_pred, w2v_pred)

# PCA e análise
#pca_pred = plot_variance(X_w2v_pred, n_components=30, title='PCA - Word2Vec (só predador)')
#X_pred_pca = pca_pred.transform(X_w2v_pred)
#train_classifiers(X_pred_pca, labels_pred)

# Clusterização só dos textos de predadores
#cluster_analysis(X_pred_pca, texts=texts_pred, n_clusters=2)