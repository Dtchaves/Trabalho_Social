import os

import numpy as np


import logging

logging.basicConfig(
    level=logging.INFO,  # Mostra mensagens de nível INFO ou superior
    format='%(message)s'  # Só mostra a mensagem (sem timestamp ou nível)
)

from utils import load_predator_ids
from preprocessing import load_conversations, prepare_data, prepare_predator_texts
from tf_idf_embeddings import TfidfEmbedder
from analysis import plot_variance, train_classifiers, latent_space_analysis
from sklearn.decomposition import PCA

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
print("Extraindo embeddings TF-IDF para conversa completa no treino...")
tfidf = TfidfEmbedder(max_features=1000)
X_tfidf = tfidf.fit_transform(texts)

print("Extraindo embeddings TF-IDF para conversa completa no teste...")
X_tfidf_test = tfidf.fit_transform(texts_test)
#plot_variance(X_tfidf, n_components=30, title='PCA - TF-IDF (conversa completa)')
#pca_tfidf = PCA(n_components=30)
#X_tfidf_pca = pca_tfidf.fit_transform(X_tfidf)
#train_classifiers(X_tfidf_pca, labels)

#indices_predadores = [i for i, label in enumerate(labels) if label == 1]
#X_pred_tfidf = X_tfidf_pca[indices_predadores]
#texts_pred_full = [texts[i] for i in indices_predadores]
#cluster_analysis(X_pred_tfidf, texts=texts_pred_full, n_clusters=2)

# === TF-IDF Embeddings: Só mensagens dos predadores ===
#print("Extraindo embeddings TF-IDF só dos predadores...")
#X_tfidf_pred = tfidf.fit_transform(texts_pred)
#pca_tfidf_pred = PCA(n_components=30)
#X_tfidf_pred_pca = pca_tfidf_pred.fit_transform(X_tfidf_pred)
#train_classifiers(X_tfidf_pred_pca, labels_pred)
#cluster_analysis(X_tfidf_pred_pca, texts=texts_pred, n_clusters=2)


logging.info("Treinando classificadores...")
train_classifiers(X_tfidf, labels, X_tfidf_test, labels_test, ids_test)


logging.info("Caracteristicas do espaço latente da base de treino...")
latent_space_analysis(X=X_tfidf,y=labels, texts=texts,ids=ids, n_clusters=2, dir="../results/TF_IDF/latent_space/training")

logging.info("Caracteristicas do espaço latente da base de teste...")
latent_space_analysis(X=X_tfidf_test,y=labels_test, texts=texts_test,ids=ids_test, n_clusters=2, dir="../results/TF_IDF/latent_space/test")

#logging.info("Caracteristicas do espaço latente da base completa (treino + teste)...")
#latent_space_analysis(
#    X=np.concatenate([X_tfidf, X_tfidf_test]),
#    y=np.concatenate([labels, labels_test]),
#    texts=texts + texts_test,
#    ids=ids + ids_test,
#    n_clusters=2,
#    dir="../results/BERT/latent_space/both"
#)