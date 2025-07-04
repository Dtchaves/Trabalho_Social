import os

os.environ['TRANSFORMERS_CACHE'] = ''

import numpy as np
from utils import load_predator_ids
from preprocessing import load_conversations, prepare_data, prepare_predator_texts
from bert_embeddings import BertEmbedder
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

# --- Só mensagens dos predadores ---
#texts_pred, labels_pred = prepare_predator_texts(conversations, predator_ids)
#logging.info(f"Total de conversas com mensagem de predador: {len(texts_pred)}")

# === BERT Embeddings: Conversa completa ===
logging.info("Extraindo embeddings BERT para conversa completa nos dados de treino...")
bert = BertEmbedder()
X_bert = bert.get_embeddings(texts)

logging.info("Extraindo embeddings BERT para conversa completa nos dados de teste...")
X_bert_test = bert.get_embeddings(texts_test)

#plot_variance(X_bert, n_components=30, title='PCA - BERT (conversa completa)')

#from sklearn.decomposition import PCA
#pca_bert = PCA(n_components=30)
#X_bert_pca = pca_bert.fit_transform(X_bert)
logging.info("Treinando classificadores...")
#train_classifiers(X_bert, labels, X_bert_test, labels_test, ids_test)

#indices_predadores = [i for i, label in enumerate(labels) if label == 1]
#X_pred_bert = X_bert[indices_predadores]
#texts_pred_full = [texts[i] for i in indices_predadores]
logging.info("Caracteristicas do espaço latente da base de treino...")
latent_space_analysis(X=X_bert,y=labels, texts=texts,ids=ids, n_clusters=2, dir="../results/BERT/latent_space/training")

logging.info("Caracteristicas do espaço latente da base de teste...")
latent_space_analysis(X=X_bert_test,y=labels_test, texts=texts_test,ids=ids_test, n_clusters=2, dir="../results/BERT/latent_space/test")

logging.info("Caracteristicas do espaço latente da base completa (treino + teste)...")
#latent_space_analysis(
#    X=np.concatenate([X_bert, X_bert_test]),
#    y=np.concatenate([labels, labels_test]),
#    texts=texts + texts_test,
#    ids=ids + ids_test,
#    n_clusters=2,
#    dir="../results/BERT/latent_space/both"
#)

# === BERT Embeddings: Só mensagens dos predadores ===
#logging.info("Extraindo embeddings BERT só dos predadores...")
#X_bert_pred = bert.get_embeddings(texts_pred)
#pca_bert_pred = PCA(n_components=30)
#X_bert_pred_pca = pca_bert_pred.fit_transform(X_bert_pred)
#train_classifiers(X_bert_pred_pca, labels_pred)
#cluster_analysis(X_bert_pred_pca, texts=texts_pred, n_clusters=2)