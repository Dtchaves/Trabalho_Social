import os

os.environ['TRANSFORMERS_CACHE'] = '/scratch/diogochaves/hf_cache'
os.environ['HF_HOME'] = '/scratch/diogochaves/hf_cache'

import numpy as np
from utils import load_predator_ids
from preprocessing import load_conversations, prepare_data, prepare_predator_texts
from analysis import plot_variance, train_classifiers, latent_space_analysis
import logging

logging.basicConfig(
    level=logging.INFO,  # Mostra mensagens de nível INFO ou superior
    format='%(message)s'  # Só mostra a mensagem (sem timestamp ou nível)
)


from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt


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


logging.info("Extraindo embeddings LM para conversa completa nos dados de treino...")
model = SentenceTransformer('all-MiniLM-L6-v2')
X_emb = model.encode(texts, show_progress_bar=True)

logging.info("Extraindo embeddings BERT para conversa completa nos dados de teste...")
X_emb_test = model.encode(texts_test, show_progress_bar=True)

logging.info("Treinando classificadores...")
#train_classifiers(X_emb, labels, X_emb_test, labels_test, ids_test)


logging.info("Caracteristicas do espaço latente da base de treino...")
latent_space_analysis(X=X_emb,y=labels, texts=texts,ids=ids, n_clusters=2, dir="../results/LM/latent_space/training")

logging.info("Caracteristicas do espaço latente da base de teste...")
latent_space_analysis(X=X_emb_test,y=labels_test, texts=texts_test,ids=ids_test, n_clusters=2, dir="../results/LM/latent_space/test")

logging.info("Caracteristicas do espaço latente da base completa (treino + teste)...")
# latent_space_analysis(
#     X=np.concatenate([X_emb, X_emb_test]),
#     y=np.concatenate([labels, labels_test]),
#     texts=texts + texts_test,
#     ids=ids + ids_test,
#     n_clusters=2,
#     dir="../results/BERT/latent_space/both"
# )
