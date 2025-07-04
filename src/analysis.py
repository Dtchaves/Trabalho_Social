import numpy as np
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,  # Mostra mensagens de nível INFO ou superior
    format='%(message)s'  # Só mostra a mensagem (sem timestamp ou nível)
)

import torch

import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F

import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score, adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score,completeness_score 
from sklearn.cluster import KMeans
from collections import Counter

def plot_variance(X, n_components=30, title='PCA'):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    var_exp = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(var_exp)+1), var_exp, marker='o', color='b')
    plt.xlabel('Número de componentes principais')
    plt.ylabel('Variância explicada acumulada')
    plt.title(title)
    plt.grid(True)
    plt.show()
    return pca

class DeepMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)


# Residual MLP com blocos intermediários
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += identity
        return F.relu(out)


class DeepResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_blocks=3):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(num_blocks)])

        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.blocks(x)
        return self.output_layer(x)

def train_torch_model(model, X_train, y_train, X_test, epochs=50, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        output = model(X_test_tensor)
        probs = torch.sigmoid(output).cpu().numpy()
        preds = (probs > 0.5).astype(int).flatten()
    return preds

def train_classifiers(X_train, y_train, X_test, y_test, ids_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_preds = {}
    y_test = np.array(y_test)
    
    logging.info("--- Logistic Regression ---")
    clf_lr = LogisticRegression(max_iter=300, class_weight='balanced')
    clf_lr.fit(X_train, y_train)
    y_pred_lr = clf_lr.predict(X_test)
    logging.info(classification_report(y_test, y_pred_lr, digits=4))
    logging.info(confusion_matrix(y_test, y_pred_lr))
    all_preds['lr'] = y_pred_lr

    logging.info("--- SVM ---")
    clf_svm = SVC(class_weight='balanced')
    clf_svm.fit(X_train, y_train)
    y_pred_svm = clf_svm.predict(X_test)
    logging.info(classification_report(y_test, y_pred_svm, digits=4))
    logging.info(confusion_matrix(y_test, y_pred_svm))
    all_preds['svm'] = y_pred_svm

    logging.info("--- Random Forest ---")
    clf_rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    clf_rf.fit(X_train, y_train)
    y_pred_rf = clf_rf.predict(X_test)
    logging.info(classification_report(y_test, y_pred_rf, digits=4))
    logging.info(confusion_matrix(y_test, y_pred_rf))
    all_preds['rf'] = y_pred_rf

    logging.info("--- MLP Classifier ---")
    clf_mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    clf_mlp.fit(X_train, y_train)
    y_pred_mlp = clf_mlp.predict(X_test)
    logging.info(classification_report(y_test, y_pred_mlp, digits=4))
    logging.info(confusion_matrix(y_test, y_pred_mlp))
    
    logging.info("--- Deep Residual MLP ---")
    model = DeepResidualMLP(X_train.shape[1]).to(device)
    y_pred_resmlp = train_torch_model(model, X_train, y_train, X_test)
    logging.info(classification_report(y_test, y_pred_resmlp, digits=4))
    logging.info(confusion_matrix(y_test, y_pred_resmlp))
    all_preds['resmlp'] = y_pred_resmlp

    logging.info("--- Deep MLP ---")
    model = DeepMLP(X_train.shape[1]).to(device)
    y_pred_deepmlp = train_torch_model(model, X_train, y_train, X_test)
    logging.info(classification_report(y_test, y_pred_deepmlp, digits=4))
    logging.info(confusion_matrix(y_test, y_pred_deepmlp))
    all_preds['deepmlp'] = y_pred_deepmlp
    
     # === Examinar erros em comum entre TODOS os classificadores ===
    logging.info("=== Erros em comum entre todos os classificadores ===")
    common_errors = set(np.where(y_test != all_preds['lr'])[0])
    for name, preds in all_preds.items():
        common_errors &= set(np.where(y_test != preds)[0])

    if ids_test is not None:
        wrong_ids = [ids_test[i] for i in sorted(common_errors)]
        logging.info(f"IDs com erro comum entre todos os classificadores: {wrong_ids}")
    else:
        logging.info(f"Índices com erro comum entre todos os classificadores: {sorted(common_errors)}")

def latent_space_analysis(X, y, texts=None, ids=None, n_clusters=2, dir="figures"):
    os.makedirs(dir, exist_ok=True)

    # # === KMeans Clustering ===
    # logging.info("=== Clustering com KMeans ===")
    # kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    # clusters = kmeans.fit_predict(X)

    # sil_score = silhouette_score(X, clusters)
    # logging.info(f"Silhouette Score: {sil_score:.3f}")
    # logging.info(f"Tamanho dos clusters: {Counter(clusters)}")

    # # === External Metrics ===
    # logging.info("=== Métricas externas ===")
    # logging.info(f"Adjusted Rand Index (ARI): {adjusted_rand_score(y, clusters):.4f}")
    # logging.info(f"Adjusted Mutual Info (AMI): {adjusted_mutual_info_score(y, clusters):.4f}")
    # logging.info(f"Homogeneity: {homogeneity_score(y, clusters):.4f}")
    # logging.info(f"Completeness: {completeness_score(y, clusters):.4f}")

    # === Dimensionality Reductions ===
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    umap = UMAP(n_components=2, random_state=42)
    X_umap = umap.fit_transform(X)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_tsne = tsne.fit_transform(X)

    # === Função auxiliar para visualizações com legenda binária ===
    def plot_binary(X_2d, labels, title, filename, label_names=['Não predador', 'Predador']):
        labels = np.array(labels).astype(int)
        plt.figure(figsize=(8, 6))
        for value, color, name in zip([0, 1], ['skyblue', 'red'], label_names):
            idx = (labels == value)
            plt.scatter(X_2d[idx, 0], X_2d[idx, 1], c=color, label=name, alpha=0.7, edgecolors='k', s=60)
        plt.title(title)
        plt.xlabel(f'{title.split("(")[0]} 1')
        plt.ylabel(f'{title.split("(")[0]} 2')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(dir, filename))
        plt.close()

    # # === Visualização dos clusters preditos ===
    # plot_binary(X_pca, clusters, f'Clusters via PCA (K={n_clusters})', "clusters_pca.png", [f"Cluster 0", f"Cluster 1"])
    # plot_binary(X_umap, clusters, f'Clusters via UMAP (K={n_clusters})', "clusters_umap.png", [f"Cluster 0", f"Cluster 1"])
    # plot_binary(X_tsne, clusters, f'Clusters via t-SNE (K={n_clusters})', "clusters_tsne.png", [f"Cluster 0", f"Cluster 1"])

    # === Visualização das labels reais ===
    plot_binary(X_pca, y, 'Distribuição das Labels Reais (PCA)', "true_labels_pca.png")
    plot_binary(X_umap, y, 'Distribuição das Labels Reais (UMAP)', "true_labels_umap.png")
    plot_binary(X_tsne, y, 'Distribuição das Labels Reais (t-SNE)', "true_labels_tsne.png")

    # === Estatísticas por cluster e por label ===
    df = pd.DataFrame(X)
    #df['cluster'] = clusters
    df['label'] = y

    logging.info("=== Estatísticas por Cluster ===")
    logging.info(df.groupby("cluster").describe())

    logging.info("=== Estatísticas por Label ===")
    logging.info(df.groupby("label").describe())

    # === IDs das conversas corretamente ou erroneamente classificadas (assumindo label 1 = predador) ===
    # correct_predators = []
    # wrong_predators = []
    # if ids is not None:
    #     for i in range(len(y)):
    #         if y[i] == 1:
    #             if clusters[i] == 1:
    #                 correct_predators.append(ids[i])
    #             else:
    #                 wrong_predators.append(ids[i])

    #     logging.info(f"Predadores classificados corretamente (label=1, cluster=1): {correct_predators}")
    #     logging.info(f"Predadores classificados erroneamente (label=1, cluster≠1): {wrong_predators}")
        
      # === Análise de Componentes: Direção que mais distingue as classes ===
    #logging.info("=== Análise de Componentes ===")
    #class0_vecs = X[y == 0]
    #class1_vecs = X[y == 1]

    #mean0 = np.mean(class0_vecs, axis=0)
    #mean1 = np.mean(class1_vecs, axis=0)
    #direction = mean1 - mean0  # vetor que aponta da classe 0 para a 1

    # Normaliza a direção
    #direction /= np.linalg.norm(direction)

    # Projeta todos os embeddings nessa direção
    #projections = X @ direction

    # Plota histograma das projeções
    #plt.figure(figsize=(8, 6))
    #plt.hist(projections[y == 0], bins=30, alpha=0.6, label='Não predador', color='skyblue', edgecolor='k')
    #plt.hist(projections[y == 1], bins=30, alpha=0.6, label='Predador', color='red', edgecolor='k')
    #plt.axvline(np.mean(projections[y == 0]), color='blue', linestyle='--')
    #plt.axvline(np.mean(projections[y == 1]), color='darkred', linestyle='--')
    #plt.title('Projeção dos Embeddings na Direção de Máxima Separação')
    #plt.xlabel('Projeção na direção média (classe 1 - classe 0)')
    #plt.ylabel('Frequência')
    #plt.legend()
    #plt.tight_layout()
    #plt.savefig(os.path.join(dir, "direcao_discriminativa.png"))
    #plt.close()

    #logging.info(f"Distância entre vetores médios: {np.linalg.norm(mean1 - mean0):.4f}")
