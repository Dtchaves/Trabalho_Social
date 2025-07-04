# Detecção de Predadores Sexuais em Conversas Textuais

Este projeto avalia modelos de linguagem pré-treinados e classificadores supervisionados na tarefa de identificar predadores sexuais com base em conversas textuais. Também foram aplicadas técnicas de embeddings, clusterização e redução de dimensionalidade para análise do espaço latente dos dados.

---

## 📁 Estrutura

### `analyses_result/`

Contém os resultados dos seguintes **modelos de linguagem pré-treinados**:

- `Falconsai/offensive_speech_detection`: detector de discurso ofensivo.
- `KoalaAI/OffensiveSpeechDetector`: modelo para identificar linguagem ofensiva.
- `KoalaAI/Text-Moderation`: modelo geral de moderação de texto.
- `dehatebert-mono-english`: especializado em detectar discurso de ódio.
- `toxic-bert`: modelo para detectar toxicidade textual.

Aplicados diretamente nos dados rotulados como predador ou não predador.

---

### `results/`

Contém os experimentos com **classificadores supervisionados** e **métodos de embeddings**, além de análises de espaço latente.

#### 🔤 Embeddings Utilizados

- **all-MiniLM-L6-v2**: modelo compacto baseado em Transformers com bom desempenho em tarefas de similaridade semântica.
- **bert-base-uncased**: versão básica do BERT, insensível a caixa alta, usada para gerar embeddings ricos em contexto.
- **TfidfEmbedder**: gera vetores esparsos com base na frequência de termos ponderada (TF-IDF).
- **train_word2vec**: Word2Vec treinado nos próprios dados, produzindo vetores densos semânticos para cada palavra.

#### ✅ Classificadores Supervisionados

- **Logistic Regression**: modelo linear simples, usado como baseline.
- **SVM (Support Vector Machine)**: modelo robusto para separação de classes com margem máxima.
- **Random Forest**: conjunto de árvores de decisão que melhora generalização.
- **MLP Classifier**: rede neural simples com uma ou poucas camadas ocultas.
- **Deep MLP**: versão mais profunda do MLP com múltiplas camadas.
- **Deep Residual MLP**: variante do Deep MLP com conexões residuais para facilitar o aprendizado em redes profundas.

#### 📊 Análise do Espaço Latente

- **KMeans**: algoritmo de clusterização que agrupa dados com base em similaridade euclidiana.
- **PCA (Principal Component Analysis)**: redução de dimensionalidade linear que projeta os dados nas direções de maior variância.
- **t-SNE (t-distributed Stochastic Neighbor Embedding)**: técnica não linear que preserva relações locais em 2D ou 3D.
- **UMAP (Uniform Manifold Approximation and Projection)**: técnica de redução de dimensionalidade que preserva estrutura global e local dos dados.

As projeções são analisadas tanto com as **labels reais** (predador/não predador) quanto com os **clusters encontrados** via KMeans.

---

## 🔧 Resumo dos Métodos

| Categoria              | Modelos/Métodos                                      |
|------------------------|------------------------------------------------------|
| Modelos Pré-Treinados  | Falconsai, KoalaAI, DeHateBERT, Toxic-BERT           |
| Embeddings             | MiniLM, BERT, TF-IDF, Word2Vec                       |
| Classificadores        | Logistic Regression, SVM, Random Forest, MLPs        |
| Clusterização          | KMeans                                               |
| Redução Dimensional    | PCA, t-SNE, UMAP                                     |

---

## 📝 Observação

- `analyses_result/`: aplicação direta de modelos de moderação de texto.
- `results/`: experimentos completos com classificadores treinados, embeddings, clusterização e visualização do espaço vetorial.

