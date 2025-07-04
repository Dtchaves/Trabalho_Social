import re
import nltk
import emoji
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

# SHAP para explicabilidade
import shap

# Aqui só tentando melhorar os dados para ver se melhora a performance para a classe 1

# 1. Baixar recursos do nltk
nltk.download('stopwords')
nltk.download('wordnet')

# 2. Funções de pré-processamento
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)
    text = re.sub(r'[^a-z0-9\s.,!?\'"]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def process_tokens(text):
    tokens = re.findall(r'\b\w+\b', text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens_stem = [stemmer.stem(t) for t in tokens]
    tokens_lemma = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens_stem, tokens_lemma

def extract_linguistic_features(text):
    tokens = re.findall(r'\b\w+\b', text)
    num_tokens = len(tokens)
    num_sentences = text.count('.') + text.count('!') + text.count('?')
    avg_sentence_length = num_tokens / num_sentences if num_sentences > 0 else num_tokens
    num_questions = text.count('?')
    question_ratio = num_questions / num_sentences if num_sentences > 0 else 0
    sexual_words = ['sex', 'nude', 'hot', 'kiss', 'panties', 'dick', 'boobs', 'fuck']
    sexual_count = sum(text.lower().count(w) for w in sexual_words)
    return [num_tokens, num_sentences, avg_sentence_length, question_ratio, sexual_count]

texts = [' '.join([text for (author, text) in conv['messages']]) for conv in conversations]
labels = [conv['label'] for conv in conversations]  # 0 = normal, 1 = predador

texts_norm = [normalize_text(t) for t in texts]

ling_features = np.array([extract_linguistic_features(t) for t in texts_norm])

model = SentenceTransformer('all-MiniLM-L6-v2')
X_emb = model.encode(texts_norm, show_progress_bar=True)

X_full = np.hstack([X_emb, ling_features])

X_train, X_test, y_train, y_test, texts_train, texts_test = train_test_split(
    X_full, labels, texts_norm, test_size=0.2, stratify=labels, random_state=42)

rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("Relatório de classificação:\n", classification_report(y_test, y_pred, digits=4))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(rf, X_full, labels, cv=cv, scoring='f1')
print("F1 por fold:", scores)
print("F1 médio:", scores.mean())

y_prob = rf.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('Falso positivo')
plt.ylabel('Verdadeiro positivo')
plt.title('Curva ROC')
plt.legend()
plt.grid(True)
plt.show()

cm = confusion_matrix(y_test, y_pred)
print("Matriz de confusão:\n", cm)
fp_idx = [i for i, (yp, yt) in enumerate(zip(y_pred, y_test)) if yp == 1 and yt == 0]
fn_idx = [i for i, (yp, yt) in enumerate(zip(y_pred, y_test)) if yp == 0 and yt == 1]
print(f"Exemplos de Falsos Positivos ({len(fp_idx)}):")
for i in fp_idx[:3]:
    print(texts_test[i][:300], "\n---")
print(f"\nExemplos de Falsos Negativos ({len(fn_idx)}):")
for i in fn_idx[:3]:
    print(texts_test[i][:300], "\n---")

shap.initjs()
explainer = shap.TreeExplainer(rf)
sample_idx = np.random.choice(range(X_test.shape[0]), min(1000, X_test.shape[0]), replace=False)
shap_values = explainer.shap_values(X_test[sample_idx])

shap.summary_plot(shap_values[1], X_test[sample_idx], feature_names=[f"emb_{i}" for i in range(X_emb.shape[1])] + ['num_tokens', 'num_sentences', 'avg_sentence_length', 'question_ratio', 'sexual_word_count'])

shap.force_plot(explainer.expected_value[1], shap_values[1][0], X_test[sample_idx][0], feature_names=[f"emb_{i}" for i in range(X_emb.shape[1])] + ['num_tokens', 'num_sentences', 'avg_sentence_length', 'question_ratio', 'sexual_word_count'])

vectorizer = CountVectorizer(max_features=1000)
X_bow = vectorizer.fit_transform(texts_norm)
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X_bow)
for idx, topic in enumerate(lda.components_):
    print(f"Tópico {idx}: ", [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:][::-1]])