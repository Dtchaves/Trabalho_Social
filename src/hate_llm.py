
import sys

import os

os.environ['TRANSFORMERS_CACHE'] = '/scratch/diogochaves/hf_cache'
os.environ['HF_HOME'] = '/scratch/diogochaves/hf_cache'

import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import logging




from utils import load_predator_ids
from preprocessing import load_conversations, prepare_data, prepare_predator_texts, prepare_non_predator_texts

# Set up logging for cleaner output
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

from transformers import logging as hf_logging
hf_logging.set_verbosity_error()


# --- Caminho para salvar os Plots ---
PLOTS_SAVE_DIR = '/home_cerberus/disk2/diogochaves/SOCIAL/predators/analyses_result'
os.makedirs(PLOTS_SAVE_DIR, exist_ok=True)
logging.info(f"Os gráficos serão salvos em: {PLOTS_SAVE_DIR}")




# --- Data Loading and Preparation ---
# --- Caminhos dos Dados ---
DATA_DIR_TRAIN = '../data/training'
XML_FILE_TRAIN = os.path.join(DATA_DIR_TRAIN, 'pan12-sexual-predator-identification-training-corpus-2012-05-01.xml')
PREDATOR_FILE_TRAIN = os.path.join(DATA_DIR_TRAIN, 'pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt')

DATA_DIR_TEST = '../data/test'
XML_FILE_TEST = os.path.join(DATA_DIR_TEST, 'pan12-sexual-predator-identification-test-corpus-2012-05-17.xml')
PREDATOR_FILE_TEST = os.path.join(DATA_DIR_TEST, 'pan12-sexual-predator-identification-groundtruth-problem1.txt')

# --- Carregar e Combinar Dados ---
logging.info("Carregando e combinando dados de treino e teste...")
predator_ids_train = load_predator_ids(PREDATOR_FILE_TRAIN)
predator_ids_test = load_predator_ids(PREDATOR_FILE_TEST)

conversations_train = load_conversations(XML_FILE_TRAIN, predator_ids_train)
conversations_test = load_conversations(XML_FILE_TEST, predator_ids_test)

all_conversations = conversations_train + conversations_test
all_predator_ids = predator_ids_train.union(predator_ids_test)

logging.info(f"Total de conversas (treino + teste): {len(all_conversations)}")
logging.info(f"Total de IDs de predadores (treino + teste): {len(all_predator_ids)}")

# Preparar textos de predadores e não-predadores
predator_texts, predator_ids, predator_types = prepare_predator_texts(all_conversations, all_predator_ids)
non_predator_texts, non_predator_ids, non_predator_types = prepare_non_predator_texts(all_conversations, all_predator_ids)

# Combinar todos os textos para classificação
all_texts = (predator_texts + non_predator_texts)
all_text_ids = (predator_ids + non_predator_ids)
all_text_types = (predator_types + non_predator_types) # 'predator' or 'non_predator'

logging.info(f"Total de segmentos de texto de predadores: {len(predator_texts)}")
logging.info(f"Total de segmentos de texto de não-predadores: {len(non_predator_texts)}")
logging.info(f"Total de segmentos de texto combinados para análise: {len(all_texts)}")

# --- Definição dos Modelos ---
models_dict = {
    "Falconsai": "Falconsai/offensive_speech_detection",
    "KoalaAI-Offensive": "KoalaAI/OffensiveSpeechDetector",
    "KoalaAI-Moderation": "KoalaAI/Text-Moderation",
    "DeHateBERT": "Hate-speech-CNERG/dehatebert-mono-english",
    "Toxic-BERT": "unitary/toxic-bert",
}

# --- Função Principal de Execução e Plotagem ---
def run_and_plot_model(selected_model_alias: str):
    """
    Executa o modelo especificado em todos os textos (predadores e não-predadores)
    e plota os resultados, salvando os gráficos no diretório especificado.

    Args:
        selected_model_alias (str): O nome curto (alias) do modelo a ser executado.
                                     Deve ser uma chave válida em `models_dict`.
    """
    if selected_model_alias not in models_dict:
        logging.error(f"Erro: Modelo '{selected_model_alias}' não encontrado. Escolha um dos seguintes: {list(models_dict.keys())}")
        return

    model_name_or_path = models_dict[selected_model_alias]
    logging.info(f"\n--- Processando modelo: {selected_model_alias} ({model_name_or_path}) ---")

    try:
        # Initialize the pipeline for text classification
        # Set return_all_scores=True for models like toxic-bert that output multiple labels
        # max_length e truncation são importantes para lidar com textos longos
        clf = pipeline("text-classification", model=model_name_or_path, truncation=True, max_length=512, return_all_scores=True)
    except Exception as e:
        logging.error(f"Erro ao carregar o modelo {model_name_or_path}: {e}")
        return

    model_results = []
    batch_size = 32 # Ajuste com base na sua memória de GPU/CPU

    # Crie uma barra de progresso simples
    num_batches = (len(all_texts) + batch_size - 1) // batch_size
    logging.info(f"Iniciando inferência para {len(all_texts)} textos em {num_batches} batches...")

    for i in range(0, len(all_texts), batch_size):
        batch_texts = all_texts[i:i+batch_size]
        batch_ids = all_text_ids[i:i+batch_size]
        batch_types = all_text_types[i:i+batch_size]

        try:
            predictions = clf(batch_texts)
        except Exception as e:
            logging.error(f"Erro ao fazer inferência no batch {i}-{i+len(batch_texts)}: {e}")
            for j in range(len(batch_texts)):
                model_results.append({
                    "conversation_id": batch_ids[j],
                    "text_type": batch_types[j],
                    "label": "ERROR",
                    "score": 0.0
                })
            continue

        for j, pred_list in enumerate(predictions):
            current_text_id = batch_ids[j]
            current_text_type = batch_types[j]

            if isinstance(pred_list, dict): # Single label output (e.g., Offensive/Non-Offensive)
                model_results.append({
                    "conversation_id": current_text_id,
                    "text_type": current_text_type,
                    "label": pred_list['label'],
                    "score": round(pred_list['score'], 4)
                })
            elif isinstance(pred_list, list) and all(isinstance(p, dict) for p in pred_list): # Multi-label output (e.g., Toxic-BERT)
                primary_label = max(pred_list, key=lambda x: x['score'])
                entry = {
                    "conversation_id": current_text_id,
                    "text_type": current_text_type,
                    "label": primary_label['label'], # Highest scoring label
                    "score": round(primary_label['score'], 4) # Score of the highest scoring label
                }
                # Add all scores as separate columns for detailed analysis
                for p_item in pred_list:
                    entry[p_item['label']] = round(p_item['score'], 4)
                model_results.append(entry)
            else:
                logging.warning(f"Formato de predição inesperado para {selected_model_alias}: {pred_list}")
                model_results.append({
                    "conversation_id": current_text_id,
                    "text_type": current_text_type,
                    "label": "UNKNOWN_FORMAT",
                    "score": 0.0
                })
        # Atualiza a barra de progresso
        sys.stdout.write(f"\rProgresso: {min(i + batch_size, len(all_texts))}/{len(all_texts)} textos processados")
        sys.stdout.flush()
    sys.stdout.write("\n") # Nova linha após a barra de progresso

    if not model_results:
        logging.warning(f"Nenhum resultado foi gerado para o modelo {selected_model_alias}. Pulando a plotagem.")
        return

    df_results = pd.DataFrame(model_results)
    print(df_results)
    logging.info(f"Resultados coletados para {selected_model_alias}. Shape: {df_results.shape}")
    # logging.info(f"Primeiras 5 linhas dos resultados:\n{df_results.head()}") # Descomente para ver o head

    # --- Plotagem dos Resultados ---
    sns.set_theme(style="whitegrid")

    logging.info("Primeiro Gráfico")
    # 1. Distribuição das Classificações por Tipo de Texto
    plt.figure(figsize=(14, 7))
    sns.countplot(data=df_results, x='label', hue='text_type', palette='viridis')
    plt.title(f'Distribuição das Classificações para {selected_model_alias} por Tipo de Texto', fontsize=16)
    plt.xlabel('Classificação')
    plt.ylabel('Contagem')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Tipo de Texto')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_SAVE_DIR, f'{selected_model_alias}_label_distribution.png'))
    plt.close() # Fechar o gráfico para liberar memória

    logging.info("Segundo Gráfico")
    # 2. Distribuição dos Scores de Confiança por Tipo de Texto (Histograma/Densidade)
    plt.figure(figsize=(14, 7))
    sns.histplot(data=df_results, x='score', hue='text_type', kde=True, palette='magma', stat='density', common_norm=False)
    plt.title(f'Distribuição dos Scores de Confiança para {selected_model_alias} por Tipo de Texto', fontsize=16)
    plt.xlabel('Score de Confiança')
    plt.ylabel('Densidade')
    plt.legend(title='Tipo de Texto')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_SAVE_DIR, f'{selected_model_alias}_score_distribution.png'))
    plt.close()

    logging.info("Terceiro Gráfico")
    # 3. Distribuição dos Scores por Classificação e Tipo de Texto (Box Plot)
    plt.figure(figsize=(16, 8))
    sns.boxplot(data=df_results, x='label', y='score', hue='text_type', palette='cividis')
    plt.title(f'Distribuição dos Scores por Classificação e Tipo de Texto para {selected_model_alias}', fontsize=16)
    plt.xlabel('Classificação')
    plt.ylabel('Score de Confiança')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Tipo de Texto')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_SAVE_DIR, f'{selected_model_alias}_label_score_boxplot.png'))
    plt.close()

    plt.figure(figsize=(14, 7))


    logging.info("Terceiro Gráfico")
    # Calcular proporção manualmente
    proportions = (
        df_results
        .groupby(['text_type', 'label'])
        .size()
        .groupby(level=0)
        .transform(lambda x: x / x.sum())
        .reset_index(name='proportion')
    )

    sns.barplot(data=proportions, x='label', y='proportion', hue='text_type', palette='viridis')

    plt.title(f'Proporção das Classificações para {selected_model_alias} por Tipo de Texto', fontsize=16)
    plt.xlabel('Classificação')
    plt.ylabel('Proporção')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Tipo de Texto')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_SAVE_DIR, f'{selected_model_alias}_label_proportion.png'))
    plt.close()

    # 4. Plot Específico para Modelos Multi-Label (Ex: Toxic-BERT, KoalaAI-Moderation)
    # Identifica as colunas de score específicas para o modelo atual, excluindo 'score' geral, 'conversation_id', etc.
    score_columns = [col for col in df_results.columns if col not in ['conversation_id', 'text_type', 'label', 'score']]

    if score_columns: # Se houver colunas de scores específicos (multi-label)
        logging.info(f"Gerando gráficos de categorias específicas para {selected_model_alias}...")
        df_melted_specific_scores = df_results.melt(
            id_vars=['conversation_id', 'text_type'],
            value_vars=score_columns,
            var_name='Categoria_Toxicidade',
            value_name='Score_Detalhado'
        )
        # Filtra para remover entradas onde o score detalhado é NaN (ocorre se uma categoria não está presente em todas as predições)
        df_melted_specific_scores = df_melted_specific_scores.dropna(subset=['Score_Detalhado'])

        if not df_melted_specific_scores.empty:
            plt.figure(figsize=(16, 9))
            sns.boxplot(data=df_melted_specific_scores, x='Categoria_Toxicidade', y='Score_Detalhado', hue='text_type', palette='Spectral')
            plt.title(f'Scores por Categoria Detalhada e Tipo de Texto para {selected_model_alias}', fontsize=16)
            plt.xlabel('Categoria de Toxicidade/Moderação')
            plt.ylabel('Score de Confiança')
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Tipo de Texto')
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_SAVE_DIR, f'{selected_model_alias}_detailed_scores_boxplot.png'))
            plt.close()
        else:
            logging.warning(f"Nenhum score detalhado válido para plotar para {selected_model_alias}. Pulando gráfico detalhado.")

    logging.info(f"Análise para o modelo {selected_model_alias} concluída e gráficos salvos.")

# --- Execução Principal ---
if __name__ == "__main__":
    # Verifique se o nome do modelo foi passado como argumento
    if len(sys.argv) < 2:
        logging.error("Uso: python seu_script.py <nome_do_modelo>")
        logging.info(f"Modelos disponíveis: {list(models_dict.keys())}")
        sys.exit(1)

    model_to_run_alias = sys.argv[1]
    run_and_plot_model(model_to_run_alias)