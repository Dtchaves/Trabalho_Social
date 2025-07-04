import re

def clean_text(text):
    text = text.replace('<email/>', '')

    text = re.sub(r'http\S+|www\S+', '', text)

    text = re.sub(r'(.)\1{3,}', r'\1\1', text)

    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"]+', '', text)

    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def load_predator_ids(predator_file):
    with open(predator_file, 'r', encoding='utf-8') as f:
        predators = {line.strip() for line in f if line.strip()}
    return predators