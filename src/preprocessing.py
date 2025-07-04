import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
from utils import clean_text

import re
import emoji


def normalize_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)
    text = re.sub(r'[^a-z0-9\s.,!?\'"]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_conversations(xml_file, predator_ids=None):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    conversations = []
    for conv in root.findall('conversation'):
        conv_id = conv.attrib.get('id', None)
        messages = []
        authors_in_conv = set()
        for message in conv.findall('message'):
            author_elem = message.find('author')
            text_elem = message.find('text')
            author = author_elem.text.strip() if author_elem is not None and author_elem.text else 'UNKNOWN'
            text = text_elem.text.strip() if text_elem is not None and text_elem.text else ''
            if text:
                messages.append((author, text))
                authors_in_conv.add(author)
        if messages:
            label = 1 if predator_ids and authors_in_conv.intersection(predator_ids) else 0
            conversations.append({'id': conv_id, 'messages': messages, 'label': label})
    return conversations

def prepare_data(conversations):
    texts = []
    labels = []
    ids = []
    for conv in conversations:
        conversation_text = ' '.join([clean_text(text) for _, text in conv['messages']])
        texts.append(conversation_text)
        labels.append(conv['label'])
        ids.append(conv['id'])
    return texts, labels, ids

def prepare_predator_texts(conversations, predator_ids):
    texts = []
    labels = []
    for conv in conversations:
        predator_msgs = [clean_text(text) for author, text in conv['messages'] if author in predator_ids]
        if predator_msgs:
            texts.append(' '.join(predator_msgs))
            labels.append(conv['label'])
    return texts, labels