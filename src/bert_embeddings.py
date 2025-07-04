import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

class BertEmbedder:
    def __init__(self, model_name='bert-base-uncased', device=None):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.bert = self.bert.to(self.device)
        self.bert.eval()

    def get_embeddings(self, texts, max_length=512, batch_size=16):
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]
            encoded = self.tokenizer(batch, padding=True, truncation=True,
                                    max_length=max_length, return_tensors='pt')
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            with torch.no_grad():
                output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                cls_embeds = output.last_hidden_state[:,0,:].cpu().numpy()
                embeddings.append(cls_embeds)
        return np.vstack(embeddings)