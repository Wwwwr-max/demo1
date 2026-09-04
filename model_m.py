import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertModel
def load_tokenizer(model_path):
    return AutoTokenizer.from_pretrained(model_path)

def make_collate_fn(tokenizer, max_length):
    def collate_fn(items):
        texts = [x["text"] for x in items]
        labels = [x["labels"] for x in items]
        out = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        out["labels"] = torch.tensor(labels, dtype=torch.long)
        return out
    return collate_fn

class Mymodel(nn.Module):
    def __init__(self, model_path, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.liner = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_emb = bert_out.last_hidden_state[:, 0, :]
        logits = self.liner(cls_emb)
        return logits