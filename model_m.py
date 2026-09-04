import json
import torch.nn as nn
import torch
from transformers import AutoTokenizer, BertModel
with open("./demo_config.json","r",encoding="utf-8") as f:
    demo_config = json.load(f)
tokenizer = AutoTokenizer.from_pretrained(demo_config['model_path'])

def collate_fn(items):
    text = [x['text'] for x in items]
    labels = [x['labels'] for x in items]
    out = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=demo_config["max_length"],
        return_tensors="pt"
    )
    out['labels'] = torch.tensor(labels,dtype=torch.long)
    return out

class Mymodel(nn.Module):
    def __init__(self,num_labels):
        super(Mymodel,self).__init__()
        self.bert = BertModel.from_pretrained(demo_config['model_path'])
        num_input = self.bert.config.hidden_size
        self.liner = nn.Linear(num_input,num_labels)

    def forward(self,input_ids,attention_mask,token_type_ids =None,labels=None):
        bert_out = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
        )
        cls_emb = bert_out.last_hidden_state[:, 0, :]
        logits = self.liner(cls_emb)
        return logits




