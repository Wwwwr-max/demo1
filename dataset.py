from torch.utils.data import Dataset
import torch
from collections import Counter
# 处理数据
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
def pro_data(root):
    items = []
    labels_counter = Counter()
    with open(root,'r',encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            id,code,name,text,key = line.split('_!_',maxsplit=4)
            labels = int(code)-100
            labels_counter[labels] += 1
            if key:
                text = key +','+text
            else:
                text = text
            items.append({
                "text":text,
                "labels":labels
            })
    return items,labels_counter

# 重写dataset方法
class Mydata(Dataset):
    def __init__(self,root):
        super(Mydata,self).__init__()
        self.items,_ = pro_data(root)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]