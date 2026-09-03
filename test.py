import torch
from transformers import AutoModelForSequenceClassification
from train import MyDataset,collate_fn
from torch.utils.data import DataLoader
test = MyDataset('./data/test_1k.txt')
test_loader = DataLoader(test,8,shuffle=False,collate_fn=collate_fn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-chinese",
    num_labels=17)
model.to(device)
model.load_state_dict(torch.load("./best_model.pth", map_location=device))
model.eval()

total_correct = 0
total_num = 0
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(**batch)
        logit = output.logits
        per = torch.argmax(logit, dim=-1)
        total_correct += (per == batch["labels"]).sum().item()
        total_num += batch["labels"].size(0)
    acc = total_correct / total_num
print(f"test_acc: {acc:.4f}")