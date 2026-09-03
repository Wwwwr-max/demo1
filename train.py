import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification,Trainer,TrainingArguments
from torch.utils.data import Dataset,DataLoader
from torch.utils.tensorboard import SummaryWriter
# 处理数据，提取出标题或关键词和分类id
def read_data(root):
    with open(root,'r',encoding='utf-8') as f:
        items = []  # 存放文本和分类
        for line in f:
            line = line.strip()
            if not line:
                continue
            id,code,name,title,key = line.split('_!_',maxsplit=4)
            code = int(code) - 100    #训练的时候需要从0开始
            # 判断取title还是key
            if key :
                text = key+','+title
            else:
                text = title
            # 塞进items
            items.append({
                'text':text,
                'label':code
            })
    return items
# 重写dataset方法。
class MyDataset(Dataset):
    def __init__(self,root):
        super().__init__()
        self.items = read_data(root)
    def __len__(self):
        return len(self.items)

    def __getitem__(self,index):
        return self.items[index]
# 读取文本和标签
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
# 将文本分词编码
def collate_fn(items):
    texts = [x['text'] for x in items]
    labels = [x['label'] for x in items]
    out = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    out["labels"] = torch.tensor(labels, dtype=torch.long)
    return out
if __name__ == "__main__":
    writer = SummaryWriter('./log')
    train = MyDataset('./data/train_3k.txt')
    dev = MyDataset('./data/dev_1k.txt')
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-chinese",
        num_labels=17
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # 加载数据集
    train_loader = DataLoader(train,12,shuffle=True,collate_fn=collate_fn)
    dev_loader = DataLoader(dev,12,shuffle=False,collate_fn=collate_fn)

    # 参数
    optim = torch.optim.AdamW(model.parameters(),lr = 1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()
    epoch = 3
    best_acc = 0.0
    # 验证集
    def val():
        model.eval()
        total_correct = 0   #预测正确总数
        total_num = 0
        with torch.no_grad():
            for batch in dev_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                output = model(**batch)
                logit = output.logits
                per = torch.argmax(logit,dim =-1)
                total_correct += (per == batch["labels"]).sum().item()
                total_num += batch["labels"].size(0)
        acc = total_correct/total_num
        model.train()
        return acc

    for i in range(epoch):
        model.train()
        total_train_loss = 0.0

        for step,batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch)
            logit = output.logits
            loss = loss_fn(logit,batch['labels'])
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_train_loss += loss.item()
            if i == 1:
                writer.add_scalar('1loss/setp',loss,step)
            elif i == 2:
                writer.add_scalar('2loss/setp',loss,step)
            else:
                writer.add_scalar('3loss/setp',loss,step)
        acc = val()
        print(f"\n==== epoch {i + 1} finish, dev acc:{acc:.4f} ====\n")
        if acc>best_acc:
            best_acc = acc
            torch.save(model.state_dict(), './best_model.pth')
            print(f">>> 更新最佳权重，best_acc = {best_acc:.4f}，保存至 {'./best_model.pth'}\n")
    writer.close()