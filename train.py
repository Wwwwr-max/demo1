import json
import torch
import dataset,model_m
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
with open("./demo_config.json",'r',encoding='utf-8') as f:
    demo_config = json.load(f)
def yz(loader, model, device):
    model.eval()
    total_correct = 0
    total_num = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch)
            per = torch.argmax(output, dim=-1)
            total_correct += (per == batch["labels"]).sum().item()
            total_num += batch["labels"].size(0)
        acc = total_correct / total_num
        model.train()
        return acc

def xl(loader):
    model.train()
    total_train_loss = 0.0
    for step, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(**batch)
        loss = loss_fn(output, batch['labels'])
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_train_loss += loss.item()
    aver_loss = total_train_loss / len(loader)
    return aver_loss

# 加载数据集
if __name__ == "__main__":
    writer = SummaryWriter('./log')
    train = dataset.Mydata(demo_config['train_path'])
    train_loader = DataLoader(train,batch_size = demo_config['batch_size'],
                                  shuffle = True,collate_fn = model_m.collate_fn)
    dev = dataset.Mydata(demo_config['dev_path'])
    dev_loader = DataLoader(dev,batch_size = demo_config['batch_size'],
                                  shuffle = False,collate_fn = model_m.collate_fn)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model_m.Mymodel(num_labels=demo_config["num_labels"])
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(),lr = demo_config['learning_rate'])
    loss_fn = torch.nn.CrossEntropyLoss()
    epoch = demo_config["epoch"]
    best_acc = 0.0
    for e in range(epoch):
        print(f"===== Epoch {e+1}/{epoch} =====")
        avg_loss = xl(train_loader)
        dev_acc = yz(dev_loader, model, device)
        writer.add_scalar('aver/epoch', avg_loss, e + 1)
        writer.add_scalar('acc/epoch', dev_acc, e + 1)
        print(f"train_loss:{avg_loss:.4f} | dev_acc:{dev_acc:.4f}")
        # 保存最优模型
        if dev_acc > best_acc:
            best_acc = dev_acc
            torch.save(model.state_dict(),"./best_model.pth")
            print(f"保存最优模型, best_acc:{best_acc:.4f}")
    print("训练结束，最优验证集精度：", best_acc)



