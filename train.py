import torch
from transformers import AutoTokenizer
import dataset,model_m,config
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
demo_config = config.load_config()
# tp,fp,fn--->对的预测为对，错的预测为对，对的预测为错
def calc_macro_prf(y_true, y_pre, num_labels):
    tp = [0]*num_labels
    fp = [0]*num_labels
    fn = [0]*num_labels
    for t, p in zip(y_true, y_pre):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    p_list = []
    r_list = []
    f1_list = []
    # 计算每个分类的p,r,f
    for i in range(num_labels):
        if tp[i] + fp[i] == 0:
            pi = 0.0
        else:
            pi = tp[i] / (tp[i] + fp[i])
        if tp[i] + fn[i] == 0:
            ri = 0.0
        else:
            ri = tp[i] / (tp[i] + fn[i])
        if pi + ri == 0:
            fi = 0.0
        else:
            fi = 2 * pi * ri / (pi + ri)
        p_list.append(pi)
        r_list.append(ri)
        f1_list.append(fi)
    p_macro = sum(p_list) / num_labels
    r_macro = sum(r_list) / num_labels
    f1_macro = sum(f1_list) / num_labels
    return  p_macro, r_macro, f1_macro
def verify(loader, model, device,num_labels):
    model.eval()
    total_correct = 0
    total_num = 0
    y_true = []
    y_pre = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch)
            per = torch.argmax(output, dim=-1)
            total_correct += (per == batch["labels"]).sum().item()
            total_num += batch["labels"].size(0)
            y_true.extend(batch['labels'].cpu().numpy().tolist())
            y_pre.extend(per.cpu().numpy().tolist())
        acc = total_correct / total_num
        p_macro, r_macro, f1_macro = calc_macro_prf(y_true, y_pre, num_labels)
        model.train()
        return acc,p_macro,r_macro,f1_macro

def train_T(loader, model, device, loss_fn, optim):
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
    _,train_counter = dataset.pro_data(demo_config['train_path'])
    all_label = sorted(train_counter.keys())
    max_label = max(all_label)
    num_labels = max_label + 1
    tokenizer = AutoTokenizer.from_pretrained(demo_config['model_path'])
    collate_fn = dataset.make_collate_fn(tokenizer, demo_config["max_length"])
    train = dataset.Mydata(demo_config['train_path'])
    train_loader = DataLoader(train,batch_size = demo_config['batch_size'],
                                  shuffle = True,collate_fn = collate_fn)
    dev = dataset.Mydata(demo_config['dev_path'])
    dev_loader = DataLoader(dev,batch_size = demo_config['batch_size'],
                                  shuffle = False,collate_fn = collate_fn)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model_m.Mymodel(model_path=demo_config["model_path"],
                            num_labels=num_labels)
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(),lr = demo_config['learning_rate'])
    loss_fn = torch.nn.CrossEntropyLoss()
    epoch = demo_config["epoch"]
    best_acc = 0.0
    best_f1 = 0.0
    patience = 3.0
    early_stop = 0.0
    for e in range(epoch):
        print(f"===== Epoch {e+1}/{epoch} =====")
        avg_loss = train_T(train_loader,model, device, loss_fn, optim)
        dev_acc, dev_p, dev_r, dev_f1 = verify(dev_loader, model, device,num_labels)
        writer.add_scalar('aver/epoch', avg_loss, e + 1)
        writer.add_scalar('acc/epoch', dev_acc, e + 1)
        writer.add_scalar('dev/p_macro', dev_p, e + 1)
        writer.add_scalar('dev/r_macro', dev_r, e + 1)
        writer.add_scalar('dev/f1_macro', dev_f1, e + 1)
        print(f"train_loss:{avg_loss:.4f} | dev_acc:{dev_acc:.4f} | p_macro:{dev_p:.4f} | r_macro:{dev_r:.4f} | f1_macro:{dev_f1:.4f}")
        # 保存最优模型
        if dev_acc>best_acc :
            best_acc = dev_acc
        if dev_f1 > best_f1:
            best_f1= dev_f1
            early_stop = 0.0
            torch.save(model.state_dict(),"./best_model.pth")
            print(f"保存最优模型, best_f1:{best_f1:.4f}")
        else:
            early_stop += 1
            if early_stop >patience:
                print(f"连续{early_stop}轮没有提升，触发早停")
                break
    print("训练结束，最优验证集精度：", best_acc)
    writer.close()



