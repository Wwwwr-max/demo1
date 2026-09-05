import torch
from transformers import AutoTokenizer
import dataset,model_m,train,config
from torch.utils.data import DataLoader
demo_config = config.load_config()
tokenizer = AutoTokenizer.from_pretrained(demo_config['model_path'])
collate_fn = dataset.make_collate_fn(tokenizer, demo_config["max_length"])
_,train_counter = dataset.pro_data(demo_config['train_path'])
all_label = sorted(train_counter.keys())
max_label = max(all_label)
num_labels = max_label + 1
test = dataset.Mydata(demo_config['test_path'])
test_loader = DataLoader(test,batch_size = demo_config['batch_size'],
                              shuffle = False,collate_fn = collate_fn)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model_m.Mymodel(model_path=demo_config["model_path"],
                        num_labels=num_labels)
model.load_state_dict(torch.load("./best_model.pth", map_location=device))
model.to(device)
test_acc, test_p, test_r, test_f1 = train.verify(test_loader, model, device, num_labels)
print(f"test_acc:{test_acc:.4f} | p_macro:{test_p:.4f} | r_macro:{test_r:.4f} | f1_macro:{test_f1:.4f}")