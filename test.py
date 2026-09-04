import json
import torch
import dataset,model_m,train
from torch.utils.data import DataLoader
with open("./demo_config.json",'r',encoding='utf-8') as f:
    demo_config = json.load(f)
test = dataset.Mydata(demo_config['test_path'])
test_loader = DataLoader(test,batch_size = demo_config['batch_size'],
                              shuffle = False,collate_fn = model_m.collate_fn)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model_m.Mymodel(num_labels=demo_config["num_labels"])
model.load_state_dict(torch.load("./best_model.pth", map_location=device))
model.to(device)
a = train.yz(test_loader, model, device)
print(f"dev_acc:{a:.4f}")
