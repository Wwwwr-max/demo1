import torch
import dataset,model_m,train,config
from torch.utils.data import DataLoader
demo_config = config.load_config()
tokenizer = model_m.load_tokenizer(demo_config["model_path"])
collate_fn = model_m.make_collate_fn(tokenizer, demo_config["max_length"])
test = dataset.Mydata(demo_config['test_path'])
test_loader = DataLoader(test,batch_size = demo_config['batch_size'],
                              shuffle = False,collate_fn = collate_fn)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model_m.Mymodel(model_path=demo_config["model_path"],
                        num_labels=demo_config["num_labels"])
model.load_state_dict(torch.load("./best_model.pth", map_location=device))
model.to(device)
a = train.yz(test_loader, model, device)
print(f"dev_acc:{a:.4f}")