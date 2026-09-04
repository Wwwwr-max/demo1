from torch.utils.data import Dataset
# 处理数据

def pro_data(root):
    items = []
    with open(root,'r',encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            id,code,name,text,key = line.split('_!_',maxsplit=4)
            labels = int(code)-100
            if key:
                text = key +','+text
            else:
                text = text
            items.append({
                "text":text,
                "labels":labels
            })
    return items

# 重写dataset方法
class Mydata(Dataset):
    def __init__(self,root):
        super(Mydata,self).__init__()
        self.items = pro_data(root)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]