import torch
import json
import datetime
import torch.nn as nn
from tqdm import tqdm
from models import CTNet
from models import Get_Data
from copy import deepcopy
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
from models.scheduler import WarmupLinearSchedule, WarmupCosineSchedule



def printlog(info):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+"=========="*8 + "%s"%nowtime)
    print(str(info)+"\n")


def IterOnce(net, opt, x, y, scheduler):
    preds, loss = net.forward(x, y)
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    scheduler.step()
    return preds, loss


# 评估指标
class Accuracy(nn.Module):
    def __init__(self):
        super().__init__()
        self.correct = nn.Parameter(torch.tensor(0.0),requires_grad=False)
        self.total = nn.Parameter(torch.tensor(0.0),requires_grad=False)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        preds = preds.argmax(dim=-1)
        m = (preds == targets).sum()
        n = targets.shape[0] 
        self.correct += m 
        self.total += n
        return m/n

    def compute(self):
        return self.correct.float() / self.total 
    
    def reset(self):
        self.correct -= self.correct
        self.total -= self.total


# 数据加载
def Get_DataLoaders(train_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def fit(net, train_loader, test_loader, opt, epochs, scheduler, device):
    metrics_dict = nn.ModuleDict({"acc":Accuracy()})
    metrics_dict.to(device)
    train_metrics_dict = deepcopy(metrics_dict)
    total_loss,step = 0,0 
    history = {}

    for epoch in range(1, epochs+1):
        printlog("Epoch {0} / {1}".format(epoch, epochs))
        # Training -------------------------------------------------
        net.train()
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (x, y) in loop:
            x = x.to(device)
            y = y.to(device)
            y = y.view(x.shape[0])
            preds, loss = IterOnce(net, opt, x, y, scheduler)
            #metrics
            step_metrics = {"train_"+name:metric_fn(preds, y).item() 
                            for name,metric_fn in train_metrics_dict.items()}
            
            step_log = dict({"train_loss":loss.item()},**step_metrics)

            total_loss += loss.item()
            
            step+=1
            if batch_idx!=len(train_loader)-1:
                loop.set_postfix(**step_log)
            else:
                epoch_loss = total_loss/step
                epoch_metrics = {"train_"+name:metric_fn.compute().item() 
                                for name,metric_fn in train_metrics_dict.items()}
                epoch_log = dict({"train_loss":epoch_loss},**epoch_metrics)
                loop.set_postfix(**epoch_log)

                for name,metric_fn in train_metrics_dict.items():
                    metric_fn.reset()
                    
        for name, metric in epoch_log.items():
            history[name] = history.get(name, []) + [metric]

        # Testing -------------------------------------------------
        net.eval()
        
        total_loss,step = 0,0
        loop = tqdm(enumerate(test_loader), total =len(test_loader),ncols=100)
        
        val_metrics_dict = deepcopy(metrics_dict)
        
        with torch.no_grad():
            for i, batch in loop: 
                features,labels = batch
                features = features.to(device)
                labels = labels.to(device)
                
                preds, loss = net(features, labels)

                step_metrics = {"val_"+name:metric_fn(preds, labels).item() 
                                for name,metric_fn in val_metrics_dict.items()}

                step_log = dict({"val_loss":loss.item()},**step_metrics)

                total_loss += loss.item()
                step+=1
                if i!=len(test_loader)-1:
                    loop.set_postfix(**step_log)
                else:
                    epoch_loss = (total_loss/step)
                    epoch_metrics = {"val_"+name:metric_fn.compute().item() 
                                    for name,metric_fn in val_metrics_dict.items()}
                    epoch_log = dict({"val_loss":epoch_loss},**epoch_metrics)
                    loop.set_postfix(**epoch_log)

                    for name,metric_fn in val_metrics_dict.items():
                        metric_fn.reset()
                        
        epoch_log["epoch"] = epoch           
        for name, metric in epoch_log.items():
            history[name] = history.get(name, []) + [metric]

    return history


def main():
    print("PyTorch version:", torch.__version__)
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    print(torch.cuda.is_available())
    print("Using Device: ", device)

    train_dataset, test_dataset = Get_Data()
    train_loader, test_loader = Get_DataLoaders(train_dataset, test_dataset, batch_size=35)
    model = CTNet(device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=0.00005, amsgrad=True)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=3000, t_total=24000)
    # scheduler = WarmupCosineSchedule(optimizer, warmup_steps=3000, t_total=25000, cycles=0.7)
    history = fit(model, train_loader, test_loader, optimizer, 30, scheduler, device)

    with open('history.json', 'w') as file:
        json.dump(history, file)

    return True


if __name__ == '__main__':
    main()