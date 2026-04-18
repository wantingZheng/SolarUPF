import numpy as np
import torch.nn.functional as F  
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models.efficient_kan import KAN
import copy
# torch.cuda.set_device(1)  # 
if torch.cuda.is_available():
    device = torch.device("cuda")
    # print(f"CUDA is_available， GPU is: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    # print("CUDA is_unavailable，using CPU")
        
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fcs1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fcs2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()        
        self.fc1 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fcs1(x)
        out = self.fcs2(out)
        out = self.relu2(out)
        q1_pred=self.fc1(out)         
        return q1_pred

class KAN_interval(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(KAN_interval, self).__init__()
        self.fcs1 = KAN([input_dim, hidden_dim])
        # self.relu1 = nn.ReLU()
        # self.fcs2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()        
        self.fc1 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fcs1(x)
        # out = self.fcs2(out)
        out = self.relu2(out)
        q1_pred=self.fc1(out)         
        return q1_pred

class RegressionDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.targets[idx]
        return x, y

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, lstm_out):
        # lstm_out shape: [batch, seq_len, hidden_dim]
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        return (attn_weights * lstm_out).sum(dim=1)  # 加权求和
        
def quantile_loss_single(pred, target, quantiles, median_weight=1.0):
    Q = len(quantiles)
    if target.dim() == 1:
        target = target.unsqueeze(1)
    target_q = target.expand(-1, Q)

    errors = target_q - pred

    losses = []
    for i, q in enumerate(quantiles):
        e = errors[:, i]
        loss_q = torch.max((q - 1) * e, q * e)

        # 如果当前 q 是 0.5，就加权
        if abs(q - 0.5) < 1e-12:
            loss_q = loss_q * median_weight

        losses.append(loss_q)

    loss_pinball = torch.stack(losses, dim=1).mean()

    sorted_idx = torch.tensor([i for _, i in sorted([(q, i) for i, q in enumerate(quantiles)])],
                              device=pred.device, dtype=torch.long)
    pred_sorted = pred.index_select(dim=1, index=sorted_idx)
    crossing_loss = torch.relu(pred_sorted[:, :-1] - pred_sorted[:, 1:]).mean()

    return loss_pinball + crossing_loss

class CustomMLP:
    def __init__(self, zhixin,input_dim=1,hidden_dim=128,lr=0.01,batch_size=256,num_epochs=70,method_choose='MLP') -> None:
        # zhixin=[0, 0.3, 0.5, 0.7 ,0.9]
        quantiles = 1 - (1 - np.array(zhixin)) / 2
        self.quantiles = quantiles
        self.fenwei = np.append(quantiles, 1 - quantiles[1:])
        input_dim = input_dim
        hidden_dim =hidden_dim
        # num_layers = num_layers
        output_dim = len(zhixin)*2-1
        self.output_dim=output_dim
        self.learn_reate=lr
        self.batch_size=batch_size
        self.num_epochs=num_epochs
        if method_choose=='MLP':
            self.model = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim) 
        else:
            self.model = KAN_interval(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim) 
            # self.model = LSTM_attention_interval(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers) 
        
        self.zhixin=zhixin
        
    def train(self, x, y,x_v,y_v):
        x1=x.copy()
        y1=y.copy()
        x_v1=x_v.copy()
        y_v1=y_v.copy()
        X_train_nor=x1
        y_train_nor=y1.reshape(x.shape[0],1)
        X_valid_nor=x_v1
        y_valid_nor=y_v1.reshape(x_v.shape[0],1)
        dataset_train = RegressionDataset(X_train_nor, y_train_nor)
        dataset_valid = RegressionDataset(X_valid_nor, y_valid_nor)

        learn_reate=self.learn_reate
        num_epochs=self.num_epochs
        batch_size=self.batch_size
        output_dim=self.output_dim

        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)

        model=self.model

        model.to(device)

        optimiser = torch.optim.Adam(model.parameters(), lr=learn_reate)
    
        zhixin=self.zhixin
        quantile_qs=self.fenwei
        patience = 10          #  self.patience
        min_delta = 0.0        #  min_delta
        best_val_loss = float('inf')
        best_state_dict = None
        no_improve_count = 0

        model=self.model

        model.to(device)

        for epoch in range(num_epochs):
            # ===== train=====
            model.train()
            train_loss_sum = 0.0
            train_batches = 0

            for x_batch, y_batch in dataloader_train:
                x_batch = x_batch.to(device).float()
                y_batch = y_batch.to(device).float()
                
                
                y_pred = model(x_batch)          
                loss = quantile_loss_single(y_pred, y_batch, quantile_qs, median_weight=1)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                train_loss_sum += loss.item()
                train_batches += 1

            avg_train_loss = train_loss_sum / max(train_batches, 1)
        
                # ===== valid=====
            model.eval()
            val_loss_sum = 0.0
            val_batches = 0

            with torch.no_grad():
                for x_val_batch, y_val_batch in dataloader_valid:
                    x_val_batch = x_val_batch.to(device).float()
                    y_val_batch = y_val_batch.to(device).float()

                    y_val_pred = model(x_val_batch)
                    val_loss = quantile_loss_single(y_val_pred, y_val_batch, quantile_qs, median_weight=1)

                    val_loss_sum += val_loss.item()
                    val_batches += 1

            avg_val_loss = val_loss_sum / max(val_batches, 1)

            print(f"Epoch {epoch:03d} | train_loss = {avg_train_loss:.6f} | val_loss = {avg_val_loss:.6f}")

            # ===== Early Stopping=====
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                best_state_dict = copy.deepcopy(model.state_dict())
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(f"Early stopping at epoch {epoch}, best_val_loss = {best_val_loss:.6f}")
                    break

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        self.model = model

        
    def predict(self, x, y_mean, y_std):
       
        batch_size=self.batch_size
        x1=x.copy()
        X_test_nor=x1     
        dataset_test = RegressionDataset(X_test_nor, X_test_nor)
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
        model=self.model

        with torch.no_grad():
            # all_predictions_list=[]
            model.eval()
            all_predictions=[]
            for x_batch, y_batch in dataloader_test:
                x_batch = x_batch.to(device)

                test_pred1= model(x_batch)

                all_predictions.append(test_pred1.clone().detach())

            all_predictions = torch.cat(all_predictions, dim=0)

            all_predictions=all_predictions.cpu()
            all_predictions_np = all_predictions.numpy()
     

            all_predictions_np1=all_predictions_np.copy()
    
        upper = np.zeros((len(x),len(self.quantiles) - 1)) 
        lower = np.zeros((len(x),len(self.quantiles) - 1))
        y = all_predictions_np1
        zhixin=self.zhixin
        for i in range(len(self.fenwei)):
            if i == 0:
                # 0.5 quantile 
                y_middle = y[:,i] * y_std + y_mean
            elif i <= len(zhixin)-1:
                upper[:, (i - 1) % len(self.quantiles)] = y[:,i] * y_std + y_mean
            else:
                lower[:, (i - len(self.zhixin)) % len(self.quantiles)] = y[:,i] * y_std + y_mean
                
        return y_middle, upper, lower