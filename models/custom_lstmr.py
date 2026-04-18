import numpy as np
import torch.nn.functional as F 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import copy
if torch.cuda.is_available():
    device = torch.device("cuda")

else:
    device = torch.device("cpu")

class LSTM_interval(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=9):
        super(LSTM_interval, self).__init__()
        self.hidden_dim = hidden_dim  
        self.num_layers = num_layers 
 
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        hidden_dim1=32
        self.fc = nn.Linear(hidden_dim, hidden_dim1)
        self.fc1 = nn.Linear(hidden_dim1, output_dim)

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.fc(out[:, -1, :])
        q1_pred=self.fc1(out)

        return q1_pred

class GRU_interval(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=9):
        super(GRU_interval, self).__init__()
        self.hidden_dim = hidden_dim 
        self.num_layers = num_layers 
   
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)

        hidden_dim1=32
        self.fc = nn.Linear(hidden_dim, hidden_dim1)
        self.fc1 = nn.Linear(hidden_dim1, output_dim)

    def forward(self, x):

        # h0 = torch.zeros(1, x.size(0), self.rnn.hidden_size).to(self.device)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
        out, _ = self.gru(x, h0)

        out = self.fc(out[:, -1, :])
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
        return (attn_weights * lstm_out).sum(dim=1) 
        
class LSTM_attention_interval(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM_attention_interval, self).__init__()
        self.hidden_dim = hidden_dim 
        self.num_layers = num_layers  

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = AttentionLayer(hidden_dim)

        hidden_dim1=32

        self.fc = nn.Linear(hidden_dim, output_dim)
    
        self.relu = nn.ReLU()

    def forward(self, x):
   
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)

        lstm_out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.attention(lstm_out)
        # out = self.fc(out[:, -1])
        out = self.fc(out)
        q1_pred=self.relu(out)
  
        return q1_pred


class CustomLSTM1:
    def __init__(self, zhixin,hidden_dim=128,num_layers=1,lr=0.01,batch_size=256,num_epochs=70,method_choose='LSTM') -> None:
        # zhixin=[0, 0.3, 0.5, 0.7 ,0.9]
        quantiles = 1 - (1 - np.array(zhixin)) / 2
        self.quantiles = quantiles
        self.fenwei = np.append(quantiles, 1 - quantiles[1:])
        input_dim = 1
        hidden_dim =hidden_dim
        num_layers = num_layers
        output_dim = len(zhixin)*2-1
        self.output_dim=output_dim
        self.learn_reate=lr
        self.batch_size=batch_size
        self.num_epochs=num_epochs
        if method_choose=='LSTM':
            self.model = LSTM_interval(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers) 
        else:
            self.model = GRU_interval(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers) 
            # self.model = LSTM_attention_interval(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers) 
        
        self.zhixin=zhixin

    def train(self, x, y, x_val, y_val):
        x1 = x.copy()
        y1 = y.copy()

        x_val1 = x_val.copy()
        y_val1 = y_val.copy()

        X_train_nor = x1.reshape(x.shape[0], x.shape[1], 1)
        y_train_nor = y1.reshape(x.shape[0], 1)

        X_val_nor = x_val1.reshape(x_val.shape[0], x_val.shape[1], 1)
        y_val_nor = y_val1.reshape(x_val.shape[0], 1)

        dataset_train = RegressionDataset(X_train_nor, y_train_nor)
        dataset_val = RegressionDataset(X_val_nor, y_val_nor)

        learn_reate = self.learn_reate
        num_epochs = self.num_epochs
        batch_size = self.batch_size
        output_dim = self.output_dim

        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

        model = self.model
        model.to(device)

        optimiser = torch.optim.Adam(model.parameters(), lr=learn_reate)

        # Early stopping settings
        patience = 20              # Number of epochs to wait without improvement
        min_delta = 1e-6           # Minimum improvement required
        best_val_loss = float('inf')
        best_epoch = -1
        patience_counter = 0
        best_model_wts = copy.deepcopy(model.state_dict())

        def quantile_loss(q, y_pred, y_true, lower_pred=None):
            e = y_true - y_pred
            loss = torch.max(q * e, (q - 1) * e).mean()

            if lower_pred is not None:
                penalty = torch.mean(torch.relu(lower_pred - y_pred))
                loss = loss + penalty

            return loss

        zhixin = self.zhixin
        quantile = 1 - (1 - zhixin)
        quantile1 = (1 - zhixin) / 2
        quantile_qs = self.fenwei

        # Function to compute total loss on one batch
        def compute_total_loss(y_pred, y_true):
            loss = 0.0
            for n in range(output_dim):
                if n == 0:
                    loss1 = quantile_loss(quantile_qs[n], y_pred[:, n], y_true)
                    loss = loss + loss1 * 2
                elif n <= len(zhixin) - 1:
                    loss1 = quantile_loss(quantile_qs[n], y_pred[:, n], y_true, y_pred[:, n - 1])
                    loss = loss + loss1
                else:
                    loss1 = quantile_loss(quantile_qs[n], y_pred[:, n], y_true)
                    loss = loss + loss1
            return loss

        for epoch in range(num_epochs):

            # =========================
            # Training
            # =========================
            model.train()
            train_loss_sum = 0.0
            train_sample_count = 0

            for x_batch, y_batch in dataloader_train:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device).view(-1)

                y_train_pred1 = model(x_batch)
                loss = compute_total_loss(y_train_pred1, y_batch)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                batch_size_now = x_batch.size(0)
                train_loss_sum += loss.item() * batch_size_now
                train_sample_count += batch_size_now

            avg_train_loss = train_loss_sum / train_sample_count

            # =========================
            # Validation
            # =========================
            model.eval()
            val_loss_sum = 0.0
            val_sample_count = 0

            with torch.no_grad():
                for x_batch, y_batch in dataloader_val:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device).view(-1)

                    y_val_pred1 = model(x_batch)
                    val_loss = compute_total_loss(y_val_pred1, y_batch)

                    batch_size_now = x_batch.size(0)
                    val_loss_sum += val_loss.item() * batch_size_now
                    val_sample_count += batch_size_now

            avg_val_loss = val_loss_sum / val_sample_count

            print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

            # =========================
            # Early stopping
            # =========================
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                patience_counter = 0
                best_model_wts = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                print(f"Best epoch: {best_epoch + 1}, Best Val Loss: {best_val_loss:.6f}")
                break

        # Restore the best model
        model.load_state_dict(best_model_wts)
        self.model = model


    def predict(self, x, y_mean, y_std):
       
        batch_size=self.batch_size
        x1=x.copy()
        X_test_nor=x1.reshape(x.shape[0],x.shape[1],1)        
        dataset_test = RegressionDataset(X_test_nor, X_test_nor)
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
        model=self.model


        # quantile_qs=self.quantile_qs
        with torch.no_grad():
            # all_predictions_list=[]
            model.eval()
            all_predictions=[]
            for x_batch, y_batch in dataloader_test:
                x_batch = x_batch.to(device)
                # y_batch = y_batch.to(device)                

                test_pred1= model(x_batch)
     
                all_predictions.append(test_pred1.clone().detach())
                # all_targets.append(y_batch.clone().detach())

            all_predictions = torch.cat(all_predictions, dim=0)
                # all_targets = torch.cat(all_targets, dim=0)
                # all_targets=all_targets.cpu()
            all_predictions=all_predictions.cpu()
            all_predictions_np = all_predictions.numpy()
     

            all_predictions_np1=all_predictions_np.copy()

                # all_targets_np1=all_targets_np*y_std+y_mean

                # all_predictions_list.append(all_predictions_np1)
    
        upper = np.zeros((len(x),len(self.quantiles) - 1))
        lower = np.zeros((len(x),len(self.quantiles) - 1))
        y = all_predictions_np1
        zhixin=self.zhixin
        for i in range(len(self.fenwei)):
            if i == 0:
                y_middle = y[:,i] * y_std + y_mean
            elif i <= len(zhixin)-1:
                upper[:, (i - 1) % len(self.quantiles)] = y[:,i] * y_std + y_mean
            else:
                lower[:, (i - len(self.zhixin)) % len(self.quantiles)] = y[:,i] * y_std + y_mean
                
        return y_middle, upper, lower