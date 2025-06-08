import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from extract_data import build_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DemoDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X)  
        self.Y = torch.from_numpy(Y)  

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class MLPPolicy(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=128, output_dim=7):
        super().__init__()
        self.net = nn.Sequential(

            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, output_dim),
        )
        
    def forward(self, x):
        return self.net(x)

def train_policy(npz_path, save_path="policy.pth",
                 batch_size=128, epochs=200, lr=1e-3):

    X, Y, mean_X, std_X = build_dataset(npz_path, normalize=True)
    dataset = DemoDataset(X, Y)


    N = len(dataset)
    n_val = int(0.1 * N)
    n_train = N - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)


    model = MLPPolicy(input_dim=10, hidden_dim=128, output_dim=7).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.MSELoss()

    #train
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for XB, YB in train_loader:
            XB, YB = XB.to(DEVICE), YB.to(DEVICE)
            pred = model(XB)
            loss = criterion(pred, YB)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * XB.shape[0]
        avg_train_loss = total_loss / n_train
        scheduler.step()


        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for XB, YB in val_loader:
                XB, YB = XB.to(DEVICE), YB.to(DEVICE)
                pred = model(XB)
                val_loss += criterion(pred, YB).item() * XB.shape[0]
        avg_val_loss = val_loss / n_val

        print(f"epoch {epoch:2d}  training loss = {avg_train_loss:.6f}  validation loss = {avg_val_loss:.6f}")

    # saves weights
    torch.save(model.state_dict(), save_path)
    print("check the path for the saved policy : ", save_path)
    return model

if __name__ == "__main__":
    train_policy("demos.npz", save_path="policy.pth",
                 batch_size=256, epochs=100, lr=1e-3)
