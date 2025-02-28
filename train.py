import seaborn as sns
import scipy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import copy
from torch.utils.data import Dataset
import torch.optim as optim
import math
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import date
from datetime import datetime
import holidays
from tqdm import tqdm_notebook as tqdm
from torch.utils.data import Dataset,DataLoader,TensorDataset

import os


torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


def train_and_test(model, train_loader, epochs=70, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx,(X_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad() 
            X_batch=X_batch.to(device)
            y_batch=y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test)
        test_loss = criterion(y_pred_test, y_test)
        print(f"Test Loss: {test_loss.item():.4f}")





# Prediction function
def predict_power_load(new_data):
    new_data = scaler_X.transform(new_data)  # Standardise les inputs
    new_data_tensor = torch.tensor(new_data, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        predicted_load = model(new_data_tensor)
    return scaler_y.inverse_transform(predicted_load.numpy())







class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128, 64], output_dim=1, dropout=0.20):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        first=True
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))  # Normalization
            layers.append(nn.LeakyReLU())
            if (not first):
                layers.append(nn.Dropout(dropout))  # Regularization
            first=False
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))  # Output layer
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def create_dataloader(X_train, y_train, batch_size=1024):
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu', generator=None)
        m.bias.data.fill_(0.01)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    
    print(os.listdir('interpolatedData'))
    write_path = 'solution_template/pred.csv'
    prediction_df = pd.read_csv(write_path)
    dropColumns= ["date","altitude","year"]
    first=True
    for fichier in os.listdir('interpolatedData'):
        if ".ipynb_checkpoints" not in fichier:
            nom_region = fichier[:-4]
            interpolatedReg = pd.read_csv('interpolatedData/'+fichier,index_col=0)
            interpolatedReg['date'] = pd.to_datetime(interpolatedReg['date'])
            interpolatedReg = interpolatedReg.sort_values("date")
        
            split_date ='2021-12-31 22:30:00'
            known_set = interpolatedReg.loc[interpolatedReg['date'] <= split_date].drop(columns=dropColumns)
            known_set.dropna(axis=0,how='any',inplace=True)
            unknown_set =interpolatedReg.loc[interpolatedReg['date'] > split_date].drop(columns=dropColumns)
            print("Training on : ", nom_region)
            print(fichier)
            # The last column is the target variable
            data = known_set.values
            X = data[:, :-1]  # Features
            y = data[:, -1].reshape(-1, 1)  # Target variable
            
            X_submit = unknown_set.values[:, :-1]
            
            # Standardize the dataset
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X = scaler_X.fit_transform(X)
            y = scaler_y.fit_transform(y)
            
    
            X_submit = scaler_X.transform(X_submit)
            
            # Convert to PyTorch tensors
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train, X_test, y_train, y_test = map(torch.tensor, (X_train, X_test, y_train, y_test))
            X_train, X_test = X_train.float(), X_test.float()
            y_train, y_test = y_train.float(), y_test.float()
        
            X_submit = torch.tensor(X_submit).float()
        
            
            train_loader = create_dataloader(X_train, y_train, batch_size=512)
            # Initialize model, loss function, and optimizer
            model = MLP(X_train.shape[1]).to(device)
            model.apply(init_weights)
    
            
            X_train =X_train.to(device)
            X_test=X_test.to(device)
            y_train=y_train.to(device)
            y_test=y_test.to(device)
            
            X_submit = X_submit.to(device)
            
        
            
            epochs = 70
            train_and_test(model,train_loader,epochs)
    
            torch.save(model, './models/'+nom_region)
        
        






