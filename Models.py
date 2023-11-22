from utils import *
import seaborn as sns
from scipy.signal import find_peaks
import sys

sys.path.insert(1, './')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


class RegressionNN_D(nn.Module):
    def __init__(self):
        super(RegressionNN_D, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1000, 128),  
            nn.BatchNorm1d(128), 
            nn.ReLU(),
            nn.Linear(128, 64),  
            nn.BatchNorm1d(64), 
            nn.ReLU(),
            nn.Linear(64, 64),  
            nn.BatchNorm1d(64), 
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    

class RegressionNN_S(nn.Module):
    def __init__(self):
        super(RegressionNN_S, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1000, 128),  
            nn.BatchNorm1d(128), 
            nn.ReLU(),
            nn.Linear(128, 64),  
            nn.BatchNorm1d(64), 
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    
def usemodel(trainX, trainY, testX, testY, method):
    if method == 'linear':
        model = LinearRegression()
    elif method=='tree_reg':
        model = DecisionTreeRegressor()
    elif method == 'forest_reg':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif method == 'gboost':
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1,
                                                        random_state=42)
    else: #method == mlp
        model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    model.fit(trainX, trainY)
    checktrainy = model.predict(trainX)
    maetrain = mean_absolute_error(checktrainy, trainY)
    pred_y = model.predict(testX)
    mae = mean_absolute_error(pred_y, testY)
    print(method + ' maetrain', maetrain)
    print(method + ' maetest', mae)

    plot_2vectors(trainY, checktrainy, method + 'train')  # label predictioon
    plot_2vectors(testY, pred_y, method + 'test')