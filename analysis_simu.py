import warnings
import pywt
from utils import *

import sys

sys.path.insert(1, './')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
\

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


from Models import LSTM, RegressionNN_S, RegressionNN_D
import Models


def load_data_file(data_file):
    if data_file.endswith('.csv'):
        data_set = pd.read_csv(data_file).to_numpy()
    elif data_file.endswith('.npy'):
        data_set = np.load(data_file)
    return data_set


plt.rcParams['figure.figsize'] = [20, 8]  # Bigger images



def train(model, data,testdata,train_setyS, test_setyS,num_epochs, name):
    model = model.cuda()
    #model = LSTM().cuda()
    criterion = nn.L1Loss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    data = torch.tensor(data).float().cuda()
    train_setyS = torch.tensor(train_setyS.reshape(-1,1)).float().cuda()
    train_dataset = TensorDataset(data, train_setyS)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    testdata = torch.tensor(testdata).float().cuda()
    test_setyS = torch.tensor(test_setyS.reshape(-1,1)).float().cuda()
    test_dataset = TensorDataset(testdata, test_setyS)
    test_loader = DataLoader(test_dataset, batch_size=32)

    num_epochs = num_epochs
    for epoch in range(num_epochs):
        model.train()
        losssum=0
        index=0
        for inputs, targets in train_loader:
            index+=1
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losssum+=loss.item()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {losssum/index:.4f}')
        with torch.no_grad():
            test_outputs = model(testdata)
            test_loss = criterion(test_outputs, test_setyS)
            test_outputs2 = model(data)
            test_loss2 = criterion(test_outputs2, train_setyS)
            print(f'Test Loss  on test: {test_loss.item():.4f}')
    torch.save(model.state_dict(), name+'.pth')
    finaloutputstest = model(testdata)
    finaloutputstrain = model(data)
    plot_2vectors(test_setyS.detach().cpu().numpy(),finaloutputstest.detach().cpu().numpy(),'test '+name)
    plot_2vectors(train_setyS.detach().cpu().numpy(),finaloutputstrain.detach().cpu().numpy(),'train '+name)

def test(model, testdata,test_setyS, name, ckpt):
    model = model.cuda()
    model.load_state_dict(torch.load(ckpt))
    criterion = nn.L1Loss().cuda()

    testdata = torch.tensor(testdata).float().cuda()
    test_setyS = torch.tensor(test_setyS.reshape(-1,1)).float().cuda()
    test_dataset = TensorDataset(testdata, test_setyS)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model.eval()
    with torch.no_grad():
        test_outputs = model(testdata)
        test_loss = criterion(test_outputs, test_setyS)
        test_outputs2 = model(data)
        test_loss2 = criterion(test_outputs2, train_setyS)
        print(f'Test Loss  on test: {test_loss.item():.4f}')
    finaloutputstest = model(testdata)
    plot_2vectors(test_setyS.detach().cpu().numpy(),finaloutputstest.detach().cpu().numpy(),'test '+name)



if __name__ == '__main__':
    data_file = 'simu_20000_0.1_90_140_train.npy'
    num_labels = 6
    data_set = load_data_file(data_file)
    # data = data_set[:16000,0:-num_labels]
    # labels = data_set[:16000, -num_labels:]
    # testdata = data_set[16000:,0:-num_labels]
    # testlabels = data_set[16000:, -num_labels:]

    data = data_set[:, 0:-num_labels]
    labels = data_set[:, -num_labels:]

    testdatafile = 'simu_10000_0.1_141_178_test.npy'
    testdata_set = load_data_file(testdatafile)
    testdata = testdata_set[:, 0:-num_labels]
    testlabels = testdata_set[:, -num_labels:]

    data_matrix = preparedata(data, labels)
    test_data_matrix = preparedata(testdata, testlabels)

    # show_heatmap(data_matrix, 'train corr')
    # show_heatmap(test_data_matrix, 'test corr')
    

    train_setx = data_matrix[:, 2:]
    test_setx = test_data_matrix[:, 2:]
    train_setyS = data_matrix[:, 1]
    test_setyS = test_data_matrix[:, 1]
    train_setyD = data_matrix[:, 0]
    test_setyD = test_data_matrix[:, 0]

    model = RegressionNN_S()
    ckpt = r'split S.pth'
    train(model, data,testdata,train_setyS, test_setyS,num_epochs=100,name='split S')
    test(model, testdata,test_setyS, 'split S',ckpt=ckpt)

    model2 = RegressionNN_D()
    ckpt2 = r'split D.pth'
    train(model, data,testdata,train_setyS, test_setyS,num_epochs=100,name='split D')
    test(model, testdata,test_setyS, 'split D',ckpt=ckpt2)

    # methodlist = ['linear','forest_reg','tree_reg', 'gboost', 'mlp']
    # Models.usemodel(train_setx, train_setyD, test_setx, test_setyD, methodlist[1])
