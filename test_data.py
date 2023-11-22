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
    num_labels = 6
    testdatafile = 'simu_10000_0.1_141_178_test.npy'
    testdata_set = load_data_file(testdatafile)
    testdata = testdata_set[:, 0:-num_labels]
    testlabels = testdata_set[:, -num_labels:]
    test_data_matrix = preparedata(testdata, testlabels)

    # show_heatmap(data_matrix, 'train corr')
    # show_heatmap(test_data_matrix, 'test corr')
    

    test_setx = test_data_matrix[:, 2:]
    test_setyS = test_data_matrix[:, 1]
    test_setyD = test_data_matrix[:, 0]

    model = RegressionNN_S()
    ckpt = r'split S.pth'
    test(model, testdata,test_setyS, 'split S',ckpt=ckpt)

    model2 = RegressionNN_D()
    ckpt2 = r'split D.pth'
    test(model, testdata,test_setyS, 'split D',ckpt=ckpt2)

    # methodlist = ['linear','forest_reg','tree_reg', 'gboost', 'mlp']
    # Models.usemodel(train_setx, train_setyD, test_setx, test_setyD, methodlist[1])
