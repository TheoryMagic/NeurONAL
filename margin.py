import os
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from utils import get_data
from load_data import load_mnist_1d
from load_data_addon import Bandit_multi


def train_cls_batch(model, X, y, num_epochs=10, lr=0.0001, batch_size=64):
    model.train()
    X = torch.cat(X).float()
    y = torch.stack(y).float().detach() #torch.cat(y).float()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_fn = nn.BCELoss().to(device)
    num = X.size(0)

    for i in range(num_epochs):
        batch_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)

            loss = torch.mean((pred - y) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss += loss.item()
        
        if batch_loss / num <= 1e-3:
            return batch_loss / num

    return batch_loss / num

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_size=100, k=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, k)

    def forward(self, x):
        return torch.sigmoid(self.fc2(self.activate(self.fc1(x))))



def run(n=1000, margin=6, budget=0.05, num_epochs=10, dataset_name='covertype', begin=0, lr=0.001):
    data = Bandit_multi(dataset_name)
    X = data.X
    Y = data.y
    X = np.array(X)

    k = 10

    if len(X.shape) == 3:
        N, h, w = X.shape
        X = np.reshape(X, (N, h*w))
    Y = np.array(Y)
    dataset = TensorDataset(torch.tensor(X.astype(np.float32)), torch.tensor(Y.astype(np.int64)))

    model = MLP(X.shape[1], k=k).to(device)
    regret = []
    X_train, y_train = [], []
    budget = int(n * budget)
    current_regret = 0.0
    query_num = 0
    inf_time = 0
    train_time = 0
    test_inf_time = 0

    for i in range(n):
        model.eval()
        try:
            x, y = dataset[i]
        except:
            break
        x = x.view(1, -1).to(device)
        temp = time.time()
        k_prob = model(x)
        inf_time = inf_time + time.time() - temp
        max_prob = k_prob.max().item()
        pred = k_prob.argmax().item()
        lbl = y.item()

        if pred != lbl:
            current_regret += 1

        if (abs(max_prob - 0.1) < margin / 2) and (query_num < budget):
            r = torch.zeros(k).to(device)
            r[lbl] = 1
            X_train.append(x)
            y_train.append(r)
            temp = time.time()
            loss = train_cls_batch(model, X_train, y_train, num_epochs=num_epochs, lr=lr)
            train_time = train_time + time.time() - temp
            query_num += 1

        regret.append(current_regret)
        print(f'{i},{query_num},{budget},{num_epochs},{current_regret}')

        f = open(f"results/{dataset_name}/margin_res.txt", 'a')
        f.write(f'{i},{query_num},{budget},{num_epochs},{current_regret}\n')
        f.close()
        
    print('-------TESTING-------')
    lim = min(n, len(dataset)-n)
    for _ in range(5):
        acc = 0
        for i in range(n, n+lim):
            model.eval()
            ind = random.randint(n, len(dataset)-1)
            x, y = dataset[ind]
            x = x.view(1, -1).to(device)
            temp = time.time()
            k_prob = model(x)
            test_inf_time = test_inf_time + time.time() - temp
            max_prob = k_prob.max().item()
            pred = k_prob.argmax().item()
            lbl = y.item()
            if pred == lbl:
                acc += 1
        print(f'Testing accuracy: {acc/lim}\n')
        f = open(f"results/{dataset_name}/margin_res.txt", 'a')
        f.write(f'Testing accuracy: {acc/lim}\n')
        f.close()

    return inf_time, train_time, test_inf_time


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
