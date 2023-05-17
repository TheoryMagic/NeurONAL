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
from load_data_addon import Bandit_multi

def neural_forward(model, x, Z):
    p = model(x)
    model.zero_grad()
    p.backward()
    g = torch.cat([p.grad.flatten().detach() for p in model.parameters()])
    sigma = gamma * g * g / Z
    sigma = torch.sqrt(torch.sum(sigma))
    u = p.item() + sigma.item()

    return u, g, sigma.item()

def train_NN_batch(model, X, y, num_epochs=10, lr=0.001, batch_size=64):
    model.train()
    X = torch.cat(X).float()
    y = torch.cat(y).float()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    num = X.size(0)

    for i in range(num_epochs):
        batch_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x).view(-1)

            loss = torch.mean((pred - y) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss += loss.item()
        
        if batch_loss / num <= 1e-3:
            return batch_loss / num

    return batch_loss / num


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_size=100):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

lam = 1.0
gamma = 0.1
device = 'cuda'
dataset_name = 'fashion'

def run(n=1000, budget=0.05, num_epochs=10, dataset_name='covertype'):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    lam = 1.0
    gamma = 0.1
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

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


    model = MLP(X.shape[1] * 2).to(device)
    regret = []
    X_train, reward = [], []
    budget = int(n * budget)
    current_regret = 0.0
    query_num = 0
    tf = time.time()
    inf_time = 0
    train_time = 0
    test_inf_time = 0

    total_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Z = lam * torch.ones((total_param)).to(device)
    ci = torch.zeros(1, X.shape[1]).to(device)

    for i in range(n):
        x, y = dataset[i]
        x = x.view(1, -1).to(device)
        x0 = torch.cat([x, ci], dim=1)
        x1 = torch.cat([ci, x], dim=1)

        temp = time.time()
        u0, g0, sigma0 = neural_forward(model, x0, Z)
        u1, g1, sigma1 = neural_forward(model, x1, Z)
        inf_time = inf_time + time.time() - temp

        ind = 0
        lt = 0
        if u0 > u1:
            pred = 0
            B = 2 * sigma0
            if abs(u0 - 0.5) <= B:
                ind = 1
        else:
            pred = 1
            B = 2 * sigma1
            if abs(u1 - 0.5) <= B:
                ind = 1

        lbl = y.item()
        if pred != lbl:
            current_regret += 1
            lt = 1

        if ind and (query_num < budget): 
            query_num += 1
            if pred == 0:
                Z += g0 * g0
                X_train.append(x0)
            else:
                Z += g1 * g1
                X_train.append(x1)

            reward.append(torch.Tensor([1-lt]))
            temp = time.time()
            train_NN_batch(model, X_train, reward, num_epochs=num_epochs)
            train_time = train_time + time.time() - temp
            
        regret.append(current_regret)

        print(f'{i},{query_num},{budget},{num_epochs},{current_regret}')
        f = open(f"results/{dataset_name}/neual_ntk_res.txt", 'a')
        f.write(f'{i},{query_num},{budget},{num_epochs},{current_regret}\n')
        f.close()
    
    print('-------TESTING-------')
    lim = min(n, len(dataset)-n)
    for _ in range(5):
        acc = 0
        for i in range(n, n+lim):
            ind = random.randint(n, len(dataset)-1)
            x, y = dataset[i]
            x = x.view(1, -1).to(device)
            x0 = torch.cat([x, ci], dim=1)
            x1 = torch.cat([ci, x], dim=1)

            temp = time.time()
            u0, g0, sigma0 = neural_forward(model, x0, Z)
            u1, g1, sigma1 = neural_forward(model, x1, Z)
            test_inf_time = test_inf_time + time.time() - temp

            if u0 > u1:
                pred = 0
            else:
                pred = 1

            lbl = y.item()
            if pred == lbl:
                acc += 1
        print(f'Testing accuracy: {acc/lim}\n')
        f = open(f"results/{dataset_name}/neual_ntk_res.txt", 'a')
        f.write(f'Testing accuracy: {acc/lim}\n')
        f.close()
    
    return inf_time, train_time, test_inf_time


