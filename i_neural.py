import os
import time
import math
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
from skimage.measure import block_reduce

class Network_exploitation(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super(Network_exploitation, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))
    
    
class Network_exploration(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super(Network_exploration, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))

def EE_forward(net1, net2, x):

    x.requires_grad = True
    f1 = net1(x)
    net1.zero_grad()
    f1.sum().backward(retain_graph=True)
    dc = torch.cat([p.grad.flatten().detach() for p in net1.parameters()])
    dc = dc / torch.linalg.norm(dc)
    dc = block_reduce(dc.cpu(), block_size=51, func=np.mean)
    dc = torch.from_numpy(dc).to(x.device)
    f2 = net2(dc)
    return f1, f2, dc

def train_NN_batch(model, X, Y, num_epochs=10, lr=0.001, batch_size=32):
    model.train()
    X = torch.cat(X).float()
    Y = torch.Tensor(Y).float().reshape(-1, 1)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    num = X.size(1)

    for i in range(num_epochs):
        batch_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            #y = torch.reshape(y, (1,-1))
            pred = model(x)#.view(-1)

            optimizer.zero_grad()
            loss = torch.mean((pred - y) ** 2)
            loss.backward()
            optimizer.step()

            batch_loss += loss.item()
        
        if batch_loss / num <= 1e-3:
            return batch_loss / num

    return batch_loss / num

def run(n=1000, margin=6, budget=0.05, num_epochs=10, dataset_name="covertype", explore_size=0, begin=0, lr=0.001):
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    data = Bandit_multi(dataset_name)
    X = data.X
    Y = data.y

    X = np.array(X)
    Y = np.array(Y)
    Y = Y.astype(np.int64) - begin
    k = len(set(Y))
     
    if len(X.shape) == 3:
        N, h, w = X.shape
        X = np.reshape(X, (N, h*w))
    dataset = TensorDataset(torch.tensor(X.astype(np.float32)), torch.tensor(Y.astype(np.int64)))

    hidden_size = 100 #default value
    regret = []
    net1 = Network_exploitation(X.shape[1] * k).to(device)
    net2 = Network_exploration(explore_size).to(device)
    X1_train, X2_train, y1, y2 = [], [], [], []
    budget = int(n * budget)
    current_regret = 0.0
    query_num = 0
    ci = torch.zeros(1, X.shape[1]).to(device)
    inf_time = 0
    train_time = 0
    test_inf_time = 0

    for i in range(n):
        try:
            x, y = dataset[i]
        except:
            break
        x = x.view(1, -1).to(device)

        # creating the long vectors
        arms = torch.zeros(k, k*x.shape[1]).to(device)
        for w in range(k):
            a = []
            for j in range(0,w):
                a.append(ci)
            a.append(x)
            for j in range(w+1,k):
                a.append(ci)
            arms[w]=torch.cat(a,dim=1)

        #should we query for a vector
        f1 = torch.zeros(k).to(device)
        f2 = torch.zeros(k).to(device)
        u = torch.zeros(k).to(device)
        dc = torch.zeros(k, explore_size).to(device)
        for j in range(k):
            temp = time.time()
            f1[j], f2[j], dc[j] = EE_forward(net1, net2, arms[j])
            inf_time = inf_time + time.time() - temp
            u[j] = f1[j] + 1 / (i+1) * f2[j]

        val, idx = u.sort()
        max_prob = torch.Tensor([val[-1], idx[-1]])
        second_max_prob = torch.Tensor([val[-2], idx[-2]])

        ind = 0
        if abs(max_prob[0].item() - second_max_prob[0].item()) < margin * 0.1:
            ind = 1


        #construct training set        
        pred = int(max_prob[1].item())

        lbl = y.item()
        if pred != lbl:
            current_regret += 1
            reward = 0 
        else:
            reward = 1

        if ind and (query_num < budget): 
            query_num += 1

            #add predicted rewards to the sets
            temp = torch.reshape(arms[pred], (1, arms[pred].shape[0]))
            X1_train.append(temp)
            temp = torch.reshape(dc[pred], (1, dc[pred].shape[0]))
            X2_train.append(temp)
            y1.append(reward)
            y2.append(reward - f1[lbl])

            temp = time.time()
            train_NN_batch(net1, X1_train, y1, num_epochs=num_epochs, lr=lr)
            train_NN_batch(net2, X2_train, y2, num_epochs=num_epochs, lr=lr)
            train_time = train_time + time.time() - temp
        regret.append(current_regret)
        print(f'{i},{query_num},{budget},{num_epochs},{current_regret}')
        f = open(f"results/{dataset_name}/ineural_res.txt", 'a')
        f.write(f'{i},{query_num},{budget},{num_epochs},{current_regret}\n')
        f.close()


    print('-------TESTING-------')
    lim = min(n, len(dataset)-n)
    for _ in range(5):
        acc = 0
        for i in range(n, n+lim):
            ind = random.randint(n, len(dataset)-1)
            x, y = dataset[ind]
            x = x.view(1, -1).to(device)

            # creating the long vectors
            arms = torch.zeros(k, k*x.shape[1]).to(device)
            for w in range(k):
                a = []
                for j in range(0,w):
                    a.append(ci)
                a.append(x)
                for j in range(w+1,k):
                    a.append(ci)
                arms[w]=torch.cat(a,dim=1)

            #inference
            f1 = torch.zeros(k).to(device)
            f2 = torch.zeros(k).to(device)
            u = torch.zeros(k).to(device)
            dc = torch.zeros(k, explore_size).to(device)
            for j in range(k):
                temp = time.time()
                f1[j], f2[j], dc[j] = EE_forward(net1, net2, arms[j])
                test_inf_time = test_inf_time + time.time() - temp
                u[j] = f1[j] + 1 / (i+1) * f2[j]

            val, idx = u.sort()
            max_prob = torch.Tensor([val[-1], idx[-1]])
            pred = int(max_prob[1].item())
            lbl = y.item()
            if pred == lbl:
                acc += 1
        print(f'Testing accuracy: {acc/lim}\n')        
        f = open(f"results/{dataset_name}/ineural_res.txt", 'a')
        f.write(f'Testing accuracy: {acc/lim}\n')
        f.close()

    return inf_time, train_time, test_inf_time


device = 'cuda'

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
