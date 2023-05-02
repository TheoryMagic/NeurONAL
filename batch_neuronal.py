import os
import time
import random
import numpy as np
from numpy.random import choice
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from skimage.measure import block_reduce
from load_data_addon import Bandit_multi

class Network_exploitation(nn.Module):
    def __init__(self, dim, hidden_size=100, k=10):
        super(Network_exploitation, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, k)

    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))
    
    
class Network_exploration(nn.Module):
    def __init__(self, dim, hidden_size=100, k=10):
        super(Network_exploration, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, k)

    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_size=100, k=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, k)

    def forward(self, x):
        return torch.sigmoid(self.fc2(self.activate(self.fc1(x))))

def EE_forward(net1, net2, x):

    x.requires_grad = True
    f1 = net1(x)
    net1.zero_grad()
    f1.sum().backward(retain_graph=True)
    dc = torch.cat([p.grad.flatten().detach() for p in net1.parameters()])
    #dc = dc / torch.linalg.norm(dc)
    dc = block_reduce(dc.cpu(), block_size=51, func=np.mean)
    dc = torch.from_numpy(dc).to(x.device)
    f2 = net2(dc)
    return f1, f2, dc

def train_NN_batch(model, X, Y, num_epochs=10, lr=0.0001, batch_size=64):
    model.train()
    X = torch.cat(X).float()
    Y = torch.stack(Y).float().detach()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    num = X.size(1)

    index = np.arange(len(X))
    np.random.shuffle(index)
    index = index[:batch_size]
    for _ in range(num_epochs):
        batch_loss = 0.0
        for i in index:
            x, y = X[i].to(device), Y[i].to(device)
            y = torch.reshape(y, (1,-1))
            pred = model(x).view(-1)

            optimizer.zero_grad()
            loss = torch.mean((pred - y) ** 2)
            loss.backward()
            optimizer.step()

            batch_loss += loss.item()
        
        if batch_loss / num <= 1e-3:
            return batch_loss / num

    return batch_loss / num

def run(n=1000, margin=6, budget=0.05, num_epochs=10, dataset_name="covertype", explore_size=0, test=1, begin=0, lr=0.0001):
    data = Bandit_multi(dataset_name)
    X = data.X
    Y = data.y

    X = np.array(X)[:n]
    if len(X.shape) == 3:
        N, h, w = X.shape
        X = np.reshape(X, (N, h*w))[:n]
    Y = np.array(Y)
    Y = (Y.astype(np.int64) - begin)[:n]

    train_dataset = TensorDataset(torch.tensor(X.astype(np.float32)), torch.tensor(Y.astype(np.int64)))

    test_x = data.X[n:]
    test_y = data.y[n:]

    if len(test_x) > n:
        test_x = test_x[:n]
        test_y = test_y[:n]
    test_y = np.array(test_y)
    test_y = (test_y.astype(np.int64) - begin)
    
    test_dataset = TensorDataset(torch.tensor(test_x.astype(np.float32)), torch.tensor(test_y.astype(np.int64)))

    k = len(set(Y))
    net1 = Network_exploitation(X.shape[1], k=k).to(device)
    net2 = Network_exploration(explore_size, k=k).to(device)

    X1_train, X2_train, y1, y2 = [], [], [], []
    budget = int(n * budget)
    inf_time = 0
    train_time = 0
    test_inf_time = 0
    R = 3000
    batch_size = 1000
    num_epochs = 20

    mu = n
    gamma = 300

    
    queried_rows = []
    for j in range(R):
        weights = []
        indices = []
        for i in tqdm(range(n)):
            if i in queried_rows:
                continue
            # load data point
            try:
                x, y = train_dataset[i]
            except:
                break
            x = x.view(1, -1).to(device)

            # predict via NeurONAL
            temp = time.time()
            f1, f2, dc = EE_forward(net1, net2, x)
            inf_time = inf_time + time.time() - temp
            u = f1[0] + 1 / (i+1) * f2
            u_sort, u_ind = torch.sort(u)
            i_hat = u_sort[-1]
            i_deg = u_sort[-2]
            neuronal_pred = int(u_ind[-1].item())

            # calculate weight
            weight = abs(i_hat - i_deg).item()
            weights.append(weight)
            indices.append(i)
        

        # create the distribution and sample b points from it
        i_hat = np.argmin(weights)
        w_hat = weights[i_hat]
        distribution = [(w_hat / (mu * w_hat + gamma * (weights[x] - w_hat))) if x != i_hat else 0 for x in range(len(weights))]
        distribution[i_hat] = 1 - sum(distribution)
        #weights = [w/s for w in weights]

        # sample from distribution
        ind = choice(a=indices, size=1, p=distribution).item()
        x, y = train_dataset[ind]
        x = x.view(1, -1).to(device)

        temp = time.time()
        f1, f2, dc = EE_forward(net1, net2, x)
        inf_time = inf_time + time.time() - temp
        u = f1[0] + 1 / (i+1) * f2
        u_sort, u_ind = torch.sort(u)
        i_hat = u_sort[-1]
        i_deg = u_sort[-2]
        neuronal_pred = int(u_ind[-1].item())
        
        # add predicted rewards to the sets
        X1_train.append(x)
        X2_train.append(torch.reshape(dc, (1, len(dc))))
        r_1 = torch.zeros(k).to(device)
        r_1[y.item()] = 1
        y1.append(r_1) 
        y2.append((r_1 - f1)[0])

        # update unlabeled set
        queried_rows.append(ind)

        # update the model
        temp = time.time()
        train_NN_batch(net1, X1_train, y1, num_epochs=num_epochs, lr=lr)
        train_NN_batch(net2, X2_train, y2, num_epochs=num_epochs, lr=lr)
        train_time = train_time + time.time() - temp

        # calculate testing regret
        if j % 100 == 0:
            current_acc = 0
            for i in tqdm(range(len(test_dataset))):
                # load data point
                try:
                    x, y = test_dataset[i]
                except:
                    break
                x = x.view(1, -1).to(device)

                # predict via NeurONAL
                temp = time.time()
                f1, f2, dc = EE_forward(net1, net2, x)
                inf_time = inf_time + time.time() - temp
                u = f1[0] + 1 / (i+1) * f2
                u_sort, u_ind = torch.sort(u)
                i_hat = u_sort[-1]
                i_deg = u_sort[-2]
                neuronal_pred = int(u_ind[-1].item())

                lbl = y.item()
                if neuronal_pred == lbl:
                    current_acc += 1
            
            testing_acc = current_acc / len(test_dataset)
            
            print(f'testing acc for round {j}: {testing_acc}')
            f = open(f"results/{dataset_name}/batch_neuronal_res.txt", 'a')
            f.write(f'testing acc for round {j}: {testing_acc}\n')
            f.close()

    return inf_time, train_time, test_inf_time


device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
