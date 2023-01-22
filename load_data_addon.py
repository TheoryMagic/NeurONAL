import pandas as pd
import pickle
import arff
from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle
class Bandit_multi:
    def __init__(self, name):
        # Fetch data
        if name == 'covertype':
            X, y = fetch_openml('covertype', version=3, return_X_y=True)
            X = pd.get_dummies(X)
            # print(X,y)
            # class: 1-7
            # avoid nan, set nan as -1
            X[np.isnan(X)] = - 1
            X = normalize(X)
        elif name == 'MagicTelescope':
            X, y = fetch_openml('MagicTelescope', version=1, return_X_y=True)
            # class: h, g
            # avoid nan, set nan as -1
            # print(X,y)
            unique_values = set(y.values)
            label_map = {value:i+1 for i,value in enumerate(unique_values)}
            y = y.map(label_map)
            X[np.isnan(X)] = - 1
            X = normalize(X)
        elif name == 'shuttle':
            X, y = fetch_openml('shuttle', version=1, return_X_y=True)
            # avoid nan, set nan as -1
            # print(X,y)
            X[np.isnan(X)] = - 1
            X = normalize(X)
        elif name == 'adult':
            X, y = fetch_openml('adult', version=2, return_X_y=True)
            X = pd.get_dummies(X)
            # avoid nan, set nan as -1
            # print(X,y)
            unique_values = set(y.values)
            label_map = {value:i+1 for i,value in enumerate(unique_values)}
            y = y.map(label_map)
            X[np.isnan(X)] = - 1
            X = normalize(X)
        elif name == 'mushroom':
            X, y = fetch_openml('mushroom', version=1, return_X_y=True)
            # print(X,y,X.info())
            X = pd.get_dummies(X)
            unique_values = set(y.values)
            label_map = {value:i+1 for i,value in enumerate(unique_values)}
            y = y.map(label_map)
            # avoid nan, set nan as -1
            X[np.isnan(X)] = - 1
            X = normalize(X)
        elif name == 'fashion':
            X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True)
            # print(X,y,X.info())
            # avoid nan, set nan as -1
            X[np.isnan(X)] = - 1
            X = normalize(X)
        elif name == 'phishing':
            file_path = './binary_data/{}.txt'.format(name)
            f = open(file_path, "r").readlines()
            n = len(f)
            m = 68
            X = np.zeros([n, 68])
            y = np.zeros([n])
            for i, line in enumerate(f):
                line = line.strip().split()
                lbl = int(line[0])
                if lbl != 0 and lbl != 1:
                    raise ValueError
                y[i] = lbl
                l = len(line)
                for item in range(1, l):
                    pos, value = line[item].split(':')
                    pos, value = int(pos), float(value)
                    X[i, pos-1] = value
        elif name == "letter":
            file_path = './dataset/binary_data/{}_binary_data.pt'.format(name)
            f = open(file_path, 'rb')
            data = pickle.load(f)
            X, y = data['X'], data['Y']   
        else:
            raise RuntimeError('Dataset does not exist')
        # Shuffle data
        self.X, self.y = shuffle(X, y)
        # generate one_hot coding:
        '''self.y_arm = np.array(self.y.values).astype(np.int)
        # cursor and other variables
        self.cursor = 0
        self.size = self.y.shape[0]
        self.n_arm = int(np.max(self.y_arm)/2+1)
        self.dim = self.X.shape[1]  + self.n_arm
        self.act_dim = self.X.shape[1]
        self.num_user = np.max(self.y_arm)+1
        print(self.dim)
        print(self.n_arm)'''

    def read_data_arff(file_path, dataset):
        data = arff.load(open(file_path, 'r'))
        data = data['data']
        n, m = len(data), len(data[0])
        X, Y = np.zeros([n, m-1]), np.zeros([n])
        if dataset == 'ijcnn':
            for i in range(n):
                entry = data[i]
                if float(entry[-1]) == -1:
                    Y[i] = 0
                elif float(entry[-1]) == 1:
                    Y[i] = 1
                else:
                    raise ValueError
                for j in range(m-1):
                    X[i, j] = float(entry[j])

        return X, Y

    def step(self):
        if self.cursor > (len(self.X)-1):
            self.cursor = 0
    
        x = self.X[self.cursor]
        y = self.y_arm[self.cursor]
        target = int(y.item()/2.0)
        X_n = []
        for i in range(self.n_arm):
            front = np.zeros((1*i))
            back = np.zeros((1*(self.n_arm - i)))
            new_d = np.concatenate((front, x, back), axis=0)
            X_n.append(new_d)
        X_n = np.array(X_n)    
        rwd = np.zeros(self.n_arm)
        rwd[target] = 1
        self.cursor += 1
        return X_n, rwd

    
class load_emnist_letter_1d:
    def __init__(self, is_shuffle=True):
        # Fetch data
        batch_size = 1
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        trainset = torchvision.datasets.EMNIST(root='./data', split = "letters", train=True,
                                        download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
        self.dataiter = iter(trainloader)

        self.n_arm = 26
        self.num_zeros = 10
        self.num_class = 26
        self.num_user = 26
        self.dim = 28*28 + self.num_zeros*(self.num_class - 1)

        
        
    def step(self):
        x, y = self.dataiter.next()
        d = x.numpy()[0][0].reshape(28*28)
        target = y.item()-1
        X_n = []
        for i in range(self.n_arm):
            front = np.zeros((self.num_zeros*i))
            back = np.zeros((self.num_zeros*(self.num_class - i-1)))
            new_d = np.concatenate((front,  d, back), axis=0)
            X_n.append(new_d)
        X_n = np.array(X_n)    
        rwd = np.zeros(self.n_arm)
        rwd[target] = 1
        return X_n, rwd
