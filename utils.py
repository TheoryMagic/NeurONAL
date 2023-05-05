import arff
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import datetime
import os
import psutil
import codecs
from torch.nn.modules.loss import _Loss
from PIL import Image, ImageEnhance

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))

def set_seed(seed):
    if seed is None:
        seed = int(str(datetime.datetime.now().timestamp())[12:])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def getMyCpu():
    data = psutil.virtual_memory()
    total = data.total
    print('TotalMemory', total)
    free = data.available
    print('FreeMemory', free)

    memory = "Memory usage:%d" % (int(round(data.percent)))
    print(memory + "%")
    CPU = "CPU:%0.2f" % psutil.cpu_percent(interval=1)
    print(CPU + '%')

    RestMemory = 100 - int(round(data.percent))
    if RestMemory > 10:
        print("\033[0;34m%s\033[0m" % "Sufficient Memory")
    elif RestMemory <= 5:
        print("\033[0;31m%s\033[0m" % "AvailableMemory<=5%")
    else:
        print("\033[0;32m%s\033[0m" % "AvailableMemory<=10%")


class LogitLoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(LogitLoss, self).__init__(size_average, reduce, reduction)


    def forward(self, input, c_idx):
        input_ = F.softmax(input, dim=-1)
        output_loss = input_[:,c_idx].mean()
        return output_loss


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def open_maybe_compressed_file(path):
    """Return a file object that possibly decompresses 'path' on the fly.
       Decompression occurs when argument `path` is a string and ends with '.gz' or '.xz'.
    """
    if not isinstance(path, torch._six.string_classes):
        return path
    if path.endswith('.gz'):
        import gzip
        return gzip.open(path, 'rb')
    if path.endswith('.xz'):
        import lzma
        return lzma.open(path, 'rb')
    return open(path, 'rb')


def read_sn3_pascalvincent_tensor(path, strict=True):
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
       Argument may be a filename, compressed filename, or file object.
    """
    # typemap
    if not hasattr(read_sn3_pascalvincent_tensor, 'typemap'):
        read_sn3_pascalvincent_tensor.typemap = {
            8: (torch.uint8, np.uint8, np.uint8),
            9: (torch.int8, np.int8, np.int8),
            11: (torch.int16, np.dtype('>i2'), 'i2'),
            12: (torch.int32, np.dtype('>i4'), 'i4'),
            13: (torch.float32, np.dtype('>f4'), 'f4'),
            14: (torch.float64, np.dtype('>f8'), 'f8')}
    # read
    with open_maybe_compressed_file(path) as f:
        data = f.read()
    # parse
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert nd >= 1 and nd <= 3
    assert ty >= 8 and ty <= 14
    m = read_sn3_pascalvincent_tensor.typemap[ty]
    s = [get_int(data[4 * (i + 1): 4 * (i + 2)]) for i in range(nd)]
    parsed = np.frombuffer(data, dtype=m[1], offset=(4 * (nd + 1)))
    assert parsed.shape[0] == np.prod(s) or not strict
    return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)


def read_label_file(path):
    with open(path, 'rb') as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 1)
    return x.long()


def read_image_file(path):
    with open(path, 'rb') as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 3)
    return x


enhancers = {
    0: lambda image, f: ImageEnhance.Color(image).enhance(f),
    1: lambda image, f: ImageEnhance.Contrast(image).enhance(f),
    2: lambda image, f: ImageEnhance.Brightness(image).enhance(f),
    3: lambda image, f: ImageEnhance.Sharpness(image).enhance(f)
}


factors = {
    0: lambda: np.random.normal(1.0, 0.3),
    1: lambda: np.random.normal(1.0, 0.1),
    2: lambda: np.random.normal(1.0, 0.1),
    3: lambda: np.random.normal(1.0, 0.3),
}


def enhance(image):
    order = [0, 1, 2, 3]
    np.random.shuffle(order)
    for i in order:
        f = factors[i]()
        image = enhancers[i](image, f)
    return image


def update_dataset(train_data, pool_data, queried_idxs, remain_idxs):
    queried_idxs_set = set(list(pool_data[2][queried_idxs]))
    trained_idxs_set = set(list(train_data[2]))
    if len(queried_idxs_set.intersection(trained_idxs_set)) != 0:
        print('Error at queried_idxs_set!!')

    if isinstance(train_data[0], list):
        train_data_ = (train_data[0] + [pool_data[0][q] for q in queried_idxs],
                       np.concatenate((train_data[1], pool_data[1][queried_idxs])),
                       np.concatenate((train_data[2], pool_data[2][queried_idxs])))
    else:
        train_data_ = (np.vstack((train_data[0], pool_data[0][queried_idxs])),
                       np.concatenate((train_data[1], pool_data[1][queried_idxs])),
                       np.concatenate((train_data[2], pool_data[2][queried_idxs])))

    if isinstance(pool_data[0], list):
        pool_data_ = ([pool_data[0][r] for r in remain_idxs], pool_data[1][remain_idxs], pool_data[2][remain_idxs])
    else:
        pool_data_ = (pool_data[0][remain_idxs], pool_data[1][remain_idxs], pool_data[2][remain_idxs])

    return train_data_, pool_data_

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

def read_data_txt(file_path, dataset):
    f = open(file_path, "r").readlines()
    n = len(f)
    if dataset == 'phishing':
        m = 68
        X = np.zeros([n, 68])
        Y = np.zeros([n])
        for i, line in enumerate(f):
            line = line.strip().split()
            lbl = int(line[0])
            if lbl != 0 and lbl != 1:
                raise ValueError
            Y[i] = lbl
            l = len(line)
            for item in range(1, l):
                pos, value = line[item].split(':')
                pos, value = int(pos), float(value)
                X[i, pos-1] = value

    return X, Y

def load_data(dataset):
    if dataset in ['ijcnn']:
        file_path = './dataset/binary_data/{}.arff'.format(dataset)
        return read_data_arff(file_path, dataset)

    elif dataset in ['phishing']:
        file_path = './dataset/binary_data/{}.txt'.format(dataset)
        return read_data_txt(file_path, dataset)

    elif dataset in ['letter', 'fashion']:
        file_path = './dataset/binary_data/{}_binary_data.pt'.format(dataset)
        f = open(file_path, 'rb')
        data = pickle.load(f)
        X, Y = data['X'], data['Y']
        return X, Y

    elif dataset in ['mnist']:
        file_path = './dataset/MNIST_data/MNIST_binary_data.pt'
        f = open(file_path, 'rb')
        data = pickle.load(f)
        X, Y = data['X'], data['Y']
        return X, Y

    elif dataset in ['cifar']:
        file_path = './dataset/CIFAR10_data/CIFAR10_binary_data.pt'
        f = open(file_path, 'rb')
        data = pickle.load(f)
        X, Y = data['X'], data['Y']
        return X, Y

def get_data(dataset_name):
    X, Y = load_data(dataset_name)
    index = np.arange(X.shape[0])
    np.random.shuffle(index)
    if X.shape[0] > 12000:
        index = index[:12000]
    print(index[:5])
    X = X[index, :]
    Y = Y[index]
    return X, Y

def get_pretrain(dataset_name, num_sample=360, X=0, Y=0):
    index = np.arange(X.shape[0])
    np.random.seed(42)
    np.random.shuffle(index)
    if X.shape[0] > 12000:
        index = index[:12000]

    X = X[index, :]
    Y = Y[index]
    n = X.shape[0]

    pre_X, pre_Y = np.zeros([num_sample, X.shape[1]]), np.zeros([num_sample])
    num = 0
    for i in range(n):
        q = random.random()
        if q > 0.9 and num < num_sample:
            pre_X[num, :] = X[i, :]
            pre_Y[num] = Y[i]
            num += 1

    return pre_X, pre_Y

if __name__ == "__main__":
    dataset = 'mnist'

    random.seed(42)
    np.random.seed(42)
    X, Y = get_data(dataset)
    print(X.shape, Y.shape)



