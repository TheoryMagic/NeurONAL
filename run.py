from margin_k import run as run_mk
from i_neural import run as run_iklv
from neural_pg import run as run_pg
from neual_ntk_d import run as run_ntkd
from alps import run as run_alps
import sys
import time
import numpy as np
import argparse
from utils import get_data
from load_data import load_mnist_1d
from load_data_addon import Bandit_multi

num_rounds = 10
num_test = num_rounds
datasets = ['letter', 'covertype', 'MagicTelescope', 'shuttle', 'adult', 'fashion']
in_es = [3079, 1350, 44, 128, 416, 15377]
npg_es = [1544, 208, 26, 34, 212, 1560]
test = [0, num_rounds, 0, num_rounds, num_rounds, num_rounds]
begin = [0, 1, 1, 1, 1, 1]

argparser = argparse.ArgumentParser()
argparser.add_argument('--b', help='budget percentage', default='0.3')
argparser.add_argument('--ne', help='number of epochs', default='40')
argparser.add_argument('--method', help='\'a\' for ALPS, \'d\' for NeuAL-NTK-D, \'m\' for Margin, \'i\' for I-NeurAL and \'n\' for NeurALPG', default='n')

args = argparser.parse_args()

budget = float(args.b)
num_epochs = int(args.ne)
method = args.method

for i in range(len(datasets)):
    if method == "a":
        print(f"ALPS on {datasets[i]}")
        inf_time, train_time, test_inf_time = run_alps(n=num_rounds, budget=budget, num_epochs=num_epochs, dataset_name=datasets[i], test=test[i])
        
        f = open('runtimes_alps.txt', 'a')
    
    if method == "d":
        print(f"NeuAL-NTK-D on {datasets[i]}")
        inf_time, train_time, test_inf_time = run_ntkd(n=num_rounds, budget=budget, num_epochs=num_epochs, dataset_name=datasets[i], test=test[i])
        
        f = open('runtimes_ntkd.txt', 'a')

    if method == 'm':
        print(f"Margin on {datasets[i]}")
        inf_time, train_time, test_inf_time = run_mk(n=num_rounds, margin=6, budget=budget, num_epochs=num_epochs, dataset_name=datasets[i], test=test[i], begin=begin[i])
        
        f = open('runtimes_margin.txt', 'a')

    if method == 'i':
        print(f"I-NeurAL on {datasets[i]}")
        inf_time, train_time, test_inf_time = run_iklv(n=num_rounds, margin=6, budget=budget, num_epochs=num_epochs, dataset_name=datasets[i], explore_size=in_es[i], test=test[i], begin=begin[i])
        
        f = open('runtimes_ineural.txt', 'a')

    if method == 'n':
        print(f"NeurALPG on {datasets[i]}")
        inf_time, train_time, test_inf_time = run_pg(n=num_rounds, margin=6, budget=budget, num_epochs=num_epochs, dataset_name=datasets[i], explore_size=npg_es[i], test=test[i], begin=begin[i])
        
        f = open('runtimes_neuralpg.txt', 'a')
    
    f.write(f'{num_rounds}, {datasets[i]}, {inf_time}, {train_time}, {test_inf_time}\n')
    f.close()
    
