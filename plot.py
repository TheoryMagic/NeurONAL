import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

datasets = ['adult', 'letter', 'fashion', 'covertype', 'shuttle', 'MagicTelescope']
methods = ['neuronal', 'margin', 'ineural', 'neual_ntk', 'alps']
colors = ['red', 'blue', 'green', 'orange', 'purple']
input_dir = 'results/'
output_dir = 'graphs'
extension = '_res.txt'

col_names = ['rounds', 'query_num', 'bud_percent', 'num_epochs', 'regret']

def get_mean_std(arr):
    return np.mean(arr, axis=0), np.std(arr, axis=0)

for d in datasets:
    for i in range(len(methods)):
        f_name = input_dir + d + '/' + methods[i] + extension

        f = pd.read_csv(f_name, header=None, nrows=10000)
        f.columns = col_names

        #x = f['query_num']
        x = range(f['regret'].size)
        plt.plot(x, f['regret'], color=colors[i], linewidth=2.0, label=methods[i])

    plt.legend()
    plt.xlabel('Rounds')
    plt.ylabel('Regret')
    plt.title([f'{d}'])
    plt.savefig(f'{output_dir}/{d}_results.png')
    plt.figure().clear()
