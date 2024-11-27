from margin import run as run_margin
from i_neural import run as run_ineural
from neuronal_stream import run as run_stream
from neuronal_pool import run as run_pool
from neual_ntk import run as run_ntk
from alps import run as run_alps
import argparse
import wandb  # 添加 wandb

def run(i, args):
    method = args.method  # 确保 method 在函数内可用
    f_name = ''
    if method == "a":
        print(f"ALPS on {datasets[i]}")
        inf_time, train_time, test_inf_time = run_alps(n=num_rounds, budget=budget, num_epochs=num_epochs, dataset_name=datasets[i], begin=begin[i])
        
        f_name = 'runtimes_alps.txt'
        
        # 记录到 wandb
        wandb.log({
            "Method": "ALPS",
            "Dataset": datasets[i],
            "Inference Time": inf_time,
            "Training Time": train_time,
            "Test Inference Time": test_inf_time
        })
    
    elif method == "d":
        print(f"NeuAL-NTK on {datasets[i]}")
        inf_time, train_time, test_inf_time = run_ntk(n=num_rounds, budget=budget, num_epochs=num_epochs, dataset_name=datasets[i], begin=begin[i], explore_size=in_es[i])
        
        f_name = 'runtimes_ntk.txt'
        
        # 记录到 wandb
        wandb.log({
            "Method": "NeuAL-NTK",
            "Dataset": datasets[i],
            "Inference Time": inf_time,
            "Training Time": train_time,
            "Test Inference Time": test_inf_time
        })

    elif method == 'm':
        print(f"Margin on {datasets[i]}")
        inf_time, train_time, test_inf_time = run_margin(n=num_rounds, margin=6, budget=budget, num_epochs=num_epochs, dataset_name=datasets[i], begin=begin[i], explore_size=in_es[i])
        
        f_name = 'runtimes_margin.txt'
        
        # 记录到 wandb
        wandb.log({
            "Method": "Margin",
            "Dataset": datasets[i],
            "Inference Time": inf_time,
            "Training Time": train_time,
            "Test Inference Time": test_inf_time
        })
    
    elif method == 'i':
        print(f"I-NeurAL on {datasets[i]}")
        inf_time, train_time, test_inf_time = run_ineural(n=num_rounds, margin=6, num_labels=budget, num_epochs=num_epochs, dataset_name=datasets[i], explore_size=in_es[i], begin=begin[i])
        
        f_name = 'runtimes_ineural.txt'
        
        # 记录到 wandb
        wandb.log({
            "Method": "I-NeurAL",
            "Dataset": datasets[i],
            "Inference Time": inf_time,
            "Training Time": train_time,
            "Test Inference Time": test_inf_time
        })
    
    elif method == 's':
        print(f"NeurONAL-Stream on {datasets[i]}")
        inf_time, train_time, test_inf_time = run_stream(n=num_rounds, margin=6, budget=budget, num_epochs=num_epochs, dataset_name=datasets[i], explore_size=npg_es[i], begin=begin[i])
        
        f_name = 'runtimes_neuronal.txt'
        
        # 记录到 wandb
        wandb.log({
            "Method": "NeurONAL-Stream",
            "Dataset": datasets[i],
            "Inference Time": inf_time,
            "Training Time": train_time,
            "Test Inference Time": test_inf_time
        })
    
    elif method == 'p':
        print(f'NeurONAL-Pool on {datasets[i]}')
        inf_time, train_time, test_inf_time = run_pool(dev=args.dev, n=num_rounds, margin=6, budget=budget, num_epochs=num_epochs, dataset_name=datasets[i], explore_size=npg_es[i], begin=begin[i], j=int(args.j))

        f_name = 'runtimes_batch_neuronal.txt'
        
        # 记录到 wandb
        wandb.log({
            "Method": "NeurONAL-Pool",
            "Dataset": datasets[i],
            "Inference Time": inf_time,
            "Training Time": train_time,
            "Test Inference Time": test_inf_time
        })
    
    with open(f_name, 'a') as f:
        f.write(f'{num_rounds}, {datasets[i]}, {inf_time}, {train_time}, {test_inf_time}\n')

num_rounds = 10000
datasets = ['letter', 'covertype', 'MagicTelescope', 'shuttle', 'adult', 'fashion']
in_es = [3079, 1350, 44, 128, 416, 15377]
npg_es = [1544, 208, 26, 34, 212, 1560]
begin = [0, 1, 1, 1, 1, 0]

argparser = argparse.ArgumentParser()
argparser.add_argument('--b', help='budget percentage', default='0.3')
argparser.add_argument('--ne', help='number of epochs', default='128')
argparser.add_argument('--method', help='\'a\' for ALPS, \'d\' for NeuAL-NTK, \'m\' for Margin, \'i\' for I-NeurAL and \'s\' for NeurONAL-Stream, \'p\' for NeurONAL-Pool', default='a')
argparser.add_argument('--dataset', help='-1 for all, 0-5 for Letter, Covertype, MT, Shuttle, Adult, or Fashion', default=0)
argparser.add_argument('--j', help='Last checkpoint number saved', default=0)
argparser.add_argument('--dev', help='GPU device number', default='3')
args = argparser.parse_args()
budget = float(args.b)
num_epochs = int(args.ne)
method = args.method
i = int(args.dataset)

# 初始化 wandb
wandb.init(
    project="NeurONAL_Experiments",  # 替换为你的 wandb 项目名称
    config=vars(args),
    reinit=True
)

if i >= 0:
    run(i, args)
else:
    for i in range(len(datasets)):
        run(i, args)

# 完成 wandb 运行
wandb.finish()
