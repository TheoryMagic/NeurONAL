### Prerequisites:

python 3.7.3

CUDA 11.2

torch 1.8.0

torchvision 0.9.0

numpy 1.16.2


### Methods

alps.py     ALPS [Desalvo et al. 2021]

margin.py   Margin

neual_ntk.py  NeuralAL-NTK-Dynamic [Wang et al. 2021]

neuronal_pool.py    Our method

neuronal_stream.py    Our method

i_neural.py     I-NeurAL [Ban et al. 2022]



### Other files

dataset/    Datasets

create_folders  File to create the necessary directories needed for execution

load_data_addon.py  Load datasets

load_data.py    Load datasets

plot_stream.ipynb     Plots regret graphs from NeurONAL-Stream

plot_pool.ipynb    Plots testing accuracy graphs from NeurONAL-Pool

run.py      Runs the methods

utils.py    Load datasets


### Usage
1) Create all the folders using `source create_folders`

2) Run `py run.py` (`-h` shows options for arguments) to run a method
