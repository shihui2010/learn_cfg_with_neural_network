# Learning Bounded Context-Free-Grammar via LSTM and the Transformer:Difference and Explanations (AAAI22')

This repo provides the code to reproduce the experiments in the papers

>Hui Shi, Sicun Gao, Yuandong Tian, Xinyun Chen, Jishen Zhao, Learning Bounded Context-Free-Grammar via LSTM and the Transformer:Difference and Explanations

## Dependency

The code need Pytorch library. (The code runs with Pytorch=1.9.0, but most Pytorch versions should work)

## Run the code

### For Canonical PDAs, generate the sequences by (for example) :

> cd dataset
> python dataset.py --name dyck --k 5 --m 10 --maxlen 20 --beam_size 1000

### For parsing SCAN, download the dataset first:
> cd dataset
> git clone https://github.com/brendenlake/SCAN

### To run the experiment 
For lstm: 

> python main.py --data_dir dataset/DYCK_k5_m10_ml20 --model lstm --n_layers 1 

For Transformer Encoder:

> python main.py --data_dir dataset/DYCK_k5_m10_ml20 --model trans --n_layers 1

For Transformer Decoder:

> python main.py --data_dir dataset/DYCK_k5_m10_ml20 --model trans --causal --n_layers 1

Adding --latent_factorization switches to latent factorization setting, while the default uses forced factorization. 
Also, adding --train_sigma to use the learnable loss weights described in the paper. while the default uses constant
loss weights (\sigma = 1)
