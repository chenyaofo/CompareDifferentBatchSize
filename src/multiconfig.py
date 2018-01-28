from datasets import *

class mnist_config:
    datasets=[mnist_dataset]
    n_channels=1
    size=28
    train_batch_sizes=[1,2,4,8,16,32,64,128]
    lr = 0.1

class cifar_config:
    datasets=[cifar10_dataset,cifar100_dataset]
    n_channels=3
    size=32
    train_batch_sizes=[1,2,4,8,16,32,64,128]
    lr = 0.1

configs = [mnist_config,cifar_config]

class Config(object):
    pass

def get_all_configs():
    rev = []
    for config in configs:
        for dataset in config.datasets:
            for batch_size in config.train_batch_sizes:
                c = Config()
                c.dataset = dataset
                c.n_channels = config.n_channels
                c.size=  config.size
                c.batch_size = batch_size
                c.lr = config.lr
                c.maxepoch=30
                rev.append(c)
    return rev

