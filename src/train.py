from gpu_manager import GPUManager
from engine import DefaultClassificationEngine
from meter import *
from model import LeNet
import torch
from torch.autograd import Variable
import torch.optim as optim
from collections import defaultdict
import time
from pprint import pprint
import pickle

from multiconfig import get_all_configs

def on_sample_hook(state):
    inputs, targets = state["sample"]
    if state["engine"].cudable:
        inputs, targets = inputs.cuda(), targets.cuda()
    inputs, targets = Variable(inputs), Variable(targets)
    state.update(dict(sample=(inputs, targets)))


def on_start_hook(state):
    if state["train"]:
        state["network"].train()
    else:
        state["network"].eval()
    state.update(dict(loss_meter=LossMeter(), acc_meter=AccuracyMeter()))


def on_forward_hook(state):
    state["loss_meter"].add(state["loss"])
    _, targets = state['sample']
    state["acc_meter"].add(targets, state["output"])
    if state['t'] % 200 == 0:
        print(
            f"EPOCH={state.get('epoch','VAL')} ITER={state['t']} LOSS={state['loss_meter'].value():.4f} ACCURACY={100*state['acc_meter'].value():.4f}%")

def main(config):
    GPUManager.auto_chooce()
    engine = DefaultClassificationEngine()
    engine.hooks.update(
        dict(
            on_sample=on_sample_hook,
            on_start=on_start_hook,
            on_forward=on_forward_hook,
        )
    )
    net = LeNet(n_channels=config.n_channels, size=config.size)
    print(net)
    if torch.cuda.is_available():
        net = net.cuda()
    train_loader = torch.utils.data.DataLoader(
        config.dataset(train=True),
        batch_size=config.batch_size, shuffle=True, num_workers=8
    )
    val_loader = torch.utils.data.DataLoader(
        config.dataset(train=False),
        batch_size=1000, shuffle=True, num_workers=8)
    optimizer = optim.SGD(net.parameters(), lr=config.lr, momentum=0.9)
    recorder = defaultdict(list)
    recorder.update(
        dict(
            dataset_name=config.dataset.__name__.split("_")[0],
            lr=config.lr,
            batch_size=config.batch_size
        )
    )
    pprint(recorder)
    for epoch in range(config.maxepoch):
        state=engine.train(network=net, iterator=train_loader, maxepoch=1, optimizer=optimizer)
        recorder["train_loss"].append(state["loss_meter"].value())
        recorder["train_acc"].append(state["acc_meter"].value())
        state=engine.validate(network=net, iterator=val_loader)
        recorder["val_loss"].append(state["loss_meter"].value())
        recorder["val_acc"].append(state["acc_meter"].value())
    filename = f"{recorder['dataset_name']}_" + time.strftime("%Y%m%d_%H%M%S", time.localtime())
    with open(f"../result/{filename}.static", "wb") as f:
        pickle.dump(recorder, f)

if __name__ == '__main__':
    configs = get_all_configs()
    main(configs[5])












