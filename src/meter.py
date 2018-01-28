import time

import torch
from torch.autograd import Variable


class Meter(object):
    def reset(self):
        pass

    def add(self):
        pass

    def value(self):
        pass


class TimeMeter(Meter):
    def __init__(self, unit):
        super(TimeMeter, self).__init__()
        self.unit = unit
        self.reset()

    def reset(self):
        self.n = 0
        self.time = time.time()

    def value(self):
        return time.time() - self.time


class AccuracyMeter(Meter):
    def __init__(self,top=1):
        super(AccuracyMeter, self).__init__()

        self.reset()

    def reset(self):
        self.total = 0
        self.correct = 0

    def add(self, targets, outputs):
        targets = targets.data if isinstance(targets, Variable) else targets
        outputs = outputs.data if isinstance(outputs, Variable) else outputs

        _, predicted = torch.max(outputs, 1)
        self.total += targets.numel()
        self.correct += predicted.eq(targets).cpu().sum()

    def value(self):
        return self.correct / self.total


class LossMeter(Meter):
    def __init__(self):
        super(LossMeter, self).__init__()
        self.reset()

    def reset(self):
        self.n = 0
        self.loss = 0.

    def add(self, loss):
        if isinstance(loss, Variable):
            loss = loss.data
        if torch.is_tensor(loss):
            self.n += 1
            self.loss += loss[0]
        else:
            raise Exception("Bad loss.")

    def value(self):
        return self.loss / self.n
