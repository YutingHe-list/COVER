import csv
import math
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)

class LogWriter(object):
    def __init__(self, name, head):
        self.name = name+'.csv'
        with open(self.name, 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(head)
            f.close()

    def writeLog(self, dict):
        with open(self.name, 'a', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(dict)
            f.close()

def dice(pre, gt):
    tmp = pre + gt
    a = np.sum(np.where(tmp == 2, 1, 0))
    b = np.sum(pre)
    c = np.sum(gt)
    dice = (2*a)/(b+c+1e-6)
    return dice

def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((num_classes, n))
    categorical[y, np.arange(n)] = 1
    output_shape = (num_classes,) + input_shape
    categorical = np.reshape(categorical, output_shape)
    return categorical

