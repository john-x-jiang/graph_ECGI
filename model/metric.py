import scipy.stats as stats
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal


def mse(output, target):
    mse = F.mse_loss(output, target, reduction='none')
    return mse

def tcc(u, x):
    m, n, w = u.shape
    res = []
    for i in range(m):
        correlation_sum = 0
        count = 0
        for j in range(n):
            a = u[i, j, :]
            b = x[i, j, :]
            if (a == a[0]).all() or (b == b[0]).all():
                count += 1
                continue
            correlation_sum = correlation_sum + stats.pearsonr(a, b)[0]
        correlation_sum = correlation_sum / (n - count)
        res.append(correlation_sum)
    res = np.array(res)
    return res


def scc(u, x):
    m, n, w = u.shape
    res = []
    for i in range(m):
        correlation_sum = 0
        count = 0
        for j in range(w):
            a = u[i, :, j]
            b = x[i, :, j]
            if (a == a[0]).all() or (b == b[0]).all():
                count += 1
                continue
            correlation_sum = correlation_sum + stats.pearsonr(a, b)[0]
        correlation_sum = correlation_sum / (w - count)
        res.append(correlation_sum)
    res = np.array(res)
    return res
