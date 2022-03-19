import os
import sys
import random
import numpy as np

import scipy.io as sio
import scipy.stats as stats
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import util

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_driver(model, data_loaders, metrics, hparams, exp_dir, data_tag):
    eval_config = hparams.evaluating
    loss_type = hparams.loss

    evaluate_epoch(model, data_loaders, metrics, exp_dir, hparams, data_tag, eval_config, loss_type=loss_type)


def evaluate_epoch(model, data_loaders, metrics, exp_dir, hparams, data_tag, eval_config, loss_type=None):
    torso_len = eval_config['torso_len']
    signal_source = eval_config['signal_source']
    model.eval()
    n_steps = 0
    mses = {}
    tccs = {}
    sccs = {}

    q_recons = {}
    all_xs = {}
    all_labels = {}

    with torch.no_grad():
        data_names = list(data_loaders.keys())
        for data_name in data_names:
            data_loader = data_loaders[data_name]
            len_epoch = len(data_loader)
            for idx, data in enumerate(data_loader):
                signal, label = data.x, data.y
                signal = signal.to(device)
                label = label.to(device)

                x = signal[:, :-torso_len, :]
                y = signal[:, -torso_len:, :]

                if signal_source == 'heart':
                    source = x
                elif signal_source == 'torso':
                    source = y

                physics_vars, statistic_vars = model(source, data_name)
                
                if loss_type == 'dmm_loss':
                    x_q, x_p = physics_vars
                    x_ = x_p
                elif loss_type == 'recon_loss' or loss_type == 'mse_loss':
                    x_, _ = physics_vars
                elif loss_func == 'elbo_loss':
                    x_, _ = physics_vars
                else:
                    raise NotImplemented

                if idx == 0:
                    q_recons[data_name] = tensor2np(x_)
                    all_xs[data_name] = tensor2np(x)
                    all_labels[data_name] = tensor2np(label)
                else:
                    q_recons[data_name] = np.concatenate((q_recons[data_name], tensor2np(x_)), axis=0)
                    all_xs[data_name] = np.concatenate((all_xs[data_name], tensor2np(x)), axis=0)
                    all_labels[data_name] = np.concatenate((all_labels[data_name], tensor2np(label)), axis=0)

                for met in metrics:
                    if met.__name__ == 'mse':
                        mse = met(x_, x)
                        mse = tensor2np(mse)
                        if idx == 0:
                            mses[data_name] = mse
                        else:
                            mses[data_name] = np.concatenate((mses[data_name], mse), axis=0)
                    if met.__name__ == 'tcc':
                        if type(x) == torch.Tensor or type(x_) == torch.Tensor:
                            x = tensor2np(x)
                            x_ = tensor2np(x_)
                        tcc = met(x_, x)
                        if idx == 0:
                            tccs[data_name] = tcc
                        else:
                            tccs[data_name] = np.concatenate((tccs[data_name], tcc), axis=0)
                    if met.__name__ == 'scc':
                        if type(x) == torch.Tensor or type(x_) == torch.Tensor:
                            x = tensor2np(x)
                            x_ = tensor2np(x_)
                        scc = met(x_, x)
                        if idx == 0:
                            sccs[data_name] = scc
                        else:
                            sccs[data_name] = np.concatenate((sccs[data_name], scc), axis=0)
    
    for met in metrics:
        if met.__name__ == 'mse':
            print_results(exp_dir, 'mse', mses)
        if met.__name__ == 'tcc':
            print_results(exp_dir, 'tcc', tccs)
        if met.__name__ == 'scc':
            print_results(exp_dir, 'scc', sccs)
    
    save_result(exp_dir, q_recons, all_xs, all_labels, data_tag)


def print_results(exp_dir, met_name, mets):
    if not os.path.exists(exp_dir + '/data'):
        os.makedirs(exp_dir + '/data')
    
    data_names = list(mets.keys())
    for data_name in data_names:
        print('{}: {} for full seq = {:05.5f}'.format(data_name, met_name, mets[data_name].mean()))
        with open(os.path.join(exp_dir, 'data/metric.txt'), 'a+') as f:
            f.write('{}: {} for full seq = {}\n'.format(data_name, met_name, mets[data_name].mean()))


def save_result(exp_dir, recons, all_xs, all_labels, data_tag):
    if not os.path.exists(exp_dir + '/data'):
        os.makedirs(exp_dir + '/data')

    data_names = list(recons.keys())
    for data_name in data_names:
        sio.savemat(
            os.path.join(exp_dir, 'data/{}_{}.mat'.format(data_name, data_tag)), 
            {'recons': recons[data_name], 'inps': all_xs[data_name], 'label': all_labels[data_name]}
        )


def tensor2np(t):
    return t.cpu().detach().numpy()