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


def evaluate_driver(model, data_loaders, metrics, hparams, exp_dir, data_tags):
    eval_config = hparams.evaluating
    loss_func = hparams.loss

    evaluate_epoch(model, data_loaders, metrics, exp_dir, hparams, data_tags, eval_config, loss_func=loss_func)


def evaluate_epoch(model, data_loaders, metrics, exp_dir, hparams, data_tags, eval_config, loss_func=None):
    torso_len = eval_config['torso_len']
    signal_source = eval_config['signal_source']
    omit = eval_config.get('omit')
    window = eval_config.get('window')
    k_shot = eval_config.get('k_shot')
    changable = eval_config.get('changable')
    data_scaler = eval_config.get('data_scaler')
    model.eval()
    n_data = 0
    total_time = 0
    mses = dict()
    tccs = dict()
    sccs = dict()

    with torch.no_grad():
        data_names = list(data_loaders.keys())
        for data_name in data_names:
            data_loader_tag = data_loaders[data_name]
            eval_tags = list(data_loader_tag.keys())

            for tag_idx, eval_tag in enumerate(eval_tags):
                recons, grnths, labels = None, None, None

                data_loader = data_loader_tag[eval_tag]
                len_epoch = len(data_loader)
                n_data += len_epoch * data_loader.batch_size
                
                for idx, data in enumerate(data_loader):
                    x, y, label = data.x, data.y, data.label
                    x = x.to(device)
                    y = y.to(device)
                    label = label.to(device)
                    
                    if window is not None:
                        x = x[:, :, :window]
                        y = y[:, :, :window]
                    
                    if omit is not None:
                        x = x[:, :, omit:]
                        y = y[:, :, omit:]
                    
                    if data_scaler is not None:
                        x = data_scaler * x
                        y = data_scaler * y

                    if signal_source == 'heart':
                        source = x
                        output = x
                    elif signal_source == 'torso':
                        source = y
                        output = x

                    physics_vars, statistic_vars = model(source, data_name)
                    x_ = physics_vars[0]

                    if idx == 0:
                        recons = tensor2np(x_)
                        grnths = tensor2np(output)
                        labels = tensor2np(label)
                    else:
                        recons = np.concatenate((recons, tensor2np(x_)), axis=0)
                        grnths = np.concatenate((grnths, tensor2np(output)), axis=0)
                        labels = np.concatenate((labels, tensor2np(label)), axis=0)

                    for met in metrics:
                        if met.__name__ == 'mse':
                            mse = met(x_, output)
                            mse = mse.mean([1, 2])
                            mse = tensor2np(mse)
                            if idx == 0:
                                mses['{}_{}'.format(data_name, eval_tag)] = mse
                            else:
                                mses['{}_{}'.format(data_name, eval_tag)] = np.concatenate(
                                    (mses['{}_{}'.format(data_name, eval_tag)], mse), 
                                    axis=0
                                    )
                        if met.__name__ == 'tcc':
                            if type(output) == torch.Tensor or type(x_) == torch.Tensor:
                                output = tensor2np(output)
                                x_ = tensor2np(x_)
                            tcc = met(x_, output)
                            if idx == 0:
                                tccs['{}_{}'.format(data_name, eval_tag)] = tcc
                            else:
                                tccs['{}_{}'.format(data_name, eval_tag)] = np.concatenate(
                                    (tccs['{}_{}'.format(data_name, eval_tag)], tcc), 
                                    axis=0
                                    )
                        if met.__name__ == 'scc':
                            if type(output) == torch.Tensor or type(x_) == torch.Tensor:
                                output = tensor2np(output)
                                x_ = tensor2np(x_)
                            scc = met(x_, output)
                            if idx == 0:
                                sccs['{}_{}'.format(data_name, eval_tag)] = scc
                            else:
                                sccs['{}_{}'.format(data_name, eval_tag)] = np.concatenate(
                                    (sccs['{}_{}'.format(data_name, eval_tag)], scc), 
                                    axis=0
                                    )

                if eval_tag in data_tags:
                    save_result(exp_dir, recons, grnths, labels, data_name, eval_tag)
    
    for met in metrics:
        if met.__name__ == 'mse':
            print_results(exp_dir, 'mse', mses)
        if met.__name__ == 'tcc':
            print_results(exp_dir, 'tcc', tccs)
        if met.__name__ == 'scc':
            print_results(exp_dir, 'scc', sccs)


def prediction_driver(model, spt_data_loaders, qry_data_loaders, metrics, hparams, exp_dir, data_tags):
    eval_config = hparams.evaluating
    loss_func = hparams.loss

    prediction_epoch(model, spt_data_loaders, qry_data_loaders, metrics, exp_dir, hparams, data_tags, eval_config, loss_func=loss_func)


def prediction_epoch(model, spt_data_loaders, qry_data_loaders, metrics, exp_dir, hparams, data_tags, eval_config, loss_func=None):
    torso_len = eval_config['torso_len']
    signal_source = eval_config['signal_source']
    omit = eval_config.get('omit')
    window = eval_config.get('window')
    k_shot = eval_config.get('k_shot')
    changable = eval_config.get('changable')
    data_scaler = eval_config.get('data_scaler')
    model.eval()
    n_data = 0
    total_time = 0
    mses = dict()
    tccs = dict()
    sccs = dict()

    with torch.no_grad():
        data_names = list(qry_data_loaders.keys())
        for data_name in data_names:
            qry_loader_tag = qry_data_loaders[data_name]
            spt_loader_tag = spt_data_loaders[data_name]

            qry_tags = list(qry_loader_tag.keys())
            spt_tags = list(spt_loader_tag.keys())
            
            for tag_idx, qry_tag in enumerate(qry_tags):
                recons, grnths, labels = None, None, None

                qry_data_loader = qry_loader_tag[qry_tag]
                spt_data_loader = spt_loader_tag[spt_tags[tag_idx]]
                len_epoch = len(qry_data_loader)
                n_data += len_epoch * qry_data_loader.batch_size
                
                data_iterator = iter(spt_data_loader)
                for idx, qry_data in enumerate(qry_data_loader):
                    try:
                        spt_data = next(data_iterator)
                    except StopIteration:
                        data_iterator = iter(spt_data_loader)
                        spt_data = next(data_iterator)

                    qry_x, qry_y, qry_label = qry_data.x, qry_data.y, qry_data.label
                    qry_x = qry_x.to(device)
                    qry_y = qry_y.to(device)
                    qry_label = qry_label.to(device)

                    if window is not None:
                        qry_x = qry_x[:, :, :window]
                        qry_y = qry_y[:, :, :window]
                    
                    if omit is not None:
                        qry_x = qry_x[:, :, omit:]
                        qry_y = qry_y[:, :, omit:]
                    
                    if data_scaler is not None:
                        qry_x = data_scaler * qry_x
                        qry_y = data_scaler * qry_y

                    if signal_source == 'heart':
                        qry_source = qry_x
                        qry_output = qry_x
                    elif signal_source == 'torso':
                        qry_source = qry_y
                        qry_output = qry_x

                    D_x = spt_data.D_x
                    D_y = spt_data.D_y
                    D_label = spt_data.D_label
                    D_x = D_x.to(device)
                    D_y = D_y.to(device)
                    D_label = D_label.to(device)

                    if window is not None:
                        D_x = D_x[:, :, :window]
                        D_y = D_y[:, :, :window]
                    
                    if omit is not None:
                        D_x = D_x[:, :, omit:]
                        D_y = D_y[:, :, omit:]
                    
                    if data_scaler is not None:
                        D_x = data_scaler * D_x
                        D_y = data_scaler * D_y

                    N, M1, T = qry_x.shape
                    N, M2, T = qry_y.shape
                    D_x = D_x.view(N, -1, M1, T)
                    D_y = D_y.view(N, -1, M2, T)

                    if signal_source == 'heart':
                        D_source = D_x
                    elif signal_source == 'torso':
                        D_source = D_y
                    
                    if changable:
                        K = D_source.shape[1]
                        sub_K = np.random.randint(low=1, high=K+1, size=1)[0]
                        D_source = D_source[:, :sub_K, :]
                        D_label = D_label[:, :sub_K, :]

                    physics_vars, statistic_vars = model.prediction(qry_source, qry_label, D_source, D_label, data_name)

                    x_ = physics_vars[0]

                    if idx == 0:
                        recons = tensor2np(x_)
                        grnths = tensor2np(qry_output)
                        labels = tensor2np(qry_label)
                    else:
                        recons = np.concatenate((recons, tensor2np(x_)), axis=0)
                        grnths = np.concatenate((grnths, tensor2np(qry_output)), axis=0)
                        labels = np.concatenate((labels, tensor2np(qry_label)), axis=0)

                    for met in metrics:
                        if met.__name__ == 'mse':
                            mse = met(x_, qry_output)
                            mse = mse.mean([1, 2])
                            mse = tensor2np(mse)
                            if idx == 0:
                                mses['{}_{}'.format(data_name, qry_tag)] = mse
                            else:
                                mses['{}_{}'.format(data_name, qry_tag)] = np.concatenate(
                                    (mses['{}_{}'.format(data_name, qry_tag)], mse), 
                                    axis=0
                                    )
                        if met.__name__ == 'tcc':
                            if type(qry_output) == torch.Tensor or type(x_) == torch.Tensor:
                                qry_output = tensor2np(qry_output)
                                x_ = tensor2np(x_)
                            tcc = met(x_, qry_output)
                            if idx == 0:
                                tccs['{}_{}'.format(data_name, qry_tag)] = tcc
                            else:
                                tccs['{}_{}'.format(data_name, qry_tag)] = np.concatenate(
                                    (tccs['{}_{}'.format(data_name, qry_tag)], tcc), 
                                    axis=0
                                    )
                        if met.__name__ == 'scc':
                            if type(qry_output) == torch.Tensor or type(x_) == torch.Tensor:
                                qry_output = tensor2np(qry_output)
                                x_ = tensor2np(x_)
                            scc = met(x_, qry_output)
                            if idx == 0:
                                sccs['{}_{}'.format(data_name, qry_tag)] = scc
                            else:
                                sccs['{}_{}'.format(data_name, qry_tag)] = np.concatenate(
                                    (sccs['{}_{}'.format(data_name, qry_tag)], scc), 
                                    axis=0
                                    )

                if qry_tag in data_tags:
                    save_result(exp_dir, recons, grnths, labels, data_name, qry_tag)

    for met in metrics:
        if met.__name__ == 'mse':
            print_results(exp_dir, 'mse', mses)
        if met.__name__ == 'tcc':
            print_results(exp_dir, 'tcc', tccs)
        if met.__name__ == 'scc':
            print_results(exp_dir, 'scc', sccs)


def print_results(exp_dir, met_name, mets):
    if not os.path.exists(exp_dir + '/data'):
        os.makedirs(exp_dir + '/data')
    
    data_names = list(mets.keys())
    met = []
    for data_name in data_names:
        if mets[data_name] is None:
            continue
        met.append(mets[data_name])
        print('{}: {} for full seq avg = {:05.5f}, std = {:05.5f}'.format(data_name, met_name, mets[data_name].mean(), mets[data_name].std()))
        with open(os.path.join(exp_dir, 'data/metric.txt'), 'a+') as f:
            f.write('{}: {} for full seq = {}\n'.format(data_name, met_name, mets[data_name].mean()))
    
    if len(met) == 0:
        return
    met = np.hstack(met)
    print('Summary: {} for full seq avg = {:05.5f}, std = {:05.5f}'.format(met_name, met.mean(), met.std()))
    with open(os.path.join(exp_dir, 'data/metric.txt'), 'a+') as f:
        f.write('Summary: {} for full seq avg = {}, std = {}\n'.format(met_name, met.mean(), met.std()))


def save_result(exp_dir, recons, grnths, labels, data_name, data_tag):
    if not os.path.exists(exp_dir + '/data'):
        os.makedirs(exp_dir + '/data')

    sio.savemat(
        os.path.join(exp_dir, 'data/{}_{}.mat'.format(data_name, data_tag)), 
        {'recons': recons, 'grnths': grnths, 'label': labels}
    )


def tensor2np(t):
    return t.cpu().detach().numpy()
