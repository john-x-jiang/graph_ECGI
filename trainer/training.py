import os
import sys
import time
import random
import numpy as np

import scipy.io
import scipy.stats as stats
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import util

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_driver(model, checkpt, epoch_start, optimizer, lr_scheduler, \
    train_loaders, valid_loaders, loss, metrics, hparams, exp_dir):
    total_loss_t = []
    total_loss_v = []
    separate_loss_t = {}
    separate_loss_v = {}

    train_config = dict(hparams.training)
    monitor_mode, monitor_metric = train_config['monitor'].split()

    metric_err = None
    not_improved_count = 0

    if checkpt is not None:
        total_loss_t, total_loss_v = checkpt['train_loss'], checkpt['valid_loss']
        separate_loss_t = checkpt['separate_loss_t']
        separate_loss_v = checkpt['separate_loss_v']

        metric_err = checkpt.get('opt_metric_err')
        if metric_err is None:
            metric_err = checkpt[monitor_metric][-1]
        not_improved_count = checkpt['not_improved_count']

    for epoch in range(epoch_start, train_config['epochs'] + 1):
        ts = time.time()
        # train epoch
        loss_t_epoch, separate_loss_t_epoch = \
            train_epoch(model, epoch, loss, optimizer, train_loaders, hparams)
        
        # valid epoch
        loss_v_epoch, separate_loss_v_epoch = \
            valid_epoch(model, epoch, loss, valid_loaders, hparams)
        te = time.time()

        # Append epoch losses to arrays
        total_loss_t.append(loss_t_epoch)
        for loss_name, loss_val in separate_loss_t_epoch.items():
            if not separate_loss_t.get(loss_name):
                separate_loss_t[loss_name] = []
            separate_loss_t[loss_name].append(loss_val)

        total_loss_v.append(loss_v_epoch)
        for loss_name, loss_val in separate_loss_v_epoch.items():
            if not separate_loss_v.get(loss_name):
                separate_loss_v[loss_name] = []
            separate_loss_v[loss_name].append(loss_val)

        # Step LR
        if lr_scheduler is not None:
            lr_scheduler.step()
            last_lr = lr_scheduler._last_lr
        else:
            last_lr = optimizer.param_groups[0]['lr']
        
        # Generate the checkpoint for this current epoch
        log = {
            # Base parameters to reload
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'not_improved_count': not_improved_count,
            'opt_metric_err': metric_err,
            
            'train_loss': total_loss_t,
            'valid_loss': total_loss_v,
            
            'separate_loss_t': separate_loss_t,
            'separate_loss_v': separate_loss_v,
        }
        
        # Save the latest model
        torch.save(log, exp_dir + '/m_latest')

        # Save the model for every saving period
        if epoch % train_config['save_period'] == 0:
            torch.save(log, exp_dir + '/m_' + str(epoch))
        
        # Print and write out epoch logs
        summary_log = '[Epoch: {:04d}, Time: {:.4f}]'.format(epoch, (te - ts) / 60)
        
        metric_logs = 'train_loss: {:05.5f}'.format(loss_t_epoch)
        for loss_name, loss_val in separate_loss_t_epoch.items():
            metric_logs += ', {}: {:05.5f}'.format(loss_name, loss_val)
        metric_logs += '\nvalid_loss: {:05.5f}'.format(loss_v_epoch)
        for loss_name, loss_val in separate_loss_v_epoch.items():
            metric_logs += ', {}: {:05.5f}'.format(loss_name, loss_val)
        
        print(summary_log)
        print(metric_logs)
        with open(os.path.join(exp_dir, 'log.txt'), 'a+') as f:
            f.write(summary_log + '\n')
            f.write(metric_logs + '\n')
        
        # Check if current epoch is better than best so far
        if epoch == 1:
            metric_err = log[monitor_metric][-1]
        else:
            improved = (monitor_mode == 'min' and log[monitor_metric][-1] <= metric_err) or \
                       (monitor_mode == 'max' and log[monitor_metric][-1] >= metric_err)
            if improved:
                metric_err = log[monitor_metric][-1]
                log['opt_metric_err'] = metric_err
                torch.save(log, exp_dir + '/m_best')
                not_improved_count = 0
            else:
                not_improved_count += 1
            
            if not_improved_count > train_config['early_stop']:
                info = "Validation performance didn\'t improve for {} epochs. Training stops.".format(train_config['early_stop'])
                break
        
    # save & plot losses
    losses = {
        'total': [
            total_loss_t,
            total_loss_v
        ],
    }
    for loss_name in train_config['loss_to_plot']:
        losses[loss_name] = [separate_loss_t[loss_name], separate_loss_v[loss_name]]
    save_losses(exp_dir, train_config['epochs'], losses)


def train_epoch(model, epoch, loss, optimizer, data_loaders, hparams):
    model.train()
    train_config = dict(hparams.training)
    kl_args = train_config.get('kl_args')
    torso_len = train_config.get('torso_len')
    signal_source = train_config.get('signal_source')
    omit = train_config.get('omit')
    window = train_config.get('window')
    k_shot = train_config.get('k_shot')
    changable = train_config.get('changable')
    meta_dataset = train_config.get('meta_dataset')
    loss_type = train_config.get('loss_type')
    loss_func = hparams.loss
    total_loss = 0
    separate_losses = {}
    n_steps = 0
    batch_size = hparams.batch_size

    data_names = list(data_loaders.keys())
    random.shuffle(data_names)
    len_epoch = len(data_loaders[data_names[0]]) * len(data_names)
    for data_name in data_names:
        data_loader = data_loaders[data_name]
        if epoch > 1 and meta_dataset:
            data_loader = data_loader.next()
        for idx, data in enumerate(data_loader):
            signal, label = data.x, data.y
            signal = signal.to(device)
            label = label.to(device)

            if window is not None:
                signal = signal[:, :, :window]

            x_heart = signal[:, :-torso_len, omit:]
            x_torso = signal[:, -torso_len:, omit:]

            if signal_source == 'heart':
                source = x_heart
                output = x_heart
            elif signal_source == 'torso':
                source = x_torso
                output = x_heart

            optimizer.zero_grad()

            kl_annealing_factor = determine_annealing_factor(kl_args['min_annealing_factor'],
                                                             kl_args['anneal_update'],
                                                             epoch - 1, len_epoch, n_steps)
            r_kl = kl_args['lambda']
            kl_factor = kl_annealing_factor * r_kl
            
            if k_shot is None:
                physics_vars, statistic_vars = model(source, data_name)
            else:
                D_x = data.D
                D_y = data.D_label
                D_x = D_x.to(device)
                D_y = D_y.to(device)

                if window is not None:
                    D_x = D_x[:, :, :window]

                N, M, T = signal.shape
                D_x = D_x.view(N, -1, M ,T)

                D_x_heart = D_x[:, :, :-torso_len, omit:]
                D_x_torso = D_x[:, :, -torso_len:, omit:]

                if signal_source == 'heart':
                    D_source = D_x_heart
                elif signal_source == 'torso':
                    D_source = D_x_torso
                
                if changable:
                    K = D_source.shape[1]
                    sub_K = np.random.randint(low=1, high=K+1, size=1)[0]
                    D_source = D_source[:, :sub_K, :]
                    D_y = D_y[:, :sub_K, :]

                physics_vars, statistic_vars = model(source, label, D_source, D_y, data_name)
            
            if loss_func == 'recon_loss':
                x_ = physics_vars[0]

                if loss_type is None:
                    loss_type = 'mse'
                
                total, separate_terms = loss(x_, output, loss_type)
            elif loss_func == 'meta_loss':
                x_ = physics_vars[0]
                mu_c, logvar_c, mu_t, logvar_t, mu_0, logvar_0 = statistic_vars

                if loss_type is None:
                    loss_type = 'mse'
                
                r1 = train_config.get('r1')
                r2 = train_config.get('r2')
                r3 = train_config.get('r3')
                l = train_config.get('l')
                if r1 is None:
                    r1 = 1
                if r2 is None:
                    r2 = 0
                if r3 is None:
                    r3 = 1
                
                total, separate_terms = \
                    loss(x_, output, mu_c, logvar_c, mu_t, logvar_t, mu_0, logvar_0, kl_factor, loss_type, r1, r2, r3, l)
            elif loss_func == 'elbo_loss':
                mu_x, logvar_x = physics_vars
                mu_z, logvar_z = statistic_vars

                total, separate_terms = loss(mu_x, logvar_x, output, mu_z, logvar_z, kl_factor)
            else:
                raise NotImplemented

            total.backward()

            total_loss += total.item()
            for loss_name, loss_val in separate_terms.items():
                if not separate_losses.get(loss_name):
                    separate_losses[loss_name] = 0
                separate_losses[loss_name] += loss_val.item()
            n_steps += 1

            optimizer.step()
            logs = 'Training epoch {}, step {}, Average loss for epoch: {:05.5f}'.format(epoch, n_steps, total_loss / n_steps)
            util.inline_print(logs)

    total_loss /= n_steps
    for loss_name, loss_val in separate_losses.items():
        separate_losses[loss_name] /= n_steps

    return total_loss, separate_losses


def valid_epoch(model, epoch, loss, data_loaders, hparams):
    model.eval()
    train_config = dict(hparams.training)
    kl_args = train_config.get('kl_args')
    torso_len = train_config.get('torso_len')
    signal_source = train_config.get('signal_source')
    omit = train_config.get('omit')
    window = train_config.get('window')
    k_shot = train_config.get('k_shot')
    changable = train_config.get('changable')
    meta_dataset = train_config.get('meta_dataset')
    loss_type = train_config.get('loss_type')
    loss_func = hparams.loss
    total_loss = 0
    separate_losses = {}
    n_steps = 0
    batch_size = hparams.batch_size

    with torch.no_grad():
        data_names = list(data_loaders.keys())
        len_epoch = len(data_loaders[data_names[0]]) * len(data_names)
        for data_name in data_names:
            data_loader = data_loaders[data_name]
            if epoch > 1 and meta_dataset:
                data_loader = data_loader.next()
            for idx, data in enumerate(data_loader):
                signal, label = data.x, data.y
                signal = signal.to(device)
                label = label.to(device)

                if window is not None:
                    signal = signal[:, :, :window]
                
                x_heart = signal[:, :-torso_len, omit:]
                x_torso = signal[:, -torso_len:, omit:]

                if signal_source == 'heart':
                    source = x_heart
                    output = x_heart
                elif signal_source == 'torso':
                    source = x_torso
                    output = x_heart

                r_kl = kl_args['lambda']
                kl_factor = 1 * r_kl
                
                if k_shot is None:
                    physics_vars, statistic_vars = model(source, data_name)
                else:
                    D_x = data.D
                    D_y = data.D_label
                    D_x = D_x.to(device)
                    D_y = D_y.to(device)

                    if window is not None:
                        D_x = D_x[:, :, :window]

                    N, M, T = signal.shape
                    D_x = D_x.view(N, -1, M ,T)
                    D_x_heart = D_x[:, :, :-torso_len, omit:]
                    D_x_torso = D_x[:, :, -torso_len:, omit:]

                    if signal_source == 'heart':
                        D_source = D_x_heart
                    elif signal_source == 'torso':
                        D_source = D_x_torso
                    
                    if changable:
                        K = D_source.shape[1]
                        sub_K = np.random.randint(low=1, high=K+1, size=1)[0]
                        D_source = D_source[:, :sub_K, :]
                        D_y = D_y[:, :sub_K, :]

                    physics_vars, statistic_vars = model(source, label, D_source, D_y, data_name)
                
                if loss_func == 'recon_loss':
                    x_, _ = physics_vars
                    total, separate_terms = loss(x_, output)
                elif loss_func == 'meta_loss':
                    x_ = physics_vars[0]
                    mu_c, logvar_c, mu_t, logvar_t, mu_0, logvar_0 = statistic_vars

                    if loss_type is None:
                        loss_type = 'mse'
                    
                    r1 = train_config.get('r1')
                    r2 = train_config.get('r2')
                    r3 = train_config.get('r3')
                    l = train_config.get('l')
                    if r1 is None:
                        r1 = 1
                    if r2 is None:
                        r2 = 0
                    if r3 is None:
                        r3 = 1
                    
                    total, separate_terms = \
                        loss(x_, output, mu_c, logvar_c, mu_t, logvar_t, mu_0, logvar_0, kl_factor, loss_type, r1, r2, r3, l)
                elif loss_func == 'elbo_loss':
                    mu_x, logvar_x = physics_vars
                    mu_z, logvar_z = statistic_vars

                    total, separate_terms = loss(mu_x, logvar_x, output, mu_z, logvar_z, kl_factor)
                else:
                    raise NotImplemented

                total_loss += total.item()
                for loss_name, loss_val in separate_terms.items():
                    if not separate_losses.get(loss_name):
                        separate_losses[loss_name] = 0
                    separate_losses[loss_name] += loss_val.item()
                n_steps += 1

    total_loss /= n_steps
    for loss_name, loss_val in separate_losses.items():
        separate_losses[loss_name] /= n_steps

    return total_loss, separate_losses


def determine_annealing_factor(min_anneal_factor,
                               anneal_update,
                               epoch, n_batch, batch_idx):
    n_updates = epoch * n_batch + batch_idx

    if anneal_update > 0 and n_updates < anneal_update:
        anneal_factor = min_anneal_factor + \
            (1.0 - min_anneal_factor) * (
                (n_updates / anneal_update)
            )
    else:
        anneal_factor = 1.0
    return anneal_factor


def save_losses(exp_dir, num_epochs, losses):
    """Plot epoch against train loss and test loss 
    """
    # plot of the train/validation error against num_epochs
    for loss_type, loss_cmb in losses.items():
        train_a, test_a = loss_cmb
        plot_loss(exp_dir, num_epochs, train_a, test_a, loss_type)
        train_a = np.array(train_a)
        test_a = np.array(test_a)
        np.save(os.path.join(exp_dir, 'loss_{}_t.npy'.format(loss_type)), train_a)
        np.save(os.path.join(exp_dir, 'loss_{}_e.npy'.format(loss_type)), test_a)


def plot_loss(exp_dir, num_epochs, train_a, test_a, loss_type):
    fig, ax1 = plt.subplots(figsize=(6, 5))
    ax1.set_xticks(np.arange(0 + 1, num_epochs + 1, step=10))
    ax1.set_xlabel('epochs')
    ax1.plot(train_a, color='green', ls='-', label='train accuracy')
    ax1.plot(test_a, color='red', ls='-', label='validation accuracy')
    h1, l1 = ax1.get_legend_handles_labels()
    ax1.legend(h1, l1, fontsize='14', frameon=False)
    ax1.grid(linestyle='--')
    plt.tight_layout()
    fig.savefig(exp_dir + '/loss_{}.png'.format(loss_type), dpi=300, bbox_inches='tight')
    plt.close()
