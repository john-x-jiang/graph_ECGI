import torch
import torch.nn as nn


def mse_loss(x_, x, reduction='sum'):
    mse = nn.MSELoss(reduction=reduction)(x_, x)
    return mse


def nll_loss(x_, x, reduction='none', loss_type='mse'):
    if loss_type == 'mse':
        return nn.MSELoss(reduction=reduction)(x_, x)
    elif loss_type == 'bce':
        x = torch.sigmoid(x)
        x_ = torch.sigmoid(x_)
        return nn.BCELoss(reduction=reduction)(x_, x)
    elif loss_type == 'bce_with_logits':
        x = torch.sigmoid(x)
        x_ = torch.sigmoid(x_)
        return nn.BCEWithLogitsLoss(reduction=reduction)(x_, x)
    else:
        raise NotImplemented


def kl_div(mu1, var1, mu2=None, var2=None):
    if mu2 is None:
        mu2 = torch.zeros_like(mu1)
    if var2 is None:
        var2 = torch.zeros_like(mu1)

    return 0.5 * (
        var2 - var1 + (
            torch.exp(var1) + (mu1 - mu2).pow(2)
        ) / torch.exp(var2) - 1)


def kl_div_stn(mu, logvar):
    return 0.5 * (
        mu.pow(2) + torch.exp(logvar) - logvar - 1
    )


def recon_loss(x_, x):
    B, T = x.shape[0], x.shape[-1]
    nll_raw_0 = mse_loss(x_[:, :, 0], x[:, :, 0], 'none')
    nll_raw = mse_loss(x_[:, :, 1:], x[:, :, 1:], 'none')

    nll_m_0 = nll_raw_0.sum() / B
    nll_m = nll_raw.sum() / (B * (T - 1))

    total = nll_m_0 + nll_m
    return total


def domain_recon_loss(x_, x, D_, D, mu_c, logvar_c, kl_annealing_factor=1, loss_type='mse'):
    B, T = x.shape[0], x.shape[-1]
    nll_raw = nll_loss(x_, x, 'none', loss_type)
    nll_m = nll_raw.sum() / B

    # K = D.shape[1]
    nll_raw_D = nll_loss(D_, D, 'none', loss_type)
    nll_m_D = nll_raw_D.sum() / B

    kl_raw_c = kl_div_stn(mu_c, logvar_c)
    kl_m_c = kl_raw_c.sum() / B

    total = kl_annealing_factor * kl_m_c + nll_m + nll_m_D

    return kl_m_c, nll_m, nll_m_D, total


def domain_recon_loss_1(x_, x, D_, D, mu_c, logvar_c, kl_annealing_factor=1, loss_type='mse'):
    B, T = x.shape[0], x.shape[-1]
    nll_raw_0 = nll_loss(x_[:, :, 0], x[:, :, 0], 'none', loss_type)
    nll_raw = nll_loss(x_[:, :, 1:], x[:, :, 1:], 'none', loss_type)

    nll_m_0 = nll_raw_0.sum() / B
    nll_m = nll_raw.sum() / (B * (T - 1))

    # K = D.shape[1]
    # nll_raw_D_0 = nll_loss(D_[:, :, :, 0], D[:, :, :, 0], 'none', loss_type)
    # nll_raw_D = nll_loss(D_[:, :, :, 1:], D[:, :, :, 1:], 'none', loss_type)

    # nll_m_D_0 = nll_raw_D_0.sum() / B
    # nll_m_D = nll_raw_D.sum() / B

    kl_raw_c = kl_div_stn(mu_c, logvar_c)
    kl_m_c = kl_raw_c.sum() / B

    # total = kl_annealing_factor * kl_m_c + nll_m_0 + nll_m + nll_m_D_0 + nll_m_D
    total = kl_annealing_factor * kl_m_c + nll_m_0 + nll_m

    return kl_m_c, nll_m, nll_m_0, total


def domain_recon_loss_2(x_, x, D_, D, mu_c, logvar_c, kl_annealing_factor=1, loss_type='mse'):
    B, T = x.shape[0], x.shape[-1]
    nll_raw_0 = nll_loss(x_[:, :, 0], x[:, :, 0], 'none', loss_type)
    nll_raw = nll_loss(x_[:, :, 1:], x[:, :, 1:], 'none', loss_type)

    nll_m_0 = nll_raw_0.sum() / B
    nll_m = nll_raw.sum() / B

    # K = D.shape[1]
    # nll_raw_D_0 = nll_loss(D_[:, :, :, 0], D[:, :, :, 0], 'none', loss_type)
    # nll_raw_D = nll_loss(D_[:, :, :, 1:], D[:, :, :, 1:], 'none', loss_type)

    # nll_m_D_0 = nll_raw_D_0.sum() / B
    # nll_m_D = nll_raw_D.sum() / B

    kl_raw_c = kl_div_stn(mu_c, logvar_c)
    kl_m_c = kl_raw_c.sum() / B

    # total = kl_annealing_factor * kl_m_c + nll_m_0 + nll_m + nll_m_D_0 + nll_m_D
    total = kl_annealing_factor * kl_m_c + nll_m_0 + nll_m

    return kl_m_c, nll_m, nll_m_0, total


def domain_recon_loss_3(x_, x, D_, D, mu_c, logvar_c, kl_annealing_factor=1, loss_type='mse'):
    B, T = x.shape[0], x.shape[-1]
    nll_raw_0 = nll_loss(x_[:, :, 0], x[:, :, 0], 'none', loss_type)
    nll_raw = nll_loss(x_[:, :, 1:], x[:, :, 1:], 'none', loss_type)

    nll_m_0 = nll_raw_0.sum() / B
    nll_m = nll_raw.sum() / (B * (T - 1))

    # K = D.shape[1]
    # nll_raw_D_0 = nll_loss(D_[:, :, :, 0], D[:, :, :, 0], 'none', loss_type)
    # nll_raw_D = nll_loss(D_[:, :, :, 1:], D[:, :, :, 1:], 'none', loss_type)

    # nll_m_D_0 = nll_raw_D_0.sum() / B
    # nll_m_D = nll_raw_D.sum() / B

    V = mu_c.shape[1]
    kl_raw_c = kl_div_stn(mu_c, logvar_c)
    kl_m_c = kl_raw_c.sum() / (B * V)

    # total = kl_annealing_factor * kl_m_c + nll_m_0 + nll_m + nll_m_D_0 + nll_m_D
    total = kl_annealing_factor * kl_m_c + nll_m_0 + nll_m

    return kl_m_c, nll_m, nll_m_0, total


def dmm_loss(x, x_q, x_p, mu1, var1, mu2, var2, kl_annealing_factor=1, r1=1, r2=0):
    B, T = x.shape[0], x.shape[-1]
    nll_raw_q = mse_loss(x_q, x[:, :, :T], 'none')
    nll_raw_p = mse_loss(x_p, x[:, :, :T], 'none')
    nll_m_q = nll_raw_q.sum() / B
    nll_m_p = nll_raw_p.sum() / B

    if mu1 is not None:
        kl_raw = kl_div(mu1, var1, mu2, var2)
        kl_raw, kl_raw_D = kl_raw[:, :, :, :-1], kl_raw[:, :, :, -1]
        kl_m = kl_raw.sum() / B
        kl_m_D = kl_raw_D.sum() / B
    else:
        kl_m = torch.zeros_like(x).sum() / B
        kl_m_D = torch.zeros_like(x).sum() / B

    loss = (kl_m + kl_m_D) * kl_annealing_factor + r1 * nll_m_q + r2 * nll_m_p

    return kl_m, nll_m_q, nll_m_p, loss
