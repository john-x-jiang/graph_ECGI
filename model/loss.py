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


def dmm_loss(x, x_, mu, logvar, kl_annealing_factor=1):
    B, T = x.shape[0], x.shape[-1]
    nll_raw = mse_loss(x_, x[:, :, :T], 'none')
    nll_m = nll_raw.sum() / B

    kl_raw = kl_div_stn(mu, logvar)
    kl_m = kl_raw.sum() / B

    loss = kl_m * kl_annealing_factor + nll_m

    return kl_m, nll_m, loss


def meta_loss(x_, x, mu_c, logvar_c, mu_t, logvar_t, mu_0, logvar_0, kl_annealing_factor=1, loss_type='mse', r1=1, r2=0, r3=1, l=1):
    B, T = x.shape[0], x.shape[-1]
    nll_raw = nll_loss(x_, x, 'none', loss_type)
    nll_0 = nll_raw[:, :, 0].sum() / B
    nll_r = nll_raw[:, :, 1:].sum() / B / (T - 1)
    nll_m = T * (nll_0 * l + nll_r)

    kl_raw_c_t = kl_div(mu_c, logvar_c, mu_t, logvar_t)
    kl_m_c_t = kl_raw_c_t.sum() / B

    kl_raw_c = kl_div_stn(mu_c, logvar_c)
    kl_m_c = kl_raw_c.sum() / B

    kl_raw_0 = kl_div_stn(mu_0, logvar_0)
    kl_m_0 = kl_raw_0.sum() / B

    total = kl_annealing_factor * (r1 * kl_m_c_t + r2 * kl_m_c + r3 * kl_m_0) + nll_m

    return kl_m_c_t, nll_m, kl_m_0, total


def elbo_loss(muTheta, logvarTheta, x, mu, logvar, annealParam):
    B, V, T = x.shape[0], x.shape[1], x.shape[-1]
    V_latent = mu.shape[1]
    diffSq = (x - muTheta).pow(2)
    precis = torch.exp(-logvarTheta)

    BCE = 0.5 * torch.sum(logvarTheta + torch.mul(diffSq,precis))
    BCE /= (B * V * T)

    KLD = -0.5 * annealParam * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= B * V_latent * T

    loss = BCE + KLD

    return KLD, BCE, loss
