import torch
import torch.nn as nn


def nll_loss(x_, x, reduction='none', loss_type='mse'):
    if loss_type == 'mse':
        return nn.MSELoss(reduction=reduction)(x_, x)
    elif loss_type == 'bce':
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


def recon_loss(x_, x, loss_type='mse'):
    B, T = x.shape[0], x.shape[-1]
    nll_raw = nll_loss(x_, x, 'none', loss_type)
    nll_m = nll_raw[:, :, :].sum() / (B * T)

    total = nll_m

    return total, {'likelihood': nll_m}


def physics_loss(y_, y, LX, loss_type='mse', r1=1, r2=0):
    B, T = y.shape[0], y.shape[-1]
    nll_raw_y = nll_loss(y_, y[:, :, :T], 'none', loss_type)
    reg_raw = nll_loss(LX, torch.zeros_like(LX), 'none', loss_type)

    nll_m_y = nll_raw_y.sum() / B
    reg_m = reg_raw.sum() / B

    total = nll_m_y + r1 * reg_m

    return total, {'likelihood_y': nll_m_y, 'regularization': reg_m}


def mixed_loss(y_, y, x_, x, LX, loss_type='mse', r1=1, r2=0, is_real=None):
    B, T = y.shape[0], y.shape[-1]
    nll_raw_y = nll_loss(y_, y[:, :, :T], 'none', loss_type)
    reg_raw = nll_loss(LX, torch.zeros_like(LX), 'none', loss_type)

    nll_raw_x = nll_loss(x_, x[:, :, :T], 'none', loss_type)

    nll_m_y = nll_raw_y.sum() / B
    reg_m = reg_raw.sum() / B

    nll_m_x = nll_raw_x.sum() / B
    reg_m_x = torch.zeros_like(reg_m)

    if not is_real:
        total = nll_m_y + r2 * nll_m_x
    else:
        total = nll_m_y + r1 * reg_m

    return total, {'likelihood_y': nll_m_y, 'likelihood_x': nll_m_x, 'regularization': reg_m}


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

    return total, {'likelihood': nll_m, 'kl_domain': kl_m_c_t, 'kl_domain_reg': kl_m_c, 'kl_initial': kl_m_0}


def meta_loss_with_mask(x_, x, mu_c, logvar_c, mu_t, logvar_t, kl_annealing_factor=1, loss_type='mse', r1=1, r2=0, l=1):
    B, T = x.shape[0], x.shape[-1]
    nll_raw = nll_loss(x_, x, 'none', loss_type)
    nll_0 = nll_raw[:, :, 0].sum() / B
    nll_r = nll_raw[:, :, 1:].sum() / B / (T - 1)
    nll_m = T * (nll_0 * l + nll_r)

    kl_raw_c_t = kl_div(mu_c, logvar_c, mu_t, logvar_t)
    kl_m_c_t = kl_raw_c_t.sum() / B

    kl_raw_c = kl_div_stn(mu_c, logvar_c)
    kl_m_c = kl_raw_c.sum() / B

    total = kl_annealing_factor * (r1 * kl_m_c_t + r2 * kl_m_c) + nll_m

    return total, {'likelihood': nll_m, 'kl_domain': kl_m_c_t, 'kl_domain_reg': kl_m_c}


def elbo_loss(muTheta, logvarTheta, x, mu, logvar, annealParam):
    B, V, T = x.shape[0], x.shape[1], x.shape[-1]
    V_latent = mu.shape[1]
    diffSq = (x - muTheta).pow(2)
    precis = torch.exp(-logvarTheta)

    nll_m = 0.5 * torch.sum(logvarTheta + torch.mul(diffSq,precis))
    nll_m /= (B * V * T)

    kl_m = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_m /= B * V_latent * T

    total = nll_m + annealParam * kl_m

    return total, {'likelihood': nll_m, 'kl': kl_m}
