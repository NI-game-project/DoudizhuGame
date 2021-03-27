import torch


def disable_gradients(network):
    # Disable calculations of gradients.
    for param in network.parameters():
        param.requires_grad = False


def calculate_kl_divergence_two_dist(dist_p, dist_q):
    kld = torch.sum(dist_p * (torch.log(dist_p) - torch.log(dist_q)))
    return kld


def calculate_quantile_loss_penalties(u, tau):
    # [1, n_quantile, 1]
    tau = tau.view(1, -1, 1)

    # [batch_size, n_quantile, n_quantile]
    quantile_penalties = torch.abs(tau - u.le(0.).float())
    return quantile_penalties


def calculate_huber_loss(u, kappa=1.):
    """
    This quantile regression loss acts as an asymmetric squared loss in an interval [-k,k] around zero
    and reverts to a standard quantile loss outside this interval.
    """
    # kappa-smoothing parameter for the Huber loss
    if kappa == 0.:
        # Pure Quantile Regression Loss
        huber_loss = u.abs()
    else:
        # Quantile Huber Loss
        # Calculate huber loss element-wisely.
        # if |u|<=k: Lku = 1/2(u**2), else: Lku = k(|u|-1/2k)
        huber_loss = torch.where(u.abs() <= kappa, 0.5 * u.pow(2), kappa * (u.abs() - 0.5 * kappa))
    return huber_loss
