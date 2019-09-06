import torch
import torch.nn.functional as F

from misc.lovasz_losses import *


def sigmoid_focal_loss_with_logits(input, target, reduction='mean', alpha=0.25, gamma=2.0, eps=1e-12):
    """
    input: (N, *)
    target: (N, *)
    """
    assert (input.shape == target.shape)

    eps = torch.tensor(eps).to(input.device)
    input = input if gamma == 0.0 else torch.clamp(input, (eps/(1.-eps)).log(), ((1.-eps)/eps).log())

    ## Ref: https://github.com/richardaecn/class-balanced-loss/blob/master/src/cifar_main.py#L226-L266
    # A numerically stable implementation of modulator.
    bce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
    modulator = 1.0 if gamma == 0.0 else torch.exp(-gamma * target * input - gamma * torch.log1p(torch.exp(-1.0 * input)))
    loss = modulator * bce_loss

    target_weight = (alpha * target + (1.-alpha) * (1.-target))
    loss = target_weight * loss

    if reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'mean':
        target_weight_sum = target.sum() * target + (1.-target).sum() * (1.-target) + eps
        loss = (loss / target_weight_sum).sum() ##loss.mean()
    return loss


def dice_loss(input, target, per_image=True, weight=[0.0, 1.0], smooth=1.0):
    assert (input.shape == target.shape)
    bs = input.size(0)
    dim = -1 if per_image else None

    input = F.sigmoid(input).view(bs, -1)
    target = target.float().view(bs, -1)

    intersection = (input * target).sum(dim=dim) * weight[1] + ((1.-input) * (1.-target)).sum(dim=dim) * weight[0]
    union = (input.sum(dim=dim) + target.sum(dim=dim)) * weight[1] + ((1.-input).sum(dim=dim) + (1.-target).sum(dim=dim)) * weight[0]
    coefficient = 2. * (intersection + smooth) / (union + smooth)
    loss = 1. - coefficient
    loss = loss.mean() if per_image else loss
    return loss


def symmetric_lovasz_hinge(input, target, margin=[1.0, 1.0]):
    return (lovasz_hinge(input, target, margin=margin) + lovasz_hinge(-input, 1.-target, margin=margin[::-1])) / 2.


def multi_scale_loss(input, target, scale_weight=None, reduction='mean', alpha=1.0, gamma=2.0, dice_weight=[0.0, 1.0], lv_margin=[1.0, 1.0], lambda_fl=1.0, lambda_dc=1.0, lambda_lv=1.0): # for auxiliary learning
    if not isinstance(input, tuple):
        fl_loss = sigmoid_focal_loss_with_logits(input, target, reduction=reduction, alpha=alpha, gamma=gamma) if lambda_fl > 0 else torch.tensor(0.0).to(target.device) # F.binary_cross_entropy_with_logits
        dc_loss = dice_loss(input, target, weight=dice_weight) if lambda_dc > 0 else torch.tensor(0.0).to(target.device)
        lv_loss = symmetric_lovasz_hinge(input, target, margin=lv_margin) if lambda_lv > 0 else torch.tensor(0.0).to(target.device)
        loss = lambda_fl * fl_loss + lambda_dc * dc_loss + lambda_lv * lv_loss
        return loss

    n_inp = len(input)
    if scale_weight is None: # scale_weight: torch tensor type
        scale = 0.4
        scale_weight = torch.pow(scale * torch.ones(n_inp).to(target.device), torch.arange(n_inp).to(target.device).float())

    ms_loss = 0.0
    for i in range(n_inp):
        target_i = F.interpolate(target, size=input[i].shape[2:], mode='nearest').detach() if target.shape != input[i].shape[2:] else target
        fl_loss = sigmoid_focal_loss_with_logits(input[i], target_i, reduction=reduction, alpha=alpha, gamma=gamma) if lambda_fl > 0 else torch.tensor(0.0).to(target.device) # F.binary_cross_entropy_with_logits
        dc_loss = dice_loss(input[i], target_i, weight=dice_weight) if lambda_dc > 0 else torch.tensor(0.0).to(target.device)
        lv_loss = symmetric_lovasz_hinge(input[i], target_i, margin=lv_margin) if lambda_lv > 0 else torch.tensor(0.0).to(target.device)
        loss = lambda_fl * fl_loss + lambda_dc * dc_loss + lambda_lv * lv_loss
        ms_loss = ms_loss + scale_weight[i] * loss
    return ms_loss
