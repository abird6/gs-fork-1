#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np
from utils.image_utils import pixels_to_bayer_mask

def l1_loss_mean(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt))

def l2_loss(network_output, gt): 
    return ((network_output - gt) ** 2)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    


''' RawSplats Loss Utils'''


def huber_loss(render, gt, delta, weight):
    l1_err = torch.abs(render - gt)
    linear_loss = delta * (l1_err - 0.5 * delta)

    weighted_quad_loss = linear_loss * weight
    weighted_linear_loss = linear_loss * ((1 - weight) * delta)
    loss = (torch.where(l1_err <= delta, weighted_quad_loss, weighted_linear_loss)).mean()
    return loss

'''
    Determines loss function to apply based on model's "loss_type" parameter.
    Loss Fn types:
        0:  L - original 3DGS loss
        1:  L1 loss w/ gamma curve - adapted from RawNeRF
        2:  L1 loss w/ gamma curve and bayer masking - adapted from RawNeRF - only works on full res images
        3:  L2 loss
        4:  L2 loss w/ gamma curve - adapted from RawNeRF
        5:  L2 loss w/ gamma curve and bayer masking - adapted from RawNeRF - only works on full res images
        6:  Huber loss - more weight on fine errors
        7:  L + L1 gamma curve weight (lambda)
        8:  L + L2 gamma curve weight (lambda)
        9:  L + L1 gamma curve weight + bayer mask (lambda)
        10: L + L2 gamma curve weight + bayer mask (lambda)

'''
def loss_fn(render, gt, bayer_mask, lp, opt, iteration):
    # convert bayer mask to tensor
    bayer_mask = torch.from_numpy(bayer_mask)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bayer_mask = bayer_mask.to(device)
    
    # gamma curve to weight errors in dark regions more
    scaling_grad = 1. / (1e-3 + render.detach())

    if lp.loss_type == 1: # L1 w/ gamma curve
        Ll1 = l1_loss(render, gt)
        loss = (1.0 - opt.lambda_dssim) * (Ll1 * scaling_grad).mean() + opt.lambda_dssim * (1.0 - ssim(render, gt))

    elif lp.loss_type == 2: # L1 w/ gamma curve and bayer masking
        Ll1 = l1_loss(render, gt)
        loss = (1.0 - opt.lambda_dssim) * (bayer_mask * (Ll1 * scaling_grad)).mean() + opt.lambda_dssim * (1.0 - ssim(render, gt))

    elif lp.loss_type == 3: # L2
        Ll2 = l2_loss(render, gt)
        loss = (1.0 - opt.lambda_dssim) * Ll2.mean() + opt.lambda_dssim * (1.0 - ssim(render, gt))

    elif lp.loss_type == 4: # L2 w/ gamma curve
        Ll2 = l2_loss(render, gt)
        loss = (1.0 - opt.lambda_dssim) * (Ll2 * (scaling_grad**2)).mean() + opt.lambda_dssim * (1.0 - ssim(render, gt))

    elif lp.loss_type == 5: # L2 w/ gamma curve and bayer masking
        Ll2 = l2_loss(render, gt)
        loss = (1.0 - opt.lambda_dssim) * (bayer_mask * (Ll2 * (scaling_grad**2))).mean() + opt.lambda_dssim * (1.0 - ssim(render, gt))

    elif lp.loss_type == 6: # huber loss
        loss = huber_loss(render, gt, delta=opt.huber_delta_thresh, weight=opt.huber_weight)

    elif lp.loss_type == 7: # L + L1 gamma curve weight (lambda)
        Ll1 = l1_loss(render, gt)
        loss = (1.0 - opt.lambda_dssim) * (Ll1.mean()) + opt.lambda_dssim * (1.0 - ssim(render, gt)) + (1.0 - opt.lambda_dssim) * ((Ll1 * scaling_grad).mean())

    elif lp.loss_type == 8: # L + L2 gamma curve weight (lambda)
        Ll1 = l1_loss(render, gt)
        Ll2 = l2_loss(render, gt)
        loss = (1.0 - opt.lambda_dssim) * (Ll1.mean()) + opt.lambda_dssim * (1.0 - ssim(render, gt)) + (1.0 - opt.lambda_dssim) * ((Ll2 * (scaling_grad**2)).mean())

    elif lp.loss_type == 9: # L + L1 gamma curve weight + bayer mask (lambda)
        Ll1 = l1_loss(render, gt)
        loss = (1.0 - opt.lambda_dssim) * (Ll1.mean()) + opt.lambda_dssim * (1.0 - ssim(render, gt)) + (1.0 - opt.lambda_dssim) * ((bayer_mask * (Ll1 * scaling_grad)).mean())

    elif lp.loss_type == 10: # L + L2 gamma curve weight + bayer mask (lambda)
        Ll1 = l1_loss(render, gt)
        Ll2 = l2_loss(render, gt)
        loss = (1.0 - opt.lambda_dssim) * (Ll1.mean()) + opt.lambda_dssim * (1.0 - ssim(render, gt)) + (1.0 - opt.lambda_dssim) * ((bayer_mask * (Ll2 * (scaling_grad**2))).mean())

    elif lp.loss_type == 11: # gamma correction loss
        loss = ((l1_loss(render, gt) / (torch.abs(render.detach() + gt) + 1e-7))**(1/1.5)).mean()

    elif lp.loss_type == 12: # gamma correction loss w/ ssim
        loss = ((1.0 - opt.lambda_dssim) * ((l1_loss(render, gt) / (torch.abs(render.detach() + gt) + 1e-7))**(1/2.2)).mean()) + (opt.lambda_dssim * (1.0 - ssim(render, gt)))

    elif lp.loss_type == 13: # L + gamma correction loss + ssim
        loss = (1.0 - opt.lambda_dssim) * l1_loss(render, gt).mean() + opt.lambda_dssim * (1.0 - ssim(render, gt)) + opt.lambda_dssim * ((l1_loss(render, gt) / (torch.abs(render.detach() + gt) + 1e-7))**(1/2.2)).mean()

    elif lp.loss_type == 14: # L + huber loss (@ 50% training iterations)
        if iteration >= opt.iterations / 2:
            loss = huber_loss(render, gt, delta=0.0002, weight=100)
        else:
            Ll1 = l1_loss(render, gt)
            loss = (1.0 - opt.lambda_dssim) * Ll1.mean() + opt.lambda_dssim * (1.0 - ssim(render, gt))

    else: # original 3DGS loss
        Ll1 = l1_loss(render, gt)
        loss = (1.0 - opt.lambda_dssim) * Ll1.mean() + opt.lambda_dssim * (1.0 - ssim(render, gt))

        
    return loss
