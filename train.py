import sys, os
import cv2
import torch
import argparse
import timeit
import random
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.utils import data
from tqdm import tqdm

from models import get_model
from loaders import get_loader, get_data_path
from misc.losses import *
from misc.lovasz_losses import *
from misc.radam import RAdam
from misc.scheduler import GradualWarmupScheduler
from misc.utils import convert_state_dict, poly_lr_scheduler, AverageMeter
from misc.metrics import runningScore

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

def group_weight(module):
    group_decay = []
    group_no_decay = []
    for name, m in module.named_modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm) or isinstance(m, nn.modules.normalization.GroupNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        else:
            if 'gamma' in [name for name, p in m.named_parameters()]:
                group_no_decay.append(m.gamma) # PAM / CAM
            if 'gamma_gap' in [name for name, p in m.named_parameters()]:
                group_no_decay.append(m.gamma_gap)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups

def train(args):
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')

    # Setup Augmentations & Transforms
    rgb_mean = [122.7717/255., 115.9465/255., 102.9801/255.] if args.norm_type == 'gn' and args.load_pretrained else [0.485, 0.456, 0.406]
    rgb_std = [1./255., 1./255., 1./255.] if args.norm_type == 'gn' and args.load_pretrained else [0.229, 0.224, 0.225]
    data_trans = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(size=(args.img_rows, args.img_cols)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=rgb_mean, std=rgb_std),
                ])

    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    t_loader = data_loader(data_path, transforms=data_trans, in_channels=args.in_channels, split='train', augmentations=True, fold_num=args.fold_num, num_folds=args.num_folds, only_non_empty=args.only_non_empty, seed=args.seed, mask_dilation_size=args.mask_dilation_size)
    v_loader = data_loader(data_path, transforms=data_trans, in_channels=args.in_channels, split='val', fold_num=args.fold_num, num_folds=args.num_folds, only_non_empty=args.only_non_empty, seed=args.seed)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=2, pin_memory=True, shuffle=args.only_non_empty, drop_last=args.only_non_empty)
    valloader = data.DataLoader(v_loader, batch_size=args.batch_size, num_workers=2, pin_memory=True)

    # Setup Model
    model = get_model(args.arch, n_classes=1, in_channels=args.in_channels, norm_type=args.norm_type, load_pretrained=args.load_pretrained, use_cbam=args.use_cbam)
    model.to(torch.device(args.device))

    running_metrics = runningScore(n_classes=2, weight_acc_non_empty=args.weight_acc_non_empty, device=args.device)

    # Check if model has custom optimizer / loss
    if hasattr(model, 'optimizer'):
        optimizer = model.optimizer
    else:
        warmup_iter = int(args.n_iter*5./100.)
        milestones = [int(args.n_iter*30./100.)-warmup_iter, int(args.n_iter*60./100.)-warmup_iter, int(args.n_iter*90./100.)-warmup_iter] # [30, 60, 90]
        gamma = 0.5 #0.1

        if args.optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(group_weight(model), lr=args.l_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(group_weight(model), lr=args.l_rate, weight_decay=args.weight_decay)
        else:#if args.optimizer_type == 'radam':
            optimizer = RAdam(group_weight(model), lr=args.l_rate, weight_decay=args.weight_decay)

        if args.num_cycles > 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.n_iter-warmup_iter)//args.num_cycles, eta_min=args.l_rate*0.1)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        scheduler_warmup = GradualWarmupScheduler(optimizer, total_epoch=warmup_iter, min_lr_mul=0.1, after_scheduler=scheduler)

    start_iter = 0
    if args.resume is not None:                                         
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device(args.device))#, encoding="latin1")

            model_dict = model.state_dict()
            if checkpoint.get('model_state', None) is not None:
                model_dict.update(convert_state_dict(checkpoint['model_state']))
            else:
                model_dict.update(convert_state_dict(checkpoint))

            start_iter = checkpoint.get('iter', -1)
            dice_val = checkpoint.get('dice', -1)
            wacc_val = checkpoint.get('wacc', -1)
            print("Loaded checkpoint '{}' (iter {}, dice {:.5f}, wAcc {:.5f})"
                  .format(args.resume, start_iter, dice_val, wacc_val))

            model.load_state_dict(model_dict)

            if checkpoint.get('optimizer_state', None) is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state'])

            del model_dict
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("No checkpoint found at '{}'".format(args.resume)) 
    start_iter = args.start_iter if args.start_iter >= 0 else start_iter

    scale_weight = torch.tensor([1.0, 0.4, 0.4, 0.4]).to(torch.device(args.device))
    dice_weight = [args.dice_weight0, args.dice_weight1]
    lv_margin = [args.lv_margin0, args.lv_margin1]
    total_loss_sum = 0.0
    ms_loss_sum = 0.0
    cls_loss_sum = 0.0
    t_loader.__gen_batchs__(args.batch_size, ratio=args.ratio)
    trainloader_iter = iter(trainloader)
    optimizer.zero_grad()
    start_train_time = timeit.default_timer()
    elapsed_train_time = 0.0
    best_dice = -100.0
    best_wacc = -100.0
    for i in range(start_iter, args.n_iter):
        #"""
        model.train()

        if i % args.iter_size == 0:
            if args.num_cycles == 0:
                scheduler_warmup.step(i)
            else:
                scheduler_warmup.step(i // args.num_cycles)

        try:
            images, labels, _ = next(trainloader_iter)
        except:
            t_loader.__gen_batchs__(args.batch_size, ratio=args.ratio)
            trainloader_iter = iter(trainloader)
            images, labels, _ = next(trainloader_iter)

        images = images.to(torch.device(args.device))
        labels = labels.to(torch.device(args.device))

        outputs, outputs_gap = model(images)

        labels_gap = torch.where(labels.sum(3, keepdim=True).sum(2, keepdim=True) > 0, torch.ones(labels.size(0),1,1,1).to(torch.device(args.device)), torch.zeros(labels.size(0),1,1,1).to(torch.device(args.device)))
        cls_loss = F.binary_cross_entropy_with_logits(outputs_gap, labels_gap) if args.lambda_cls > 0 else torch.tensor(0.0).to(labels.device)
        ms_loss = multi_scale_loss(outputs, labels, scale_weight=scale_weight, reduction='mean', alpha=args.alpha, gamma=args.gamma, dice_weight=dice_weight, lv_margin=lv_margin, lambda_fl=args.lambda_fl, lambda_dc=args.lambda_dc, lambda_lv=args.lambda_lv)
        total_loss = ms_loss + args.lambda_cls * cls_loss
        total_loss = total_loss / float(args.iter_size)
        total_loss.backward()
        total_loss_sum = total_loss_sum + total_loss.item()
        ms_loss_sum = ms_loss_sum + ms_loss.item()
        cls_loss_sum = cls_loss_sum + cls_loss.item()

        if (i+1) % args.print_train_freq == 0:
            print("Iter [%7d/%7d] Loss: %7.4f (MS: %7.4f / CLS: %7.4f)" % (i+1, args.n_iter, total_loss_sum, ms_loss_sum, cls_loss_sum))

        if (i+1) % args.iter_size == 0:
            optimizer.step()
            optimizer.zero_grad()
            total_loss_sum = 0.0
            ms_loss_sum = 0.0
            cls_loss_sum = 0.0
        #"""

        if args.eval_freq > 0 and (i+1) % args.eval_freq == 0:
            state = {'iter': i+1,
                     'model_state': model.state_dict(),}
                     #'optimizer_state': optimizer.state_dict(),}
            if (i+1) % int(args.eval_freq / args.save_freq) == 0:
                torch.save(state, "checkpoints/{}_{}_{}_{}x{}_{}-{}_model.pth".format(args.arch, args.dataset, i+1, args.img_rows, args.img_cols, args.fold_num, args.num_folds))

            dice_val = 0.0
            thresh = 0.5
            mask_sum_thresh = 0
            mean_loss_val = AverageMeter()
            model.eval()
            with torch.no_grad():
                for i_val, (images_val, labels_val, _) in enumerate(valloader):
                    images_val = images_val.to(torch.device(args.device))
                    labels_val = labels_val.to(torch.device(args.device))

                    outputs_val, outputs_gap_val = model(images_val)
                    pred_val = (F.sigmoid(outputs_val if not isinstance(outputs_val, tuple) else outputs_val[0]) > thresh).long() #outputs_val.max(1)[1]

                    pred_val_sum = pred_val.sum(3).sum(2).sum(1)
                    for k in range(labels_val.size(0)):
                        if pred_val_sum[k] < mask_sum_thresh:
                            pred_val[k, :, :, :] = torch.zeros_like(pred_val[k, :, :, :])

                    labels_gap_val = torch.where(labels_val.sum(3, keepdim=True).sum(2, keepdim=True) > 0, torch.ones(labels_val.size(0),1,1,1).to(torch.device(args.device)), torch.zeros(labels_val.size(0),1,1,1).to(torch.device(args.device)))
                    cls_loss_val = F.binary_cross_entropy_with_logits(outputs_gap_val, labels_gap_val) if args.lambda_cls > 0 else torch.tensor(0.0).to(labels_val.device)
                    ms_loss_val = multi_scale_loss(outputs_val, labels_val, scale_weight=scale_weight, reduction='mean', alpha=args.alpha, gamma=args.gamma, dice_weight=dice_weight, lv_margin=lv_margin, lambda_fl=args.lambda_fl, lambda_dc=args.lambda_dc, lambda_lv=args.lambda_lv)
                    loss_val = ms_loss_val + args.lambda_cls * cls_loss_val
                    mean_loss_val.update(loss_val.item(), n=labels_val.size(0))

                    running_metrics.update(labels_val.long(), pred_val.long())

            dice_val, dice_empty_val, dice_non_empty_val, miou_val, wacc_val, acc_empty_val, acc_non_empty_val = running_metrics.get_scores()
            print('Dice (per image): {:.5f} (empty: {:.5f} / non-empty: {:.5f})'.format(dice_val, dice_empty_val, dice_non_empty_val))
            print('wAcc: {:.5f} (empty: {:.5f} / non-empty: {:.5f})'.format(wacc_val, acc_empty_val, acc_non_empty_val))
            print('Overall mIoU: {:.5f}'.format(miou_val))
            print('Mean val loss: {:.4f}'.format(mean_loss_val.avg))
            state['dice'] = dice_val
            state['wacc'] = wacc_val
            state['miou'] = miou_val
            running_metrics.reset()
            mean_loss_val.reset()

            if (i+1) % int(args.eval_freq / args.save_freq) == 0:
                torch.save(state, "checkpoints/{}_{}_{}_{}x{}_{}-{}_model.pth".format(args.arch, args.dataset, i+1, args.img_rows, args.img_cols, args.fold_num, args.num_folds))
            if best_dice <= dice_val:
                best_dice = dice_val
                torch.save(state, "checkpoints/{}_{}_{}_{}x{}_{}-{}_model.pth".format(args.arch, args.dataset, 'best-dice', args.img_rows, args.img_cols, args.fold_num, args.num_folds))
            if best_wacc <= wacc_val:
                best_wacc = wacc_val
                torch.save(state, "checkpoints/{}_{}_{}_{}x{}_{}-{}_model.pth".format(args.arch, args.dataset, 'best-wacc', args.img_rows, args.img_cols, args.fold_num, args.num_folds))

            elapsed_train_time = timeit.default_timer() - start_train_time
            print('Training time (iter {0:5d}): {1:10.5f} seconds'.format(i+1, elapsed_train_time))

        if args.saving_last_time > 0 and (i+1) % args.iter_size == 0 and (timeit.default_timer() - start_train_time) > args.saving_last_time:
            state = {'iter': i+1,
                     'model_state': model.state_dict(),#}
                     'optimizer_state': optimizer.state_dict(),}
            torch.save(state, "checkpoints/{}_{}_{}_{}x{}_{}-{}_model.pth".format(args.arch, args.dataset, i+1, args.img_rows, args.img_cols, args.fold_num, args.num_folds))
            return

    print('best_dice: {:.5f}; best_wacc: {:.5f}'.format(best_dice, best_wacc))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='bisenet-resnet18',
                        help='Architecture to use [\'fcn8s, unet, segnet, pspnet, icnet, etc\']')
    parser.add_argument('--norm_type', nargs='?', type=str, default='gn',
                        help='Architecture to use [\'BN, GN, etc\']')
    parser.add_argument('--dataset', nargs='?', type=str, default='siim',
                        help='Dataset to use [\'pascal, camvid, ade20k, cityscapes, etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=1024,
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=1024,
                        help='Width of the input image')
    parser.add_argument('--in_channels', nargs='?', type=int, default=3,
                        help='Input image channels')
    parser.add_argument('--mask_dilation_size', nargs='?', type=int, default=1,
                        help='Mask dilation size (for training)')
    parser.add_argument('--device', nargs='?', type=str, default='cuda',
                        help='Device to use [\'cuda, cpu, etc\']')

    parser.add_argument('--n_iter', nargs='?', type=int, default=90000,
                        help='# of the iters')
    parser.add_argument('--batch_size', nargs='?', type=int, default=12,
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-3,
                        help='Learning Rate')
    parser.add_argument('--momentum', nargs='?', type=float, default=0.9,
                        help='Momentum (SGD)')
    parser.add_argument('--weight_decay', nargs='?', type=float, default=1e-4,
                        help='Weight Decay')
    parser.add_argument('--optimizer_type', nargs='?', type=str, default='radam',
                        help='Optimizer to use [\'SGD, Adam, RAdam, etc\']')
    parser.add_argument('--iter_size', nargs='?', type=int, default=1,
                        help='Accumulated batch gradient size')
    parser.add_argument('--resume', nargs='?', type=str, default=None,   
                        help='Path to previous saved model to restart from')

    parser.add_argument('--only_non_empty', dest='only_non_empty', action='store_true',
                        help='Enable to load only non-empty mask images | False by default')
    parser.add_argument('--no-only_non_empty', dest='only_non_empty', action='store_false',
                        help='Disable to load only non-empty mask images | False by default')
    parser.set_defaults(only_non_empty=False)

    parser.add_argument('--ratio', nargs='?', type=float, default=0.5,
                        help='Empty mask ratio in each batch')

    parser.add_argument('--weight_acc_non_empty', nargs='?', type=float, default=1.0,
                        help='Weight for accuracy of non-empty mask images')

    parser.add_argument('--use_cbam', dest='use_cbam', action='store_true',
                        help='Enable to use CBAM | False by default')
    parser.add_argument('--no-use_cbam', dest='use_cbam', action='store_false',
                        help='Disable to use CBAM | False by default')
    parser.set_defaults(use_cbam=False)

    parser.add_argument('--load_pretrained', dest='load_pretrained', action='store_true',
                        help='Enable to load pretrained model | True by default')
    parser.add_argument('--no-load_pretrained', dest='load_pretrained', action='store_false',
                        help='Disable to load pretrained model | True by default')
    parser.set_defaults(load_pretrained=True)

    parser.add_argument('--seed', nargs='?', type=int, default=1234, 
                        help='Random seed')
    parser.add_argument('--num_cycles', nargs='?', type=int, default=1, 
                        help='Cosine Annealing Cyclic LR')

    parser.add_argument('--alpha', nargs='?', type=float, default=0.25,
                        help='Alpha for focal loss')
    parser.add_argument('--gamma', nargs='?', type=float, default=2.0,
                        help='Gamma for focal loss')

    parser.add_argument('--lambda_fl', nargs='?', type=float, default=1.0,
                        help='lambda_fl')
    parser.add_argument('--lambda_dc', nargs='?', type=float, default=0.0,
                        help='lambda_dc')
    parser.add_argument('--lambda_lv', nargs='?', type=float, default=0.0,
                        help='lambda_lv')
    parser.add_argument('--lambda_cls', nargs='?', type=float, default=0.0,
                        help='lambda_cls')

    parser.add_argument('--dice_weight0', nargs='?', type=float, default=1.0,
                        help='dice_weight for empty')
    parser.add_argument('--dice_weight1', nargs='?', type=float, default=1.0,
                        help='dice_weight for non-empty')
    parser.add_argument('--lv_margin0', nargs='?', type=float, default=1.0,
                        help='lv_margin for empty')
    parser.add_argument('--lv_margin1', nargs='?', type=float, default=1.0,
                        help='lv_margin for non-empty')

    parser.add_argument('--fold_num', nargs='?', type=int, default=0,
                        help='Fold number in each class for training')
    parser.add_argument('--num_folds', nargs='?', type=int, default=5,
                        help='Number of folds for training')
    parser.add_argument('--print_train_freq', nargs='?', type=int, default=100,
                        help='Frequency (iterations) of training logs display')
    parser.add_argument('--eval_freq', nargs='?', type=int, default=1,
                        help='Frequency (iters) of evaluation of current model')
    parser.add_argument('--save_freq', nargs='?', type=float, default=1.0,
                        help='Frequency (iters) of saving current model (divided by eval_freq)')
    parser.add_argument('--saving_last_time', nargs='?', type=int, default=-1,
                        help='Last iters of saving current model (seconds)')

    parser.add_argument('--start_iter', nargs='?', type=int, default=-1,
                        help='Starting iteration number (-1 to ignore)')

    args = parser.parse_args()
    print(args)
    train(args)
