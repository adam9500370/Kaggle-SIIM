import sys, os
import cv2
import torch
import argparse
import timeit
import random
import collections
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.backends import cudnn
from torch.utils import data
from tqdm import tqdm

from models import get_model
from loaders import get_loader, get_data_path
from misc.utils import convert_state_dict, AverageMeter
from misc.metrics import runningScore

cudnn.benchmark = True
cudnn.deterministic = True

def test(args):
    if not os.path.exists(args.root_results):
        os.makedirs(args.root_results)

    model_file_name = os.path.split(args.model_path)[1]
    model_name = model_file_name[:model_file_name.find('_')]

    # Setup Transforms
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
    loader = data_loader(data_path, split=args.split, in_channels=args.in_channels, transforms=data_trans, fold_num=args.fold_num, num_folds=args.num_folds, no_gt=args.no_gt, seed=args.seed)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    n_classes = loader.n_classes
    testloader = data.DataLoader(loader, batch_size=args.batch_size)#, num_workers=2, pin_memory=True)

    # Setup Model
    model = get_model(model_name, n_classes=1, in_channels=args.in_channels, norm_type=args.norm_type, use_cbam=args.use_cbam)
    model.cuda()

    checkpoint = torch.load(args.model_path)#, encoding="latin1")
    state = convert_state_dict(checkpoint['model_state'])
    model_dict = model.state_dict()
    model_dict.update(state)
    model.load_state_dict(model_dict)

    saved_iter = checkpoint.get('iter', -1)
    dice_val = checkpoint.get('dice', -1)
    wacc_val = checkpoint.get('wacc', -1)
    print("Loaded checkpoint '{}' (iter {}, dice {:.5f}, wAcc {:.5f})"
           .format(args.model_path, saved_iter, dice_val, wacc_val))

    running_metrics = runningScore(n_classes=2, weight_acc_non_empty=args.weight_acc_non_empty)

    y_prob = np.zeros((loader.__len__(), 1, 1024, 1024), dtype=np.float32)
    y_pred_sum = np.zeros((loader.__len__(),), dtype=np.int32)
    pred_dict = collections.OrderedDict()
    num_non_empty_masks = 0
    model.eval()
    with torch.no_grad():
        for i, (images, labels, _) in tqdm(enumerate(testloader)):
            images = images.cuda()
            labels = labels.cuda()
            if args.tta:
                bs, c, h, w = images.size()
                images = torch.cat([images, torch.flip(images, dims=[3])], dim=0) # hflip

            outputs = model(images, return_aux=False)
            prob = F.sigmoid(outputs)
            if args.tta:
                prob = prob.view(-1, bs, 1, h, w)
                prob[1, :, :, :, :] = torch.flip(prob[1, :, :, :, :], dims=[3])
                prob = prob.mean(0)
            pred = (prob > args.thresh).long()
            pred_sum = pred.sum(3).sum(2).sum(1)
            y_prob[i*args.batch_size:i*args.batch_size+labels.size(0), :, :, :] = prob.cpu().numpy()
            y_pred_sum[i*args.batch_size:i*args.batch_size+labels.size(0)] = pred_sum.cpu().numpy()

        y_pred_sum_argsorted = np.argsort(y_pred_sum)[::-1]
        pruned_idx = int(y_pred_sum_argsorted.shape[0]*args.non_empty_ratio)
        mask_sum_thresh = int(y_pred_sum[y_pred_sum_argsorted[pruned_idx]]) if pruned_idx < y_pred_sum_argsorted.shape[0] else 0

        for i, (_, labels, names) in tqdm(enumerate(testloader)):
            labels = labels.cuda()

            prob = torch.from_numpy(y_prob[i*args.batch_size:i*args.batch_size+labels.size(0), :, :, :]).float().cuda()
            pred = (prob > args.thresh).long()
            pred_sum = pred.sum(3).sum(2).sum(1)
            for k in range(labels.size(0)):
                if pred_sum[k] > mask_sum_thresh:
                    num_non_empty_masks += 1
                else:
                    pred[k, :, :, :] = torch.zeros_like(pred[k, :, :, :])
                    if args.only_non_empty:
                        pred[k, :, 0, 0] = 1

            if not args.no_gt:
                running_metrics.update(labels.long(), pred.long())

            """
            if args.split == 'test':
                for k in range(labels.size(0)):
                    name = names[0][k]
                    if pred_dict.get(name, None) is None:
                        mask = pred[k, 0, :, :].cpu().numpy()
                        rle = loader.mask2rle(mask)
                        pred_dict[name] = rle
            #"""

    print('# non-empty masks: {:5d} (non_empty_ratio: {:.5f} / mask_sum_thresh: {:6d})'.format(num_non_empty_masks, args.non_empty_ratio, mask_sum_thresh))
    if not args.no_gt:
        dice, dice_empty, dice_non_empty, miou, wacc, acc_empty, acc_non_empty = running_metrics.get_scores()
        print('Dice (per image): {:.5f} (empty: {:.5f} / non-empty: {:.5f})'.format(dice, dice_empty, dice_non_empty))
        print('wAcc: {:.5f} (empty: {:.5f} / non-empty: {:.5f})'.format(wacc, acc_empty, acc_non_empty))
        print('Overall mIoU: {:.5f}'.format(miou))
    running_metrics.reset()

    if args.split == 'test':
        fold_num, num_folds = model_file_name.split('_')[4].split('-')
        prob_file_name = 'prob-{}_{}x{}_{}_{}_{}-{}'.format(args.split, args.img_rows, args.img_cols, model_name, saved_iter, fold_num, num_folds)
        np.save(os.path.join(args.root_results, '{}.npy'.format(prob_file_name)), y_prob)

        """
        # Create submission
        csv_file_name = '{}_{}x{}_{}_{}_{}-{}_{}_{}'.format(args.split, args.img_rows, args.img_cols, model_name, saved_iter, fold_num, num_folds, args.thresh, args.non_empty_ratio)
        sub = pd.DataFrame.from_dict(pred_dict, orient='index')
        sub.index.names = ['ImageId']
        sub.columns = ['EncodedPixels']
        sub.to_csv(os.path.join(args.root_results, '{}.csv'.format(csv_file_name)))
        #"""



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--model_path', nargs='?', type=str, default='bisenet-resnet18_siim_90000_0-5_model.pth',
                        help='Path to the saved model')
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

    parser.add_argument('--batch_size', nargs='?', type=int, default=1,
                        help='Batch Size')
    parser.add_argument('--split', nargs='?', type=str, default='test',
                        help='Split of dataset to test on')

    parser.add_argument('--fold_num', nargs='?', type=int, default=0,
                        help='Fold number in each class for training')
    parser.add_argument('--num_folds', nargs='?', type=int, default=5,
                        help='Number of folds for training')

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

    parser.add_argument('--no_gt', dest='no_gt', action='store_true',
                        help='Disable verification | True by default')
    parser.add_argument('--gt', dest='no_gt', action='store_false',
                        help='Enable verification | True by default')
    parser.set_defaults(no_gt=True)

    parser.add_argument('--seed', nargs='?', type=int, default=1234,
                        help='Random seed')

    parser.add_argument('--root_results', nargs='?', type=str, default='results',
                        help='Root path for saving results (.csv, .npy, etc)')

    parser.add_argument('--thresh', nargs='?', type=float, default=0.5,
                        help='Thresh for prediction')
    parser.add_argument('--non_empty_ratio', nargs='?', type=float, default=0.0,
                        help='Ratio for non-empty predictions')

    parser.add_argument('--only_non_empty', dest='only_non_empty', action='store_true',
                        help='Enable to load only non-empty mask images | False by default')
    parser.add_argument('--no-only_non_empty', dest='only_non_empty', action='store_false',
                        help='Disable to load only non-empty mask images | False by default')
    parser.set_defaults(only_non_empty=False)

    parser.add_argument('--weight_acc_non_empty', nargs='?', type=float, default=1.0,
                        help='Weight for accuracy of non-empty mask images')

    parser.add_argument('--tta', dest='tta', action='store_true',
                        help='Enable Test Time Augmentation (TTA) | False by default')
    parser.add_argument('--no-tta', dest='tta', action='store_false',
                        help='Disable Test Time Augmentation (TTA) | False by default')
    parser.set_defaults(tta=False)

    args = parser.parse_args()
    print(args)
    test(args)
