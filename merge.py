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

from torch.backends import cudnn
from torch.utils import data
from tqdm import tqdm

from models import get_model
from loaders import get_loader, get_data_path
from misc.utils import convert_state_dict, AverageMeter
from misc.metrics import runningScore

import glob

def merge(args):
    if not os.path.exists(args.root_results):
        os.makedirs(args.root_results)

    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    loader = data_loader(data_path, split=args.split, transforms=None, fold_num=0, num_folds=1, no_gt=args.no_gt, seed=args.seed, no_load_images=True)

    n_classes = loader.n_classes
    testloader = data.DataLoader(loader, batch_size=args.batch_size)#, num_workers=2, pin_memory=True)

    avg_y_prob = np.zeros((loader.__len__(), 1, 1024, 1024), dtype=np.float32)
    avg_y_pred_sum = np.zeros((loader.__len__(),), dtype=np.int32)
    fold_list = []
    for prob_file_name in glob.glob(os.path.join(args.root_results, '*.npy')):
        prob = np.load(prob_file_name, mmap_mode='r')
        for i in range(loader.__len__()):
            avg_y_prob[i, :, :, :] += prob[i, :, :, :]
        fold_list.append(prob_file_name)
        print(prob_file_name)
    avg_y_prob = avg_y_prob / len(fold_list)
    ##avgprob_file_name = 'prob_{}_avg'.format(len(fold_list))
    ##np.save(os.path.join(args.root_results, '{}.npy'.format(avgprob_file_name)), avg_y_prob)

    avg_y_pred = (avg_y_prob > args.thresh).astype(np.int)
    avg_y_pred_sum = avg_y_pred.sum(3).sum(2).sum(1)

    avg_y_pred_sum_argsorted = np.argsort(avg_y_pred_sum)[::-1]
    pruned_idx = int(avg_y_pred_sum_argsorted.shape[0]*args.non_empty_ratio)
    mask_sum_thresh = int(avg_y_pred_sum[avg_y_pred_sum_argsorted[pruned_idx]]) if pruned_idx < avg_y_pred_sum_argsorted.shape[0] else 0

    running_metrics = runningScore(n_classes=2, weight_acc_non_empty=args.weight_acc_non_empty)

    pred_dict = collections.OrderedDict()
    num_non_empty_masks = 0
    for i, (_, labels, names) in tqdm(enumerate(testloader)):
        labels = labels.cuda()

        prob = avg_y_prob[i*args.batch_size:i*args.batch_size+labels.size(0), :, :, :]
        pred = (prob > args.thresh).astype(np.int)
        pred = torch.from_numpy(pred).long().cuda()

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

        for k in range(labels.size(0)):
            name = names[0][k]
            if pred_dict.get(name, None) is None:
                mask = pred[k, 0, :, :].cpu().numpy()
                rle = loader.mask2rle(mask)
                pred_dict[name] = rle

    print('# non-empty masks: {:5d} (non_empty_ratio: {:.5f} / mask_sum_thresh: {:6d})'.format(num_non_empty_masks, args.non_empty_ratio, mask_sum_thresh))
    if not args.no_gt:
        dice, dice_empty, dice_non_empty, miou, acc, acc_empty, acc_non_empty = running_metrics.get_scores()
        print('Dice (per image): {:.5f} (empty: {:.5f} / non-empty: {:.5f})'.format(dice, dice_empty, dice_non_empty))
        print('Classification accuracy: {:.5f} (empty: {:.5f} / non-empty: {:.5f})'.format(acc, acc_empty, acc_non_empty))
        print('Overall mIoU: {:.5f}'.format(miou))
    running_metrics.reset()

    # Create submission
    csv_file_name = 'merged_{}_{}_{}_{}'.format(args.split, len(fold_list), args.thresh, args.non_empty_ratio)
    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['ImageId']
    sub.columns = ['EncodedPixels']
    sub.to_csv(os.path.join(args.root_results, '{}.csv'.format(csv_file_name)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--dataset', nargs='?', type=str, default='siim',
                        help='Dataset to use [\'pascal, camvid, ade20k, cityscapes, etc\']')

    parser.add_argument('--batch_size', nargs='?', type=int, default=1,
                        help='Batch Size')
    parser.add_argument('--split', nargs='?', type=str, default='test',
                        help='Split of dataset to test on')

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

    args = parser.parse_args()
    print(args)
    merge(args)
