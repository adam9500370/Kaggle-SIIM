import numpy as np
import pandas as pd
import os
import collections
from tqdm import tqdm

def mask2rle(mask, width=1024, height=1024, maskColor=1):
    rle = []
    lastColor = 0
    currentPixel = 0
    runStart = -1
    runLength = 0

    if mask.sum() == 0:
        return "-1"
    elif mask.sum() == maskColor and mask[0][0] == maskColor:
        return "0 1"

    mask = mask.T
    for x in range(width):
        for y in range(height):
            currentColor = mask[x][y]
            if currentColor != lastColor:
                if currentColor == maskColor:
                    runStart = currentPixel
                    runLength = 1
                else:
                    rle.append(str(runStart))
                    rle.append(str(runLength))
                    runStart = -1
                    runLength = 0
                    currentPixel = 0
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor
            currentPixel += 1

    if lastColor == maskColor:
        rle.append(str(runStart))
        rle.append(str(runLength))

    return " ".join(rle) if len(rle) > 0 else "-1"


def rle2mask(rle, width=1024, height=1024, maskColor=1):
    mask = np.zeros(width * height, dtype=np.uint8)
    array = np.asarray([int(x) for x in rle.split()])
    if array.shape[0] <= 1:
        return mask.reshape(width, height).T
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = maskColor
        current_position += lengths[index]

    return mask.reshape(width, height).T

#"""
root_path = '../datasets/siim/'
##train_rle_df = pd.read_csv(os.path.join(root_path, 'train-rle.csv'), index_col=0)
train_rle_df = pd.read_csv(os.path.join(root_path, 'stage_2_train.csv'), index_col=0)
unique_idx = pd.unique(train_rle_df.index)

new_dict = collections.OrderedDict()
for idx in tqdm(unique_idx):
    mask = np.zeros((1024, 1024), dtype=np.uint8)
    for rle in train_rle_df[train_rle_df.index == idx]['EncodedPixels']:##[' EncodedPixels']:
        mask += rle2mask(rle)
    mask = (mask > 0).astype(np.uint8)
    new_rle = mask2rle(mask)
    new_dict[idx] = new_rle

new_csv = pd.DataFrame.from_dict(new_dict, orient='index')
new_csv.index.names = ['ImageId']
new_csv.columns = ['EncodedPixels']
##new_csv.to_csv(os.path.join(root_path, 'train-rle_unique.csv'))
new_csv.to_csv(os.path.join(root_path, 'stage_2_train_unique.csv'))
#"""
