import os
import cv2
import torch
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
import pydicom
import torchvision.transforms.functional as TF
import random
from skimage import exposure

from torch.utils import data

from misc.utils import recursive_glob


class siimLoader(data.Dataset):
    def __init__(self, root, split="train",
                 transforms=None, augmentations=False, in_channels=3, mask_dilation_size=1,
                 no_gt=False, fold_num=0, num_folds=1, seed=1234, no_load_images=False, only_non_empty=False):
        self.root = root
        self.split = split
        self.transforms = transforms
        self.augmentations = augmentations
        self.in_channels = in_channels
        self.mask_dilation_size = mask_dilation_size
        self.no_gt = no_gt
        self.n_classes = 2
        self.no_load_images = no_load_images
        self.only_non_empty = only_non_empty
        self.files = {}
        self.files_empty = {}
        self.files_non_empty = {}

        if self.split != 'test':
            ##train_df = pd.read_csv(os.path.join(self.root, self.split.replace('val', 'train') + '-rle_unique.csv'), index_col=0)
            train_df = pd.read_csv(os.path.join(self.root, 'stage_2_' + self.split.replace('val', 'train') + '_unique.csv'), index_col=0)
            self.train_labels = train_df.to_dict('index')
        else:
            ##test_df = pd.read_csv(os.path.join(self.root, 'sample_submission.csv'), index_col=0)
            test_df = pd.read_csv(os.path.join(self.root, 'stage_2_sample_submission.csv'), index_col=0)
            self.test_labels = test_df.to_dict('index')

        ##root_img_path = os.path.join(self.root, 'dicom-images-' + self.split.replace('val', 'train'))
        root_img_path = os.path.join(self.root, 'stage_{}_images'.format(1 if self.split != 'test' else 2))
        fs = recursive_glob(rootdir=root_img_path, suffix='.dcm')

        if self.split != 'test':
            fs = [f for f in fs if os.path.basename(f)[:-4] in self.train_labels]
            N = len(fs)
            start_idx = N * fold_num // num_folds
            end_idx = N * (fold_num + 1) // num_folds
            print('{:5s}: {:2d}/{:2d} [{:6d}, {:6d}] - {:6d}'.format(self.split, fold_num, num_folds, start_idx, end_idx, N))
            self.files[self.split] = []
            self.files_empty[self.split] = []
            self.files_non_empty[self.split] = []
            torch.manual_seed(seed)
            rp = torch.randperm(N).tolist()
            for i in range(N):
                f = fs[rp[i]]
                lbl = self.train_labels[os.path.basename(f)[:-4]]['EncodedPixels']
                if ((i >= start_idx and i < end_idx) and self.split == 'val') or (not (i >= start_idx and i < end_idx) and self.split == 'train') or (num_folds == 1):
                    self.__append_files__(f, lbl)
        else:
            fs = [f for f in fs if os.path.basename(f)[:-4] in self.test_labels]
            self.files[self.split] = fs

        ##self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (self.split, self.root))
        else:
            print("Found %d %s images" % (len(self.files[self.split]), self.split))

    def __append_files__(self, f, lbl):
        if '-1' in lbl:
            self.files_empty[self.split].append(f)
            if self.only_non_empty:
                return
        else:
            self.files_non_empty[self.split].append(f)
        self.files[self.split].append(f)

    def __len__(self):
        return len(self.files[self.split])

    def __gen_batchs__(self, batch_size, ratio=0.5): # dataloader: shuffle=False
        if self.only_non_empty: return
        np.random.shuffle(self.files_empty[self.split])
        np.random.shuffle(self.files_non_empty[self.split])
        num_empty = len(self.files_empty[self.split])
        num_non_empty = len(self.files_non_empty[self.split])
        batch_empty = int(batch_size * ratio)
        batch_non_empty = batch_size - batch_empty
        repeat_non_empty_times = int((num_empty * batch_empty) / float(num_non_empty * batch_non_empty) + 1)
        tmp_files_non_empty = self.files_non_empty[self.split] * repeat_non_empty_times
        self.files[self.split] = []
        for i in range(int(num_empty / batch_size)):
            tmp_fs = self.files_empty[self.split][i*batch_empty:(i+1)*batch_empty] + tmp_files_non_empty[i*batch_non_empty:(i+1)*batch_non_empty]
            np.random.shuffle(tmp_fs)
            self.files[self.split].extend(tmp_fs)
        print('New batchs: {}/{} (batch_empty: {} / batch_non_empty: {})'.format(len(self.files[self.split]), batch_size, batch_empty, batch_non_empty))

    def __getitem__(self, index):
        img_path = self.files[self.split][index].rstrip()
        img_id = os.path.basename(img_path)[:-4]

        lbl = np.zeros((1,1,1), dtype=np.float) - 1.
        if not self.no_gt:
            lbl = self.rle2mask(self.train_labels[img_id]['EncodedPixels'])
            lbl = np.array(lbl, dtype=np.uint8)
            if self.mask_dilation_size > 1:
                kernel = np.ones((self.mask_dilation_size, self.mask_dilation_size), dtype=np.uint8)
                lbl = cv2.dilate(lbl, kernel, iterations=1)
            lbl = np.expand_dims(lbl, axis=2)

        if not self.no_load_images:
            dcm_data = pydicom.dcmread(img_path)
            img = dcm_data.pixel_array
            ##img = self.clahe.apply(img)
            img = np.array(img, dtype=np.uint8)
            img = np.expand_dims(img, axis=2)
            img = img if self.in_channels == 1 else np.tile(img, (1, 1, self.in_channels))
            ##img = (exposure.equalize_adapthist(img) * 255.).astype(np.uint8) # contrast correction

            if self.augmentations:
                # random crop then resize
                h, w = img.shape[:2]
                new_h, new_w = random.randrange(int(float(h)*7/8), h+1), random.randrange(int(float(w)*7/8), w+1)
                idx_h, idx_w = random.randrange(h-new_h+1), random.randrange(w-new_w+1)
                img = img[idx_h:idx_h+new_h, idx_w:idx_w+new_w, :]
                lbl = lbl[idx_h:idx_h+new_h, idx_w:idx_w+new_w, :]

                if random.random() > 0.5: # random hflip
                    img = np.flip(img, axis=1)
                    if not self.no_gt:
                        lbl = np.flip(lbl, axis=1)

            if self.transforms is not None:
                img = self.transforms(img)
        else:
            img = torch.zeros(self.in_channels,1,1, dtype=torch.float)

        if not self.no_gt:
            lbl = torch.from_numpy(np.copy(lbl.transpose(2, 0, 1))).float()
            if lbl.shape[1:] != (1024, 1024):
                lbl = F.interpolate(lbl.unsqueeze(0), size=(1024, 1024), mode='nearest').squeeze(0)

        return img, lbl, [img_id]

    def mask2rle(self, mask, width=1024, height=1024, maskColor=1):
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

    def rle2mask(self, rle, width=1024, height=1024, maskColor=1):
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



if __name__ == '__main__':
    from tqdm import tqdm
    from torchvision import transforms
    local_path = '../datasets/siim/'

    data_trans = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(size=1024),
                    transforms.CenterCrop(size=(1024, 1024)),
                    transforms.ToTensor(),
                ])

    num_folds = 1
    for fold_num in range(num_folds):
        dst = siimLoader(local_path, split="train", transforms=data_trans, fold_num=fold_num, num_folds=num_folds)
    ##dst = siimLoader(local_path, split="test", fold_num=0, num_folds=num_folds)

    """
    loader = data.DataLoader(dst, batch_size=1)
    lbl_sum = torch.zeros(dst.__len__()).float().cuda()
    for i, data in tqdm(enumerate(loader)):
        lbl_sum[i] = data[1].cuda().sum()
        continue
    print('{:10.3f} {:10.3f} {:10.3f} {:10.3f}'.format(lbl_sum.min(), lbl_sum.max(), lbl_sum.mean(), lbl_sum.std()))
    #      0.000 161421.000   3196.878   9874.594 # stage-1 train
    print('{:10.3f} {:10.3f} {:10.3f} {:10.3f}'.format(lbl_sum[lbl_sum > 0].min(), lbl_sum[lbl_sum > 0].max(), lbl_sum[lbl_sum > 0].mean(), lbl_sum[lbl_sum > 0].std()))
    #     55.000 161421.000  14344.965  16664.062 # stage-1 train
    #"""
