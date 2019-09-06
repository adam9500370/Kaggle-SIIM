# Kaggle-SIIM
PyTorch implementation for [Kaggle SIIM-ACR Pneumothorax Segmentation Challenge](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation)

**42th / 1475** (top 3%)

Leaderboard scores (the mean of the Dice coefficients for each image in the test set):
* Stage-1
	- Without using leak: 0.8696
	- With using leak: 0.8742
* Stage-2
	- Without retraining: 0.8502
	- With retraining: 0.8512


## Method
* Small models
	- UNet [1] (ResNet18, 48 MB) for segmentation (10-fold)
	- BiSeNet [2] (ResNet18, 53 MB) for segmentation (10-fold) and classification (5-fold)
	- With auxiliary heads and dual attention [3] in both models
* Losses
	- Weighted binary cross-entropy
	- Symmetric Lovász-hinge [4] with margin
	- Additional binary cross-entropy for classification
* Optimizer: RAdam [5]
* Scheduler: CosineAnnealingLR (1 cycle)
* Augmentation: horizontal flip, random crop and resize
* Resolution: 768x768
* Segmentation part
	- Batch size: 6 * 4 (with gradient accumulation)
	- Using only pneumothorax data
* Classification part
	- Batch size: 6 * 8 (with gradient accumulation)
	- Using pneumothorax:non-pneumothorax = 50%:50% in each batch
	- Using dilated masks (128x128 kernel) to roughly locate pneumothorax region
* Without using external data and TTA


## Usage

### Modify the appropriate dataset path in `config.json`

### Training for each model
* `bash run_train_seg-bisenet.sh`
* `bash run_train_seg-unet.sh`
* `bash run_train_cls-bisenet.sh`

### Inference for each model
* `bash run_test_seg-bisenet.sh`
* `bash run_test_seg-unet.sh`
* `bash run_test_cls-bisenet.sh`

### Ensemble all the classification and segmentation models
* Submission1: non_empty_ratio = 0.17
	1. `bash run_merge_cls1.sh`
	2. `bash run_merge_seg1.sh`
* Submission2: non_empty_ratio = 0.18
	1. `bash run_merge_cls2.sh`
	2. `bash run_merge_seg2.sh`


## References
[1] [U-Net: Convolutional Networks for Biomedical Image Segmentation (U-Net)](https://arxiv.org/abs/1505.04597)

[2] [BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation](https://arxiv.org/abs/1808.00897)

[3] [Dual Attention Network for Scene Segmentation](https://arxiv.org/abs/1809.02983)

[4] [The Lovász-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks](https://arxiv.org/abs/1705.08790)

[5] [On the Variance of the Adaptive Learning Rate and Beyond](https://arxiv.org/abs/1908.03265)
