import torch

class runningScore(object):
    def __init__(self, n_classes=2, weight_acc_non_empty=1.0, device='cuda'):
        self.n_classes = n_classes
        self.weight_acc_non_empty = weight_acc_non_empty
        self.device = device
        self.confusion_matrix_each = []
        self.confusion_matrix_all = torch.zeros(n_classes, n_classes, device=torch.device(self.device)).long()

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) * (label_true < self.n_classes)
        if mask.sum() == 0: return 0 # pass
        hist = torch.bincount(
            self.n_classes * label_true[mask] +
            label_pred[mask], minlength=self.n_classes**2).reshape(self.n_classes, self.n_classes)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            cm = self._fast_hist(lt.view(-1), lp.view(-1))
            self.confusion_matrix_each.append(cm)
            self.confusion_matrix_all += cm

    def get_scores(self):
        hist_each = torch.stack(self.confusion_matrix_each).float()

        num_empty = 0.0
        dice_empty = torch.tensor(0.0).to(torch.device(self.device))
        dice_non_empty = torch.tensor(0.0).to(torch.device(self.device))
        acc_empty = torch.tensor(0.0).to(torch.device(self.device))
        acc_non_empty = torch.tensor(0.0).to(torch.device(self.device))
        for i in range(hist_each.size(0)):
            d = 2 * hist_each[i,1,1] / (hist_each[i,0,1] + hist_each[i,1,0] + hist_each[i,1,1]*2) if (hist_each[i,0,1] + hist_each[i,1,0] + hist_each[i,1,1]*2) > 0 else 1.
            if hist_each[i,1,0] + hist_each[i,1,1] == 0: # empty (for GT == 1)
                num_empty += 1
                dice_empty += d
                acc_empty += d
            else:#if hist_each[i,1,0] + hist_each[i,1,1] > 0: # non-empty
                dice_non_empty += d
                acc_non_empty += ((hist_each[i,0,1] + hist_each[i,1,1]) > 0).float() # (pred == 1).sum() > 0

        dice = dice_empty + dice_non_empty
        dice /= max(hist_each.size(0), 1)
        dice_empty /= max(num_empty, 1)
        dice_non_empty /= max((hist_each.size(0) - num_empty), 1)

        wacc = acc_empty + self.weight_acc_non_empty * acc_non_empty
        wacc /= max(hist_each.size(0), 1)
        acc_empty /= max(num_empty, 1)
        acc_non_empty /= max((hist_each.size(0) - num_empty), 1)

        hist_all = self.confusion_matrix_all.float()
        cls_iou = torch.diag(hist_all) / (hist_all.sum(dim=1) + hist_all.sum(dim=0) - torch.diag(hist_all))
        miou = (cls_iou[cls_iou >= 0]).mean()
        return dice.cpu().numpy(), dice_empty.cpu().numpy(), dice_non_empty.cpu().numpy(), miou.cpu().numpy(), wacc.cpu().numpy(), acc_empty.cpu().numpy(), acc_non_empty.cpu().numpy()

    def reset(self):
        self.confusion_matrix_each = []
        self.confusion_matrix_all = torch.zeros(self.n_classes, self.n_classes, device=torch.device(self.device)).long()
