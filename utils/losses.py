import torch
import torch.nn.functional as F

def partical_COS(out_1, out_2, mask):
    loss = 1-torch.sum(torch.sum(F.normalize(out_1, dim=1) * F.normalize(out_2, dim=1), dim=1, keepdim=True) * mask)/torch.sum(mask)
    return loss

def partical_MAE(y_true, y_pred, mask):
    return torch.sum(torch.abs(y_true - y_pred) * mask)/torch.sum(mask)