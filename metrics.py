import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from torch import Tensor

# dice computation function from https://github.com/milesial/Pytorch-UNet/blob/8f317cb13c17ef25a86b25a0c24390e04cd4db82/utils/dice_score.py#L26

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size() 
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]

def dice_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor) -> float:
    # only calculate the foreground class
    y_preds = y_preds[:,[1],...]
    # y_targets = y_targets[:,[1],...]
    print(y_preds.shape,y_targets.shape)

    return multiclass_dice_coeff(y_preds,y_targets)

def roc_auc_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor) -> float:

    y_true = y_targets.detach().cpu().numpy()
    y_pred = y_preds.detach().cpu().numpy()
    # print(y_pred.shape,y_true.shape,y_pred)
    y_pred = y_pred[:,1]#np.array([pred[1] for pred in y_pred])
    # print(y_pred.shape,y_true)
    return roc_auc_score(y_true, y_pred)

def accuracy_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor) -> float:

    y_true = y_targets.detach().cpu().numpy()
    y_preds = torch.argmax(y_preds, dim=1)
    y_pred = y_preds.detach().cpu().numpy()
    # print(y_pred.shape,y_true.shape,y_pred)
    # y_pred = y_pred[:,1]#np.array([pred[1] for pred in y_pred])
    # print(y_pred.shape,y_true)
    return accuracy_score(y_true, y_pred)