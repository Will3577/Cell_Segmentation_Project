import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

def roc_auc_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor) -> float:

    y_true = y_targets.detach().cpu().numpy()
    y_pred = y_preds.detach().cpu().numpy()
    print(y_pred.shape,y_true.shape,y_pred)
    y_pred = y_pred[:,1]#np.array([pred[1] for pred in y_pred])
    print(y_pred.shape,y_true)
    return roc_auc_score(y_true, y_pred)

def accuracy_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor) -> float:

    y_true = y_targets.detach().cpu().numpy()
    y_preds = torch.argmax(y_preds, dim=1)
    y_pred = y_preds.detach().cpu().numpy()
    # y_pred = 
    print(y_pred.shape,y_true.shape,y_pred)
    # y_pred = y_pred[:,1]#np.array([pred[1] for pred in y_pred])
    print(y_pred.shape,y_true)
    return accuracy_score(y_true, y_pred)