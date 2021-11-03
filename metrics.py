import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def roc_auc_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor) -> float:

    y_true = y_targets.detach().cpu().numpy()
    y_pred = y_preds.detach().cpu().numpy()
    y_pred = np.transpose([pred[:, 1] for pred in y_pred])
    return roc_auc_score(y_true, y_pred)