import matplotlib.pyplot as plt
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from typing import Callable, List, Union
from torch import optim
import numpy as np
from few_shot.metrics import NAMED_METRICS
from sklearn.metrics import confusion_matrix, f1_score
import scikitplot as skplt
from tqdm import tqdm


import pickle
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc


def multiclass_roc(y_test, y_score, n_classes=3):

    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # calculate dummies once
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return fpr, tpr, roc_auc


def evaluate(model: Module, dataloader: DataLoader, prepare_batch: Callable, metrics: List[Union[str, Callable]],
             loss_fn: Callable = None, prefix: str = 'val_', suffix: str = '', eval_fn: Callable = None, n: int = 5, k: int = 3, q: int = 5):
    """Evaluate a model on one or more metrics on a particular dataset

    # Arguments
        model: Model to evaluate
        dataloader: Instance of torch.utils.data.DataLoader representing the dataset
        prepare_batch: Callable to perform any desired preprocessing
        metrics: List of metrics to evaluate the model with. Metrics must either be a named metric (see `metrics.py`) or
            a Callable that takes predictions and ground truth labels and returns a scalar value
        loss_fn: Loss function to calculate over the dataset
        prefix: Prefix to prepend to the name of each metric - used to identify the dataset. Defaults to 'val_' as
            it is typical to evaluate on a held-out validation dataset
        suffix: Suffix to append to the name of each metric.
    """
    logs = {}
    seen = 0
    totals = {m: 0 for m in metrics}
    if loss_fn is not None:
        totals['loss'] = 0

    per_class_accuracies = []
    y_pred_list = []
    y_list = []
    for batch in tqdm(dataloader):
        x, y = prepare_batch(batch)

        loss, y_pred = eval_fn(
            model,
            optim.Adam(model.parameters()),
            loss_fn,
            x,
            y,
            n_shot=n,
            k_way=k,
            q_queries=q,
            train=False,
            order=1, inner_train_steps=1, inner_lr=0.4, device='cuda'
        )

        seen += y_pred.shape[0]

        # print('='*60)
        pred = torch.argmax(y_pred, dim=1).view(-1)
        matrix = confusion_matrix(
            y.cpu().detach().numpy(), pred.cpu().detach().numpy())
        pca = matrix.diagonal()/matrix.sum(axis=1)
        per_class_accuracies.append(pca)
        # print('='*60)

        y_pred_list.append(y_pred.cpu().detach().numpy())
        y_list.append(y.cpu().detach().numpy())

        totals['loss'] += loss.item() * y_pred.shape[0]

        for m in metrics:
            if isinstance(m, str):
                v = NAMED_METRICS[m](y, y_pred)
            else:
                # Assume metric is a callable function
                v = m(y, y_pred)
            totals[m] += v * y_pred.shape[0]

    for m in ['loss'] + metrics:
        logs[prefix + m + suffix] = totals[m] / seen

    return logs
