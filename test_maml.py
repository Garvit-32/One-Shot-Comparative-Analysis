from torch.utils.data import DataLoader
from few_shot.datasets import OmniglotDataset, MiniImageNet
from few_shot.core import NShotTaskSampler, create_nshot_task_label
from few_shot.callbacks import *
from config import PATH
from few_shot.eval_maml import evaluate
from few_shot.models import FewShotClassifier
import torch.nn as nn
from few_shot.maml import meta_gradient_step
import torch
import random


import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='omniglot')
parser.add_argument('--n', default=1, type=int)
parser.add_argument('--k', default=5, type=int)
parser.add_argument('--q', default=1, type=int)
parser.add_argument('--meta-batch-size', default=32, type=int)
parser.add_argument('--path', type=str,
                    default='save_models/maml/omniglot_order=1_n=1_k=5_metabatch=32_train_steps=1_val_steps=3.pth')

dataset = args = parser.parse_args()

evaluation_episodes = 1000
episodes_per_epoch = 1000
n_test = args.n
k_test = args.k
q_test = args.q
meta_batch_size = args.meta_batch_size

device = 'cuda'
PATH = 'save_models/maml/omniglot_order=1_n=1_k=5_metabatch=32_train_steps=1_val_steps=3.pth'


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(42)


if dataset == 'omniglot':
    fc_layer_size = 64
    num_input_channels = 1
    dataset_class = OmniglotDataset
elif dataset == 'miniImageNet':
    dataset_class = MiniImageNet
    fc_layer_size = 1600
    num_input_channels = 3

meta_model = FewShotClassifier(
    num_input_channels, k_test, fc_layer_size).to(device, dtype=torch.double)

meta_model.load_state_dict(torch.load(PATH))
meta_model.eval()


evaluation = dataset_class('evaluation')
evaluation_taskloader = DataLoader(
    evaluation,
    batch_sampler=NShotTaskSampler(
        evaluation, episodes_per_epoch, n_test, k_test, q_test, num_tasks=meta_batch_size),
    num_workers=4
)
loss_fn = nn.CrossEntropyLoss().to(device)


def prepare_meta_batch(n, k, q, meta_batch_size):
    def prepare_meta_batch_(batch):
        x, y = batch
        x = x.reshape(meta_batch_size, n*k + q*k,
                      num_input_channels, x.shape[-2], x.shape[-1])
        # Move to device
        x = x.double().to(device)
        # Create label
        y = create_nshot_task_label(k, q).cuda().repeat(meta_batch_size)
        return x, y

    return prepare_meta_batch_


logs = evaluate(meta_model, evaluation_taskloader, prepare_batch=prepare_meta_batch(
    n_test, k_test, q_test, meta_batch_size), metrics=['categorical_accuracy'], prefix='test_', loss_fn=loss_fn, eval_fn=meta_gradient_step, n=n_test, k=k_test, q=q_test)


print(logs)
