from torch.optim import Adam
from torch.utils.data import DataLoader
import argparse

from few_shot.datasets import OmniglotDataset, MiniImageNet
from few_shot.models import get_few_shot_encoder
from few_shot.core import NShotTaskSampler, EvaluateFewShot, prepare_nshot_task
from few_shot.proto import proto_net_episode
from few_shot.train import fit
from few_shot.callbacks import *
from few_shot.utils import setup_dirs
from config import PATH
from few_shot.eval_proto import evaluate
import torch
import random

import os


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(42)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='omniglot')

parser.add_argument('--n', default=1, type=int)
parser.add_argument('--k', default=5, type=int)
parser.add_argument('--q', default=1, type=int)
parser.add_argument(
    '--path', default='save_models/proto_nets/omniglot_nt=1_kt=5_qt=5_nv=1_kv=5_qv=1.pth', type=str)
args = parser.parse_args()

evaluation_episodes = 1000
episodes_per_epoch = 1000
n_test = args.n
k_test = args.k
q_test = args.q
PATH = args.path
dataset = args.dataset
device = 'cuda'

if args.dataset == 'omniglot':
    dataset_class = OmniglotDataset
    num_input_channels = 1

elif args.dataset == 'miniImageNet':
    dataset_class = MiniImageNet
    num_input_channels = 3

model = get_few_shot_encoder(num_input_channels)
model.to(device, dtype=torch.double)

model.load_state_dict(torch.load(PATH))
model.eval()


evaluation = dataset_class('evaluation')
evaluation_taskloader = DataLoader(
    evaluation,
    batch_sampler=NShotTaskSampler(
        evaluation, episodes_per_epoch, n_test, k_test, q_test),
    num_workers=4
)

loss_fn = torch.nn.NLLLoss().cuda()


logs = evaluate(model, evaluation_taskloader, prepare_batch=prepare_nshot_task(
    n_test, k_test, q_test), metrics=['categorical_accuracy'], prefix='test_', loss_fn=loss_fn, eval_fn=proto_net_episode, n=n_test, k=k_test, q=q_test)


print(logs)
