"""
Reproduce Matching Network results of Vinyals et al
"""
from few_shot.models import MatchingNetwork
import argparse
from torch.utils.data import DataLoader
from torch.optim import Adam

from few_shot.datasets import OmniglotDataset, MiniImageNet
from few_shot.core import NShotTaskSampler, prepare_nshot_task, EvaluateFewShot
from few_shot.matching import matching_net_episode
from few_shot.train import fit
from few_shot.callbacks import *
from few_shot.utils import setup_dirs
from config import PATH
from few_shot.eval_matching import evaluate
import random
import torch
import os
setup_dirs()
assert torch.cuda.is_available()
device = torch.device('cuda')


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
parser.add_argument('--lstm-layers', default=1, type=int)
parser.add_argument('--unrolling-steps', default=2, type=int)
parser.add_argument(
    '--path', default='save_models/matching_nets/omniglot_n=1_k=5_q=5_nv=1_kv=5_qv=1_dist=cosine_fce=None.pth', type=str)
args = parser.parse_args()

n_test = args.n
k_test = args.k
q_test = args.q
fce = None
lstm_layers = args.unrolling_steps
unrolling_steps = args.unrolling_steps
evaluation_episodes = 1000
episodes_per_epoch = 1000
device = 'cuda'
dataset = args.dataset
PATH = args.path


if args.dataset == 'omniglot':
    dataset_class = OmniglotDataset
    num_input_channels = 1
    lstm_input_size = 64
elif args.dataset == 'miniImageNet':
    dataset_class = MiniImageNet
    num_input_channels = 3
    lstm_input_size = 1600


model = MatchingNetwork(n_test, k_test, q_test, None, num_input_channels,
                        lstm_layers=lstm_layers,
                        lstm_input_size=lstm_input_size,
                        unrolling_steps=unrolling_steps,
                        device=device)
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
    n_test, k_test, q_test), metrics=['categorical_accuracy'], prefix='test_', loss_fn=loss_fn, eval_fn=matching_net_episode, n=n_test, k=k_test, q=q_test)


print(logs)
