import torch
import os
import shutil
from typing import Tuple, List

from config import EPSILON, PATH


def mkdir(dir):

    try:
        os.mkdir(dir)
    except:
        pass


def rmdir(dir):

    try:
        shutil.rmtree(dir)
    except:
        pass


def setup_dirs():
    mkdir(PATH + '/log_files/')
    mkdir(PATH + '/log_files/proto_nets')
    mkdir(PATH + '/log_files/matching_nets')
    mkdir(PATH + '/log_files/maml')
    mkdir(PATH + '/save_models/')
    mkdir(PATH + '/save_models/proto_nets')
    mkdir(PATH + '/save_models/matching_nets')
    mkdir(PATH + '/save_models/maml')


def pairwise_distances(x: torch.Tensor,
                       y: torch.Tensor,
                       matching_fn: str) -> torch.Tensor:

    n_x = x.shape[0]
    n_y = y.shape[0]

    if matching_fn == 'l2':
        distances = (
            x.unsqueeze(1).expand(n_x, n_y, -1) -
            y.unsqueeze(0).expand(n_x, n_y, -1)
        ).pow(2).sum(dim=2)
        return distances
    elif matching_fn == 'cosine':
        normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)
        normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)

        expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

        cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
        return 1 - cosine_similarities
    elif matching_fn == 'dot':
        expanded_x = x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = y.unsqueeze(0).expand(n_x, n_y, -1)

        return -(expanded_x * expanded_y).sum(dim=2)
    else:
        raise(ValueError('Unsupported similarity function'))


def copy_weights(from_model: torch.nn.Module, to_model: torch.nn.Module):

    if not from_model.__class__ == to_model.__class__:
        raise(ValueError("Models don't have the same architecture!"))

    for m_from, m_to in zip(from_model.modules(), to_model.modules()):
        is_linear = isinstance(m_to, torch.nn.Linear)
        is_conv = isinstance(m_to, torch.nn.Conv2d)
        is_bn = isinstance(m_to, torch.nn.BatchNorm2d)
        if is_linear or is_conv or is_bn:
            m_to.weight.data = m_from.weight.data.clone()
            if m_to.bias is not None:
                m_to.bias.data = m_from.bias.data.clone()


def autograd_graph(tensor: torch.Tensor) -> Tuple[
    List[torch.autograd.Function],
    List[Tuple[torch.autograd.Function, torch.autograd.Function]]
]:

    nodes, edges = list(), list()

    def _add_nodes(tensor):
        if tensor not in nodes:
            nodes.append(tensor)

            if hasattr(tensor, 'next_functions'):
                for f in tensor.next_functions:
                    if f[0] is not None:
                        edges.append((f[0], tensor))
                        _add_nodes(f[0])

            if hasattr(tensor, 'saved_tensors'):
                for t in tensor.saved_tensors:
                    edges.append((t, tensor))
                    _add_nodes(t)

    _add_nodes(tensor.grad_fn)

    return nodes, edges
