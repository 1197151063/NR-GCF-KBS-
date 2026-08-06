import world
import torch
import numpy as np
from torch import Tensor, LongTensor
from torch.utils.data import DataLoader,WeightedRandomSampler
from torch_geometric.utils import bipartite_subgraph
from dataloader import Loader
import sys
import os
import logging
from datetime import datetime
import json
device = world.device
config = world.config
log_writer = None

def maybe_num_nodes(edge_index:Tensor):
    return edge_index[0].max()+1,edge_index[1].max()+1

def Full_Sampling(dataset:Loader):
    """
    With Normlized Sampling on Graph
    """
    train_edge_index = dataset.train_edge_index.to(device)
    num_items = dataset.num_items
    weights = dataset.sampling_weights
    batch_size = config['bpr_batch_size']
    mini_batch = []
    sampler = WeightedRandomSampler(
        weights,
        num_samples=train_edge_index.size(1)
    )
    train_loader = DataLoader(
        range(train_edge_index.size(1)),
        sampler=sampler,
        batch_size=batch_size
    )
    for index in train_loader:
        pos_edge_label_index = train_edge_index[:,index]
        neg_edge_label_index = torch.randint(0, num_items,(index.numel(), ), device=device)
        edge_label_index = torch.stack([
            pos_edge_label_index[0],
            pos_edge_label_index[1],
            neg_edge_label_index,
        ])
        mini_batch.append(edge_label_index)
    return mini_batch
    
def dropout_node_bipartite(
    edge_index: Tensor,
    p: float = 0.5,
    num_users = None,
    num_items = None,
    relabel_nodes: bool = False,
) -> Tensor:
    r"""Randomly drops nodes from bipartite graph
    :obj:`edge_index` with probability :obj:`p` using samples from
    a Bernoulli distribution.

    The method returns (1) the retained :obj:`edge_index`, (2) the edge mask
    indicating which edges were retained. (3) the node mask indicating
    which nodes were retained.

    Args:
        edge_index (LongTensor): The edge indices.
        p (float, optional): Dropout probability. (default: :obj:`0.5`)
        size: The number of two type of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index[0]` and `edge_index[1]`. 
        relabel_nodes (bool, optional): If set to `True`, the resulting
            `edge_index` will be relabeled to hold consecutive indices
            starting from zero.

    :rtype: (:class:`LongTensor`, :class:`BoolTensor`, :class:`BoolTensor`)
    """
    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')
    if num_users is None:
        num_users = edge_index[0].max() + 1
    if num_items is None:
        num_items = edge_index[1].max() + 1

    prob_users = torch.rand(num_users, device=edge_index.device)
    prob_items = torch.rand(num_items, device=edge_index.device)
    user_index = torch.arange(0,num_users,device=edge_index.device)
    item_index = torch.arange(0,num_items,device=edge_index.device)
    user_mask = prob_users > p
    item_mask = prob_items > p
    node_mask = (user_index[user_mask],item_index[item_mask])

    edge_index, _ = bipartite_subgraph(
        node_mask,
        edge_index,
        relabel_nodes=relabel_nodes,
        size=(num_users,num_items),
        return_edge_mask=False,
    )
    return edge_index    
    

def Fast_Sampling(dataset:Loader):
    """
    With Uniformal Sampling on Graph
    """
    train_edge_index = dataset.train_edge_index.to(device)
    num_items = dataset.num_items
    batch_size = config['bpr_batch_size']
    mini_batch = []
    train_loader = DataLoader(
            range(train_edge_index.size(1)),
            shuffle=True,
            batch_size=batch_size)
    for index in train_loader:
        pos_edge_label_index = train_edge_index[:,index]
        neg_edge_label_index = torch.randint(0, num_items,(index.numel(), ), device=device)
        edge_label_index = torch.stack([
            pos_edge_label_index[0],
            pos_edge_label_index[1],
            neg_edge_label_index,
        ])
        mini_batch.append(edge_label_index)
    return mini_batch
        
def Multi_Neg_Sampling(dataset:Loader,num_neg):
    train_edge_index = dataset.train_edge_index.to(device)
    num_items = dataset.num_items
    weights = dataset.sampling_weights
    batch_size = config['bpr_batch_size']
    sampler = WeightedRandomSampler(
        weights,
        num_samples=train_edge_index.size(1)
    )
    mini_batch = []
    train_loader = DataLoader(
        range(train_edge_index.size(1)),
        sampler=sampler,
        batch_size=batch_size
    )
    for index in train_loader:
        pos_edge_label_index = train_edge_index[:,index]
        neg_edge_label_index = torch.randint(0, num_items,(index.numel(),num_neg), device=device)
        edge_label_index = torch.stack([
            pos_edge_label_index[0],
            pos_edge_label_index[1],
            neg_edge_label_index,
        ])
        mini_batch.append(edge_label_index)
    return mini_batch

def negative_sampling_formal(dataset:Loader):
    train_edge_index = dataset.train_edge_index.to(device)
    num_items = dataset.num_items
    weights = dataset.sampling_weights
    batch_size = config['bpr_batch_size']
    mini_batch = []
    sampler = WeightedRandomSampler(
        weights,
        num_samples=train_edge_index.size(1)
    )
    train_loader = DataLoader(
        range(train_edge_index.size(1)),
        sampler=sampler,
        batch_size=batch_size
    )
    for index in train_loader:
        pos_edge_label_index = train_edge_index[:,index]
        neg_edge_label_index = negative_sampling_bipartite(dataset, pos_edge_label_index[0])
        edge_label_index = torch.stack([
            pos_edge_label_index[0],
            pos_edge_label_index[1],
            neg_edge_label_index,
        ])
        mini_batch.append(edge_label_index)
    return mini_batch

def negative_sampling_bipartite(dataset:Loader,batch_users):
    train_edge_index = dataset.train_edge_index
    num_items = dataset.num_items
    device = train_edge_index.device
    users = batch_users
    batch_size = users.size(0)
    user_all = train_edge_index[0]
    item_all = train_edge_index[1]
    pos_pair_codes = (user_all * num_items + item_all).cpu().numpy()
    pos_pair_set = set(pos_pair_codes)
    neg_items = torch.randint(0, num_items, (batch_size,), device=device)
    user_ids = users
    codes = (user_ids * num_items + neg_items).cpu().numpy()
    mask = torch.from_numpy(np.isin(codes, list(pos_pair_set))).to(device)
    retry_idx = torch.where(mask)[0]
    while retry_idx.numel() > 0:
        tmp = torch.randint(0, num_items, (retry_idx.numel(),), device=device)
        neg_items[retry_idx] = tmp
        codes = (user_ids[retry_idx] * num_items + tmp).cpu().numpy()
        mask = torch.from_numpy(np.isin(codes, list(pos_pair_set))).to(device)
        retry_idx = retry_idx[mask]
    return neg_items




def early_stopping(recall,
                   ndcg,
                   best,
                   patience,
                   model):
    """Apply global Recall@20 early stopping.

    ``ndcg`` and ``model`` remain in the signature for compatibility with the
    existing training entry points.  Only a strict Recall@20 improvement resets
    the counter.  The stop flag is raised on the epoch that completes the
    configured number of consecutive non-improving evaluations.
    """
    del ndcg, model
    if recall > best:
        patience = 0
        best = recall
        print('[BEST Recall@20]')
        return 0, best, patience

    patience += 1
    should_stop = patience >= world.patience
    if should_stop:
        print(
            '[EARLY STOP] Recall@20 did not improve for '
            f'{world.patience} consecutive epochs.'
        )
    return int(should_stop), best, patience

def eval(node_count,topk_index,logits,ground_truth,k):
    isin_mat = ground_truth.gather(1, topk_index)
    # Calculate recall
    recall = float((isin_mat.sum(dim=-1) / node_count.clamp(1e-6)).sum())
    # Calculate NDCG
    log_positions = torch.log2(torch.arange(2, k + 2, device=logits.device).float())
    dcg = (isin_mat / log_positions).sum(dim=-1)
    ideal_dcg = torch.zeros_like(dcg)
    for i in range(len(dcg)):
        ideal_dcg = (1.0 / log_positions[:node_count[i].clamp(max=k).int()]).sum()
    ndcg = float((dcg / ideal_dcg.clamp(min=1e-6)).sum())
    return recall,ndcg

# ====================end Metrics=============================
# =========================================================
def init_logger(model_name,dataset_name,log_dir='../log'):
    global log_writer
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"{model_name}_{dataset_name}_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.handlers = [] 
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    log_writer = logger
    return log_path


def print_log(*args, **kwargs):
    message = ' '.join(str(a) for a in args)
    if log_writer:
        log_writer.info(message)
    else:
        print(message, **kwargs)

def write_final_log(best_epoch, recall, ndcg, config):
    print_log("\n========== BEST RESULT ==========")
    print_log(f"Best Epoch: {best_epoch}")
    print_log(f"Recall@20: {recall:.4f}, NDCG@20: {ndcg:.4f}")
    print_log("Config:")
    print_log(json.dumps(config, indent=2))
