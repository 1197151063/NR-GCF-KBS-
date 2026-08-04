import torch
from dataloader import Loader
import world
from procedure import test
from torch.utils.data import DataLoader
import time
import utils
import os
import random
import numpy as np
from model import NRGCF,RecModel
from utils import init_logger, print_log, write_final_log


if world.config['dataset'] == 'yelp2018':
    config = {
        'init':'normal',#NORMAL DISTRIBUTION
        'init_weight':world.init_weight,#INIT WEIGHT
        'dim':64,#EMBEDDING_SIZE
        'decay':world.decay,#L2_NORM
        'K':3,
        'beta':0.8,#BETA
        'lambda': world.lambda_,
        'lr':world.lr,#LEARNING_RATE
    }

if world.config['dataset'] == 'amazon-book':
    config = {
        'init':'normal',#NORMAL DISTRIBUTION
        'init_weight':world.init_weight,#INIT WEIGHT
        'dim':64,#EMBEDDING_SIZE
        'decay':world.decay,#L2_NORM
        'K':3,
        'beta':0.8,#BETA
        'lambda': world.lambda_,
        'lr':world.lr,#LEARNING_RATE
    }

def Fast_Sampling(dataset:Loader):
    """
    With Uniformal Sampling on Graph
    """
    train_edge_index = dataset.train_edge_index.to(device)
    num_items = dataset.num_items
    batch_size = 2048
    mini_batch = []
    indexes = []
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
        indexes.append(index)
    return mini_batch,indexes

def train(dataset:Loader,
          model:NRGCF,
          opt:torch.optim.Optimizer,
          epoch,
          edge_loss_history=None):
    model = model
    model.train()
    edge_index,indexes = Fast_Sampling(dataset=dataset)
    aver_loss = 0.
    total_batch = len(edge_index)
    for edge_label_index,index in zip(edge_index,indexes):
        opt.zero_grad()
        loss = model.get_loss(edge_label_index)
        if epoch < 15:
            instance_loss = model.get_instance_loss(edge_label_index)
            model.update_momentum(index, instance_loss,epoch)
            if edge_loss_history is not None:
                edge_loss_history.observe(index, instance_loss)
        loss.backward()
        opt.step()   
        aver_loss += (loss)
    aver_loss /= total_batch
    return aver_loss


def seed_runtime(seed):
    """Apply the existing --seed argument before data/model construction."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_runtime(world.seed)
device = world.device
dataset = Loader()
log_path = init_logger(model_name='NR-GCF-new', dataset_name=world.config['dataset'])


train_edge_index = dataset.train_edge_index.to(device)
original_train_edge_index = (
    train_edge_index if world.args.export_edge_diagnostics else None
)
test_edge_index = dataset.test_edge_index.to(device)
num_users = dataset.num_users
num_items = dataset.num_items
model = NRGCF(num_users=num_users,
                 num_items=num_items,
                 edge_index=train_edge_index,
                 config=config).to(device)
opt = torch.optim.Adam(params=model.parameters(),lr=config['lr'])
edge_loss_history = None
if world.args.export_edge_diagnostics:
    from edge_diagnostics import EdgeLossHistory
    edge_loss_history = EdgeLossHistory(original_train_edge_index.size(1))
best = 0.
patience = 0.
max_score = 0.
best_recall = 0.
best_epoch = 0
best_ndcg = 0.
# print(model.generate_weight(train_edge_index))
for epoch in range(1, world.TRAIN_epochs + 1):
    start_time = time.time()
    loss = train(dataset=dataset,
                 model=model,
                 opt=opt,
                 epoch=epoch,
                 edge_loss_history=edge_loss_history)
    if epoch == 15:
        raw_momentum_for_diagnostics = None
        normalized_score_for_diagnostics = None
        if edge_loss_history is not None:
            raw_momentum_for_diagnostics = model.momentum_loss.detach().clone()
        momentum_loss = model.momentum_loss
        x_max = torch.max(momentum_loss)
        x_min = torch.min(momentum_loss)
        momentum_loss = (momentum_loss - x_min) / (x_max - x_min)
        if edge_loss_history is not None:
            normalized_score_for_diagnostics = momentum_loss.detach().clone()
        momentum_loss[momentum_loss > config['beta']] = 0
        retained_edge_mask = momentum_loss > 0
        train_edge_index = train_edge_index[:, retained_edge_mask]
        if edge_loss_history is not None:
            from edge_diagnostics import (
                DiagnosticsInvarianceGuard,
                EdgeDiagnosticsExporter,
                write_invariance_report,
            )
            tracked_tensors = {
                'original_train_edge_index': original_train_edge_index,
                'raw_momentum_loss': raw_momentum_for_diagnostics,
                'normalized_edge_score': normalized_score_for_diagnostics,
                'post_threshold_score': momentum_loss,
                'retained_edge_mask': retained_edge_mask,
                'filtered_train_edge_index': train_edge_index,
            }
            invariance_guard = None
            if world.args.edge_diagnostics_verify_invariance:
                invariance_guard = DiagnosticsInvarianceGuard(model, tracked_tensors)
            repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            exporter = EdgeDiagnosticsExporter(
                args=world.args,
                model_config=config,
                output_dir=world.args.edge_diagnostics_dir,
                repo_dir=repo_dir,
            )
            exporter.export(
                edge_index=original_train_edge_index,
                num_users=num_users,
                num_items=num_items,
                history=edge_loss_history,
                raw_momentum=raw_momentum_for_diagnostics,
                normalized_score=normalized_score_for_diagnostics,
                post_threshold_score=momentum_loss,
                retained_mask=retained_edge_mask,
                filtering_epoch=epoch,
                warmup_epoch_count=14,
                threshold=config['beta'],
            )
            if invariance_guard is not None:
                invariance_result = invariance_guard.verify()
                write_invariance_report(
                    world.args.edge_diagnostics_dir, invariance_result
                )
                if not invariance_result['passed']:
                    raise RuntimeError(
                        'Edge diagnostics invariance verification failed: '
                        + str(invariance_result)
                    )
            edge_loss_history = None
            original_train_edge_index = None
            del tracked_tensors
            del exporter
            del invariance_guard
            del raw_momentum_for_diagnostics
            del normalized_score_for_diagnostics
        del retained_edge_mask
        if world.args.edge_diagnostics_stop_after_filter:
            print_log('Stopped after epoch-15 filtering point by explicit diagnostics smoke-test option.')
            break
    end_time = time.time()
    recall,ndcg = test([20],model,train_edge_index,test_edge_index,num_users)
    flag,best,patience = utils.early_stopping(recall[20],ndcg[20],best,patience,model)
    if patience == 0:
        best_epoch = epoch
        best_recall = recall[20]
        best_ndcg = ndcg[20]
    if flag == 1:
        break
    print_log(f'Epoch: {epoch:03d}, aver_loss : {loss:.5f}, R@20: '
            f'{recall[20]:.4f}, N@20: {ndcg[20]:.4f}, '
            f'time:{end_time-start_time:.2f} seconds')
write_final_log(best_epoch=best_epoch, recall=best_recall, ndcg=best_ndcg, config=config)
print_log(f"Log saved to: {log_path}")
