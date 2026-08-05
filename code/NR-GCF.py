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

config['representation_modulation_mode'] = (
    world.args.representation_modulation_mode
)
config['representation_modulation_ramp_epochs'] = (
    world.args.representation_modulation_ramp_epochs
)

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
          edge_loss_history=None,
          stable_edge_momentum=None,
          stable_filtering_epoch=None,
          update_legacy_momentum=True):
    model = model
    model.train()
    edge_index,indexes = Fast_Sampling(dataset=dataset)
    aver_loss = 0.
    total_batch = len(edge_index)
    for edge_label_index,index in zip(edge_index,indexes):
        opt.zero_grad()
        loss = model.get_loss(edge_label_index)
        needs_legacy_loss = update_legacy_momentum and epoch < 15
        needs_stable_loss = (
            stable_edge_momentum is not None
            and epoch <= int(stable_filtering_epoch)
        )
        if needs_legacy_loss or needs_stable_loss:
            instance_loss = model.get_instance_loss(edge_label_index)
            if needs_legacy_loss:
                model.update_momentum(index, instance_loss,epoch)
            if edge_loss_history is not None and needs_legacy_loss:
                edge_loss_history.observe(index, instance_loss)
            if needs_stable_loss:
                stable_edge_momentum.update(index, instance_loss)
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
if (world.args.export_edge_diagnostics
        and world.args.edge_filter_mode != 'current'):
    raise ValueError(
        '--export-edge-diagnostics records the exact legacy NR-GCF decision '
        'and therefore can only be combined with --edge-filter-mode current. '
        'The comparison modes write compact JSON via --edge-reliability-dir.'
    )
if world.args.representation_modulation_ramp_epochs < 0:
    raise ValueError('--representation-modulation-ramp-epochs cannot be negative')
if not 0.0 <= float(world.lambda_) <= 1.0:
    raise ValueError('--lambda_ must be within [0, 1]')
if (world.args.representation_modulation_mode
        == 'reliability_weighted_stage_two'
        and world.args.edge_filter_mode != 'hard_structure_momentum'):
    raise ValueError(
        'reliability_weighted_stage_two requires '
        '--edge-filter-mode hard_structure_momentum so its frozen confidence '
        'uses the validated structure-momentum semantics.'
    )
dataset = Loader()
log_path = init_logger(model_name='NR-GCF-new', dataset_name=world.config['dataset'])


train_edge_index = dataset.train_edge_index.to(device)
evaluation_train_edge_index = train_edge_index
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
uses_stable_momentum = (
    world.args.edge_filter_mode == 'hard_structure_momentum'
)
active_filtering_epoch = (
    int(world.args.edge_reliability_filtering_epoch)
    if uses_stable_momentum else 15
)
if active_filtering_epoch < 2 or active_filtering_epoch > world.TRAIN_epochs:
    raise ValueError(
        'Filtering epoch must be between 2 and the configured training epochs.'
    )
stable_edge_momentum = None
if uses_stable_momentum:
    from edge_reliability import StableEdgeMomentum
    stable_edge_momentum = StableEdgeMomentum(
        edge_count=train_edge_index.size(1),
        decay=world.args.edge_reliability_momentum_decay,
        device=device,
    )
best = 0.
patience = 0.
max_score = 0.
best_recall = 0.
best_epoch = 0
best_ndcg = 0.
# print(model.generate_weight(train_edge_index))
for epoch in range(1, world.TRAIN_epochs + 1):
    start_time = time.time()
    model.set_training_epoch(epoch)
    loss = train(dataset=dataset,
                 model=model,
                 opt=opt,
                 epoch=epoch,
                 edge_loss_history=edge_loss_history,
                 stable_edge_momentum=stable_edge_momentum,
                 stable_filtering_epoch=active_filtering_epoch,
                 update_legacy_momentum=not uses_stable_momentum)
    if epoch == active_filtering_epoch:
        if world.args.edge_filter_mode == 'current':
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
            filtered_train_edge_index = train_edge_index[:, retained_edge_mask]
            if world.args.export_edge_reliability_summary:
                from edge_reliability import (
                    build_reliability_policy,
                    write_reliability_summary,
                )
                current_policy = build_reliability_policy(
                    edge_index=train_edge_index,
                    raw_momentum=model.momentum_loss.detach().clone(),
                    num_users=num_users,
                    num_items=num_items,
                    mode='none',
                    topk=world.args.edge_diagnostics_topk,
                    chunk_size=world.args.edge_diagnostics_chunk_size,
                    min_degree=world.args.edge_diagnostics_min_degree,
                    momentum_quantile=world.args.edge_reliability_momentum_quantile,
                    structure_quantile=world.args.edge_reliability_structure_quantile,
                    structure_weight=world.args.edge_reliability_structure_weight,
                    minimum_weight=world.args.edge_reliability_min_weight,
                )
                current_policy['mode'] = 'current'
                current_policy['retained_mask'] = (
                    retained_edge_mask.detach().to(device='cpu')
                )
                current_policy['propagation_weight'] = (
                    current_policy['retained_mask'].to(torch.float32)
                )
                current_policy['decision'] = {
                    'mode': 'current',
                    'rule': 'exact current NR-GCF min-max score, values above beta set to zero, retain post-threshold score > 0',
                    'beta': float(config['beta']),
                    'raw_momentum_min': float(x_min.detach().cpu().item()),
                    'raw_momentum_max': float(x_max.detach().cpu().item()),
                    'uses_synthetic_labels': False,
                }
                current_policy['representation_modulation'] = {
                    'mode': world.args.representation_modulation_mode,
                    'lambda': float(world.lambda_),
                    'ramp_epochs': int(
                        world.args.representation_modulation_ramp_epochs
                    ),
                    'stage_one_modulation_active': (
                        world.args.representation_modulation_mode
                        == 'legacy_always'
                    ),
                }
                repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                write_reliability_summary(
                    output_dir=world.args.edge_reliability_dir,
                    policy=current_policy,
                    dataset=world.config['dataset'],
                    seed=world.seed,
                    requested_noise_ratio=world.args.requested_noise_ratio,
                    filtering_epoch=epoch,
                    labels_path=world.args.edge_reliability_labels_file,
                    noise_validation_path=(
                        world.args.edge_reliability_noise_validation_file
                    ),
                    repo_dir=repo_dir,
                )
                del current_policy
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
                    'filtered_train_edge_index': filtered_train_edge_index,
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
                del tracked_tensors
                del exporter
                del invariance_guard
                del raw_momentum_for_diagnostics
                del normalized_score_for_diagnostics
            # Preserve the corrected legacy behavior: filtering changes both
            # BPR positives and the normalized propagation graph.
            model.set_training_graph(filtered_train_edge_index)
            dataset.train_edge_index = filtered_train_edge_index.detach().cpu()
            dataset.sampling_weights = dataset.get_edge_weights(
                dataset.train_edge_index
            )
            train_edge_index = filtered_train_edge_index
            print_log(
                'Stage-two graph applied (current NR-GCF): '
                f'{train_edge_index.size(1)} retained edges; '
                'BPR sampling and propagation share the reconstructed graph; '
                f'evaluation mask remains {evaluation_train_edge_index.size(1)} '
                'pre-filter observed edges.'
            )
            model.activate_stage_two_modulation(filtering_epoch=epoch)
            if world.args.representation_modulation_mode == 'paper_stage_two':
                print_log(
                    'Paper-consistent stage two armed: unweighted cross-type '
                    'modulation begins with the next optimization epoch.'
                )
            del retained_edge_mask
            del filtered_train_edge_index
        else:
            # Pilot policies are frozen at their configured filtering point.
            # They never alter the BPR loss formula or add a stage-two norm.
            from edge_reliability import (
                build_reliability_policy,
                write_reliability_summary,
            )
            if uses_stable_momentum:
                raw_momentum = stable_edge_momentum.snapshot()
                momentum_semantics = (
                    'per_edge_ema_instance_bpr_loss_decay_'
                    + str(world.args.edge_reliability_momentum_decay)
                )
            else:
                raw_momentum = model.momentum_loss.detach().clone()
                momentum_semantics = 'legacy_runtime_momentum'
            policy = build_reliability_policy(
                edge_index=train_edge_index,
                raw_momentum=raw_momentum,
                num_users=num_users,
                num_items=num_items,
                mode=world.args.edge_filter_mode,
                topk=world.args.edge_diagnostics_topk,
                chunk_size=world.args.edge_diagnostics_chunk_size,
                min_degree=world.args.edge_diagnostics_min_degree,
                momentum_quantile=world.args.edge_reliability_momentum_quantile,
                structure_quantile=world.args.edge_reliability_structure_quantile,
                structure_weight=world.args.edge_reliability_structure_weight,
                minimum_weight=world.args.edge_reliability_min_weight,
                momentum_semantics=momentum_semantics,
            )
            if uses_stable_momentum:
                policy['warmup_epoch_count'] = active_filtering_epoch
            policy['representation_modulation'] = {
                'mode': world.args.representation_modulation_mode,
                'lambda': float(world.lambda_),
                'ramp_epochs': int(
                    world.args.representation_modulation_ramp_epochs
                ),
                'stage_one_modulation_active': (
                    world.args.representation_modulation_mode
                    == 'legacy_always'
                ),
            }
            repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            write_reliability_summary(
                output_dir=world.args.edge_reliability_dir,
                policy=policy,
                dataset=world.config['dataset'],
                seed=world.seed,
                requested_noise_ratio=world.args.requested_noise_ratio,
                filtering_epoch=epoch,
                labels_path=world.args.edge_reliability_labels_file,
                noise_validation_path=(
                    world.args.edge_reliability_noise_validation_file
                ),
                repo_dir=repo_dir,
            )

            if world.args.edge_filter_mode in (
                    'hard_consensus', 'hard_structure_only',
                    'hard_structure_momentum'):
                retained_edge_mask = policy['retained_mask'].to(
                    device=train_edge_index.device
                )
                filtered_train_edge_index = train_edge_index[:, retained_edge_mask]
                model.set_training_graph(filtered_train_edge_index)
                dataset.train_edge_index = filtered_train_edge_index.detach().cpu()
                dataset.sampling_weights = dataset.get_edge_weights(
                    dataset.train_edge_index
                )
                train_edge_index = filtered_train_edge_index
                print_log(
                    f'Stage-two {world.args.edge_filter_mode} applied: '
                    f'{train_edge_index.size(1)} retained edges; ordinary BPR '
                    'sampling and propagation use the same hard-filtered graph.'
                )
                del retained_edge_mask
                del filtered_train_edge_index
            elif world.args.edge_filter_mode in (
                    'soft_reliability', 'gated_soft_reliability'):
                propagation_weight = policy['propagation_weight'].to(
                    device=train_edge_index.device,
                    dtype=model.user_embedding.weight.dtype,
                )
                model.set_training_graph(
                    train_edge_index, edge_weight=propagation_weight
                )
                print_log(
                    f'Stage-two {world.args.edge_filter_mode} applied: all positive edges '
                    'remain uniformly sampled by the original BPR objective; '
                    'frozen reliability weights affect propagation only.'
                )
                del propagation_weight
            else:
                print_log(
                    'Stage-two none applied: graph, positive sampling, BPR '
                    'objective, and evaluation mask are unchanged.'
                )
            user_modulation_weight = None
            item_modulation_weight = None
            if (world.args.representation_modulation_mode
                    == 'reliability_weighted_stage_two'):
                user_modulation_weight = torch.from_numpy(
                    policy['user_node_confidence']
                )
                item_modulation_weight = torch.from_numpy(
                    policy['item_node_confidence']
                )
            model.activate_stage_two_modulation(
                filtering_epoch=epoch,
                user_weight=user_modulation_weight,
                item_weight=item_modulation_weight,
            )
            if world.args.representation_modulation_mode in (
                    'paper_stage_two', 'reliability_weighted_stage_two'):
                print_log(
                    'Representation modulation stage armed: mode='
                    f'{world.args.representation_modulation_mode}, '
                    f'lambda={world.lambda_}, ramp_epochs='
                    f'{world.args.representation_modulation_ramp_epochs}; '
                    'activation begins with the next optimization epoch.'
                )
            del user_modulation_weight
            del item_modulation_weight
            del raw_momentum
            del policy
            stable_edge_momentum = None
        original_train_edge_index = None
        if world.args.edge_diagnostics_stop_after_filter:
            print_log(
                f'Stopped after epoch-{active_filtering_epoch} filtering point '
                'by explicit diagnostics smoke-test option.'
            )
            break
    end_time = time.time()
    # Evaluation always masks the complete observed input train split. This
    # keeps the candidate set identical before/after filtering and across
    # filtering methods.
    recall,ndcg = test(
        [20], model, evaluation_train_edge_index,
        test_edge_index, num_users
    )
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
if (world.args.edge_filter_mode != 'current'
        or world.args.export_edge_reliability_summary):
    from edge_reliability import write_training_summary
    write_training_summary(
        output_dir=world.args.edge_reliability_dir,
        mode=world.args.edge_filter_mode,
        requested_epochs=world.TRAIN_epochs,
        epochs_completed=epoch,
        best_epoch=best_epoch,
        best_recall=best_recall,
        best_ndcg=best_ndcg,
        final_loss=float(loss.detach().cpu().item()),
        propagation_edge_count=model.active_edge_count,
        bpr_positive_edge_count=dataset.train_edge_index.size(1),
        representation_modulation_mode=(
            world.args.representation_modulation_mode
        ),
        representation_modulation_ramp_epochs=(
            world.args.representation_modulation_ramp_epochs
        ),
        representation_modulation_lambda=world.lambda_,
    )
write_final_log(best_epoch=best_epoch, recall=best_recall, ndcg=best_ndcg, config=config)
print_log(f"Log saved to: {log_path}")
