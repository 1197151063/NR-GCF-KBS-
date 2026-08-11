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


config = {
    # Preserve the released normal initialization for BPR experiments. SSM
    # and AU use standard Xavier Uniform because their normalized objectives
    # should not inherit the BPR-specific small-normal initialization.
    'init': (
        ('normal' if world.training_objective == 'bpr' else 'uniform')
        if world.embedding_init == 'auto'
        else ('normal' if world.embedding_init == 'normal' else 'uniform')
    ),
    'init_weight': world.init_weight,
    'dim':64,#EMBEDDING_SIZE
    'decay':world.decay,#L2_NORM
    'K':3,
    'beta':0.8,#BETA
    'lambda': world.lambda_,
    'lr':world.lr,#LEARNING_RATE
    'training_objective': world.training_objective,
    'num_neg': world.num_neg,
    'tau': world.tau,
    'objective_message_dropout': world.objective_message_dropout,
    'adap_tau_mode': world.adap_tau_mode,
    'adap_tau_temperature_2': world.adap_tau_temperature_2,
    'adap_tau_loss_quantile': world.adap_tau_loss_quantile,
    'adap_tau_recalibration_epoch': world.adap_tau_recalibration_epoch,
    'adap_tau_degree_quantile': world.adap_tau_degree_quantile,
    'adap_tau_initial_positive_gap': world.adap_tau_initial_positive_gap,
    'au_uniformity_weight': world.au_uniformity_weight,
    'au_uniformity_t': world.au_uniformity_t,
}

config['representation_modulation_mode'] = (
    world.args.representation_modulation_mode
)
config['representation_modulation_ramp_epochs'] = (
    world.args.representation_modulation_ramp_epochs
)

def Fast_Sampling(dataset:Loader, sample_bpr_negative=True):
    """
    With Uniformal Sampling on Graph
    """
    train_edge_index = dataset.train_edge_index.to(device)
    num_items = dataset.num_items
    batch_size = int(world.config['bpr_batch_size'])
    mini_batch = []
    indexes = []
    if sample_bpr_negative:
        # Preserve the released NR-GCF/BPR sampler and its torch RNG stream.
        train_loader = DataLoader(
                range(train_edge_index.size(1)),
                shuffle=True,
                batch_size=batch_size)
    else:
        # Adap_tau shuffles the complete interaction array with NumPy before
        # slicing consecutive batches. Keep this path separate so normalized
        # objectives do not perturb the original BPR path.
        order = np.arange(train_edge_index.size(1), dtype=np.int64)
        np.random.shuffle(order)
        train_loader = (
            torch.from_numpy(order[start:start + batch_size]).long()
            for start in range(0, order.size, batch_size)
        )
    for index in train_loader:
        pos_edge_label_index = train_edge_index[:,index]
        if sample_bpr_negative:
            neg_edge_label_index = torch.randint(
                0, num_items, (index.numel(),), device=device
            )
            edge_label_index = torch.stack([
                pos_edge_label_index[0],
                pos_edge_label_index[1],
                neg_edge_label_index,
            ])
        else:
            # Adap_tau LightGCN uses no_sample: the other positive items in
            # this batch become the B-1 negatives inside the objective.
            edge_label_index = pos_edge_label_index
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
    needs_legacy_loss = update_legacy_momentum and epoch < 15
    needs_stable_loss = (
        stable_edge_momentum is not None
        and epoch <= int(stable_filtering_epoch)
    )
    sample_bpr_negative = (
        model.config.get('training_objective') == 'bpr'
        or needs_legacy_loss
        or needs_stable_loss
    )
    model.prepare_objective_epoch(dataset.train_edge_index.to(device), epoch)
    edge_index,indexes = Fast_Sampling(
        dataset=dataset, sample_bpr_negative=sample_bpr_negative
    )
    aver_loss = 0.
    total_batch = len(edge_index)
    adap_tau_loss_sum = None
    adap_tau_observation_count = None
    if model.config.get('training_objective') == 'adap_tau':
        adap_tau_loss_sum = torch.zeros(
            model.num_users, dtype=torch.float32, device=device
        )
        adap_tau_observation_count = torch.zeros_like(adap_tau_loss_sum)
    for edge_label_index,index in zip(edge_index,indexes):
        opt.zero_grad()
        if adap_tau_loss_sum is not None:
            loss, objective_aux = model.get_loss(
                edge_label_index, return_aux=True
            )
            users = edge_label_index[0]
            unit_loss = objective_aux['unit_temperature_loss'].to(
                dtype=adap_tau_loss_sum.dtype
            )
            adap_tau_loss_sum.index_add_(0, users, unit_loss)
            adap_tau_observation_count.index_add_(
                0, users, torch.ones_like(unit_loss)
            )
        else:
            loss = model.get_loss(edge_label_index)
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
    if adap_tau_loss_sum is not None:
        model.finish_objective_epoch(
            adap_tau_loss_sum, adap_tau_observation_count
        )
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
        if world.training_objective in ('ssm', 'adap_tau'):
            # Match the Adap_tau entry point without changing BPR execution.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


seed_runtime(world.seed)
device = world.device
if world.patience < 1:
    raise ValueError('--patience must be a positive integer')
if world.decay < 0:
    raise ValueError('--decay must be non-negative')
if world.config['bpr_batch_size'] < 1:
    raise ValueError('--bpr_batch must be positive')
if (world.training_objective in ('ssm', 'adap_tau')
        and world.config['bpr_batch_size'] < 2):
    raise ValueError('in-batch SSM/Adap-tau requires --bpr_batch >= 2')
if world.training_objective == 'ssm':
    if world.tau <= 0:
        raise ValueError('--tau must be positive for SSM')
if not 0.0 <= world.objective_message_dropout < 1.0:
    raise ValueError('--objective-message-dropout must be within [0, 1)')
if world.training_objective == 'adap_tau':
    if world.adap_tau_temperature_2 <= 0:
        raise ValueError('--adap-tau-temperature-2 must be positive')
    if not 0.0 <= world.adap_tau_loss_quantile <= 1.0:
        raise ValueError('--adap-tau-loss-quantile must be within [0, 1]')
    if world.adap_tau_recalibration_epoch < 0:
        raise ValueError('--adap-tau-recalibration-epoch cannot be negative')
    if not 0.0 <= world.adap_tau_degree_quantile <= 1.0:
        raise ValueError('--adap-tau-degree-quantile must be within [0, 1]')
    if world.adap_tau_initial_positive_gap <= 0:
        raise ValueError('--adap-tau-initial-positive-gap must be positive')
if world.training_objective == 'au':
    if world.au_uniformity_weight < 0:
        raise ValueError('--au-uniformity-weight must be non-negative')
    if world.au_uniformity_t <= 0:
        raise ValueError('--au-uniformity-t must be positive')
if (world.training_objective != 'bpr'
        and world.args.edge_filter_mode != 'none'):
    raise ValueError(
        'SSM/AU/Adap-tau objective pilots require --edge-filter-mode none so the '
        'ranking objective and graph filtering are not changed together. '
        'Their integration with edge reliability must be evaluated separately.'
    )
if (world.args.export_edge_diagnostics
        and world.args.edge_filter_mode != 'current'):
    raise ValueError(
        '--export-edge-diagnostics records the exact legacy NR-GCF decision '
        'and therefore can only be combined with --edge-filter-mode current. '
        'The comparison modes write compact JSON via --edge-reliability-dir.'
    )
if world.args.representation_modulation_ramp_epochs < 0:
    raise ValueError('--representation-modulation-ramp-epochs cannot be negative')
if not 0.0 <= world.args.edge_reliability_max_removal_ratio <= 1.0:
    raise ValueError(
        '--edge-reliability-max-removal-ratio must be within [0, 1]'
    )
if world.args.edge_reliability_filtering_schedule == 'adaptive':
    if world.args.edge_filter_mode != 'hard_structure_momentum':
        raise ValueError(
            'adaptive filtering schedule requires '
            '--edge-filter-mode hard_structure_momentum'
        )
    if world.args.edge_reliability_adaptive_min_epoch < 2:
        raise ValueError('--edge-reliability-adaptive-min-epoch must be >= 2')
    if (world.args.edge_reliability_adaptive_max_epoch
            < world.args.edge_reliability_adaptive_min_epoch):
        raise ValueError(
            '--edge-reliability-adaptive-max-epoch must be >= min epoch'
        )
    if world.args.edge_reliability_adaptive_max_epoch > world.TRAIN_epochs:
        raise ValueError(
            '--edge-reliability-adaptive-max-epoch cannot exceed --epochs'
        )
    if not 0.0 <= world.args.edge_reliability_adaptive_min_coverage <= 1.0:
        raise ValueError(
            '--edge-reliability-adaptive-min-coverage must be within [0, 1]'
        )
    if not 0.0 <= world.args.edge_reliability_adaptive_jaccard <= 1.0:
        raise ValueError(
            '--edge-reliability-adaptive-jaccard must be within [0, 1]'
        )
    if world.args.edge_reliability_adaptive_stable_checks < 1:
        raise ValueError(
            '--edge-reliability-adaptive-stable-checks must be positive'
        )
if (world.args.representation_modulation_mode in (
        'reliability_weighted_always',
        'reliability_weighted_stage_two')
        and world.args.edge_filter_mode != 'hard_structure_momentum'):
    raise ValueError(
        'reliability-weighted modulation requires '
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
print_log('Training objective: ' + str(model.objective_metadata()))
opt = torch.optim.Adam(params=model.parameters(),lr=config['lr'])
edge_loss_history = None
if world.args.export_edge_diagnostics:
    from edge_diagnostics import EdgeLossHistory
    edge_loss_history = EdgeLossHistory(original_train_edge_index.size(1))
uses_stable_momentum = (
    world.args.edge_filter_mode == 'hard_structure_momentum'
)
uses_adaptive_filtering = (
    uses_stable_momentum
    and world.args.edge_reliability_filtering_schedule == 'adaptive'
)
filtering_disabled = world.args.edge_filter_mode == 'none'
configured_filtering_epoch = (
    None
    if filtering_disabled else (
        int(world.args.edge_reliability_adaptive_max_epoch)
        if uses_adaptive_filtering
        else (
            int(world.args.edge_reliability_filtering_epoch)
            if uses_stable_momentum else 15
        )
    )
)
active_filtering_epoch = configured_filtering_epoch
if (not filtering_disabled
        and (active_filtering_epoch < 2
             or active_filtering_epoch > world.TRAIN_epochs)):
    raise ValueError(
        'Filtering epoch must be between 2 and the configured training epochs.'
    )
stable_edge_momentum = None
adaptive_filtering_controller = None
cached_structural_features = None
if uses_stable_momentum:
    from edge_reliability import StableEdgeMomentum
    stable_edge_momentum = StableEdgeMomentum(
        edge_count=train_edge_index.size(1),
        decay=world.args.edge_reliability_momentum_decay,
        device=device,
    )
    if uses_adaptive_filtering:
        from edge_reliability import AdaptiveFilteringTrigger
        adaptive_filtering_controller = AdaptiveFilteringTrigger(
            min_epoch=world.args.edge_reliability_adaptive_min_epoch,
            max_epoch=world.args.edge_reliability_adaptive_max_epoch,
            min_coverage=(
                world.args.edge_reliability_adaptive_min_coverage
            ),
            jaccard_threshold=(
                world.args.edge_reliability_adaptive_jaccard
            ),
            stable_checks=(
                world.args.edge_reliability_adaptive_stable_checks
            ),
        )
best = 0.
patience = 0.
max_score = 0.
best_recall = 0.
best_epoch = 0
best_ndcg = 0.
best_post_filter_score = None
best_post_filter_epoch = None
best_post_filter_recall = None
best_post_filter_ndcg = None
representation_modulation_trace = []
objective_training_trace = []
stopped_early = False
filtering_applied = False
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
                 update_legacy_momentum=(
                     world.training_objective == 'bpr'
                     and not uses_stable_momentum
                 ))
    filter_now = (
        not filtering_disabled
        and not filtering_applied
        and epoch == active_filtering_epoch
    )
    policy_for_filter = None
    raw_momentum_for_filter = None
    momentum_observed_mask_for_filter = None
    if (uses_adaptive_filtering
            and not filtering_applied
            and epoch >= world.args.edge_reliability_adaptive_min_epoch
            and epoch <= world.args.edge_reliability_adaptive_max_epoch):
        from edge_reliability import (
            build_reliability_policy,
            compute_two_hop_structure_features,
        )
        raw_momentum_for_filter = stable_edge_momentum.snapshot(
            require_all=False
        )
        momentum_observed_mask_for_filter = (
            stable_edge_momentum.observed_mask()
        )
        coverage = stable_edge_momentum.coverage()
        if cached_structural_features is None:
            cached_structural_features = compute_two_hop_structure_features(
                edge_index=train_edge_index,
                num_users=num_users,
                num_items=num_items,
                topk=world.args.edge_diagnostics_topk,
                chunk_size=world.args.edge_diagnostics_chunk_size,
            )
        policy_for_filter = build_reliability_policy(
            edge_index=train_edge_index,
            raw_momentum=raw_momentum_for_filter,
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
            max_removal_ratio=(
                world.args.edge_reliability_max_removal_ratio
            ),
            momentum_semantics=(
                'per_edge_ema_instance_bpr_loss_decay_'
                + str(world.args.edge_reliability_momentum_decay)
            ),
            momentum_observed_mask=momentum_observed_mask_for_filter,
            structural_features=cached_structural_features,
        )
        filter_now, readiness = adaptive_filtering_controller.observe(
            epoch=epoch,
            coverage=coverage,
            retained_mask=policy_for_filter['retained_mask'].numpy(),
        )
        print_log(
            'Adaptive filtering readiness: '
            f'epoch={epoch}, coverage={coverage:.6f}, '
            f'removed={readiness["removed_edge_count"]}, '
            f'jaccard={readiness["removed_set_jaccard"]}, '
            'stable_checks='
            f'{readiness["consecutive_stable_checks"]}/'
            f'{world.args.edge_reliability_adaptive_stable_checks}, '
            f'trigger={filter_now}, reason={readiness["trigger_reason"]}.'
        )
        if filter_now:
            active_filtering_epoch = epoch
        else:
            del policy_for_filter
            del raw_momentum_for_filter
            del momentum_observed_mask_for_filter
            policy_for_filter = None
            raw_momentum_for_filter = None
            momentum_observed_mask_for_filter = None
    if filter_now:
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
                    max_removal_ratio=(
                        world.args.edge_reliability_max_removal_ratio
                    ),
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
                    'lambda': (
                        float(world.lambda_)
                        if world.args.representation_modulation_mode
                        == 'blend_always' else None
                    ),
                    'lambda_note': (
                        'active propagation/CrossNorm blend weight'
                        if world.args.representation_modulation_mode
                        == 'blend_always'
                        else 'ignored by direct NRGCF cross_norm'
                    ),
                    'ramp_epochs': int(
                        world.args.representation_modulation_ramp_epochs
                    ),
                    'stage_one_modulation_active': (
                        world.args.representation_modulation_mode in (
                            'legacy_always', 'original_always', 'blend_always',
                            'reliability_weighted_always',
                        )
                    ),
                    'scale_definition': 'uncapped_cross_type_rms',
                }
                current_policy['training_objective'] = (
                    model.objective_metadata()
                )
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
            if world.args.representation_modulation_mode in (
                    'original_stage_two', 'paper_stage_two'):
                print_log(
                    'Original-code stage two armed: direct unweighted '
                    'cross_norm begins with the next optimization epoch.'
                )
            del retained_edge_mask
            del filtered_train_edge_index
        else:
            # Pilot policies are frozen at their configured filtering point.
            # They never alter the configured objective or add a stage-two norm.
            from edge_reliability import (
                build_reliability_policy,
                write_reliability_summary,
            )
            if policy_for_filter is not None:
                raw_momentum = raw_momentum_for_filter
                momentum_semantics = (
                    'per_edge_ema_instance_bpr_loss_decay_'
                    + str(world.args.edge_reliability_momentum_decay)
                )
                policy = policy_for_filter
            elif uses_stable_momentum:
                raw_momentum = stable_edge_momentum.snapshot(
                    require_all=True
                )
                momentum_observed_mask_for_filter = (
                    stable_edge_momentum.observed_mask()
                )
                momentum_semantics = (
                    'per_edge_ema_instance_bpr_loss_decay_'
                    + str(world.args.edge_reliability_momentum_decay)
                )
            else:
                raw_momentum = model.momentum_loss.detach().clone()
                momentum_semantics = 'legacy_runtime_momentum'
            if policy_for_filter is None:
                policy = build_reliability_policy(
                    edge_index=train_edge_index,
                    raw_momentum=raw_momentum,
                    num_users=num_users,
                    num_items=num_items,
                    mode=world.args.edge_filter_mode,
                    topk=world.args.edge_diagnostics_topk,
                    chunk_size=world.args.edge_diagnostics_chunk_size,
                    min_degree=world.args.edge_diagnostics_min_degree,
                    momentum_quantile=(
                        world.args.edge_reliability_momentum_quantile
                    ),
                    structure_quantile=(
                        world.args.edge_reliability_structure_quantile
                    ),
                    structure_weight=(
                        world.args.edge_reliability_structure_weight
                    ),
                    minimum_weight=world.args.edge_reliability_min_weight,
                    max_removal_ratio=(
                        world.args.edge_reliability_max_removal_ratio
                    ),
                    momentum_semantics=momentum_semantics,
                    momentum_observed_mask=(
                        momentum_observed_mask_for_filter
                    ),
                    structural_features=cached_structural_features,
                )
            if uses_stable_momentum:
                policy['warmup_epoch_count'] = epoch
            if uses_adaptive_filtering:
                policy['adaptive_filtering'] = (
                    adaptive_filtering_controller.metadata()
                )
            else:
                policy['adaptive_filtering'] = {
                    'schedule': 'fixed',
                    'configured_filtering_epoch': int(
                        configured_filtering_epoch
                    ),
                    'actual_filtering_epoch': int(epoch),
                }
            policy['representation_modulation'] = {
                'mode': world.args.representation_modulation_mode,
                'lambda': (
                    float(world.lambda_)
                    if world.args.representation_modulation_mode
                    == 'blend_always' else None
                ),
                'lambda_note': (
                    'active propagation/CrossNorm blend weight'
                    if world.args.representation_modulation_mode
                    == 'blend_always'
                    else 'ignored by direct NRGCF cross_norm'
                ),
                'ramp_epochs': int(
                    world.args.representation_modulation_ramp_epochs
                ),
                'stage_one_modulation_active': (
                    world.args.representation_modulation_mode in (
                        'legacy_always', 'original_always', 'blend_always',
                        'reliability_weighted_always',
                    )
                ),
                'scale_definition': 'uncapped_cross_type_rms',
            }
            policy['training_objective'] = model.objective_metadata()
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
                    'Stage-two none applied: graph, positive sampling, '
                    f'{world.training_objective.upper()} objective, and '
                    'evaluation mask are unchanged.'
                )
            user_modulation_weight = None
            item_modulation_weight = None
            if world.args.representation_modulation_mode in (
                    'reliability_weighted_always',
                    'reliability_weighted_stage_two'):
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
                    'original_stage_two', 'paper_stage_two',
                    'reliability_weighted_always',
                    'reliability_weighted_stage_two'):
                if (world.args.representation_modulation_mode
                        == 'reliability_weighted_always'):
                    print_log(
                        'Always-on direct cross_norm retained; its RMS '
                        'estimator is now frozen reliability-weighted.'
                    )
                else:
                    print_log(
                        'Representation modulation stage armed: mode='
                        f'{world.args.representation_modulation_mode}, '
                        'operator=direct_cross_norm, ramp_epochs='
                        f'{world.args.representation_modulation_ramp_epochs}; '
                        'activation begins with the next optimization epoch.'
                    )
            del user_modulation_weight
            del item_modulation_weight
            del raw_momentum
            del policy
            stable_edge_momentum = None
            cached_structural_features = None
            policy_for_filter = None
            raw_momentum_for_filter = None
            momentum_observed_mask_for_filter = None
        filtering_applied = True
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
    modulation_snapshot = model.modulation_snapshot()
    modulation_snapshot['epoch'] = int(epoch)
    representation_modulation_trace.append(modulation_snapshot)
    objective_snapshot = model.objective_epoch_snapshot()
    if objective_snapshot is not None:
        objective_training_trace.append(objective_snapshot)
    # Filtering and any reliability-weighted modulation are installed before
    # this epoch's evaluation.  Include the trigger epoch itself and use the
    # same Recall@20 monitor as global early stopping.
    if filtering_applied and epoch >= active_filtering_epoch:
        post_filter_score = recall[20]
        if (best_post_filter_score is None
                or post_filter_score > best_post_filter_score):
            best_post_filter_score = post_filter_score
            best_post_filter_epoch = epoch
            best_post_filter_recall = recall[20]
            best_post_filter_ndcg = ndcg[20]
    flag,best,patience = utils.early_stopping(recall[20],ndcg[20],best,patience,model)
    if patience == 0:
        best_epoch = epoch
        best_recall = recall[20]
        best_ndcg = ndcg[20]
    objective_suffix = ''
    if objective_snapshot is not None:
        objective_suffix = (
            ', inv_tau:[{low:.4f},{mean:.4f},{high:.4f}], w0:{w0:.4f}'
        ).format(
            low=objective_snapshot['user_inverse_temperature_min'],
            mean=objective_snapshot['user_inverse_temperature_mean'],
            high=objective_snapshot['user_inverse_temperature_max'],
            w0=objective_snapshot['positive_inverse_temperature'],
        )
    print_log(f'Epoch: {epoch:03d}, aver_loss : {loss:.5f}, R@20: '
            f'{recall[20]:.4f}, N@20: {ndcg[20]:.4f}'
            f'{objective_suffix}, time:{end_time-start_time:.2f} seconds')
    if flag == 1:
        stopped_early = True
        print_log(
            'Global early stopping at epoch '
            f'{epoch}: Recall@20 did not improve for '
            f'{world.patience} consecutive epochs. '
            f'Best Recall@20={best_recall:.6f} at epoch {best_epoch}.'
        )
        break
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
        representation_modulation_trace=representation_modulation_trace,
        best_post_filter_epoch=best_post_filter_epoch,
        best_post_filter_recall=best_post_filter_recall,
        best_post_filter_ndcg=best_post_filter_ndcg,
        early_stopping_patience=world.patience,
        early_stopped=stopped_early,
        early_stopping_wait=patience,
        filtering_schedule=(
            world.args.edge_reliability_filtering_schedule
            if uses_stable_momentum else 'fixed'
        ),
        configured_filtering_epoch=configured_filtering_epoch,
        actual_filtering_epoch=(
            active_filtering_epoch if filtering_applied else None
        ),
        adaptive_filtering_trace=(
            adaptive_filtering_controller.trace
            if adaptive_filtering_controller is not None else None
        ),
        training_objective=model.objective_metadata(),
        objective_training_trace=objective_training_trace,
    )
write_final_log(best_epoch=best_epoch, recall=best_recall, ndcg=best_ndcg, config=config)
print_log(f"Log saved to: {log_path}")
