from torch import nn,Tensor,LongTensor
from torch_geometric.utils import degree
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import SparseTensor
from torch_sparse import SparseTensor,matmul
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import math
import torch.nn.functional as F
import torch
import world
from adap_tau_objectives import (
    adap_tau_in_batch_instance_loss,
    adap_tau_inverse_temperature,
    initial_adap_tau_inverse_temperature,
    ssm_in_batch_instance_loss,
)
from utils import dropout_node_bipartite
from torch_geometric.utils import dropout_edge,dropout_path,bipartite_subgraph
device = world.device


class RecModel(MessagePassing):
    def __init__(self,
                 num_users:int,
                 num_items:int,
                 config,
                 edge_index:LongTensor):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = num_users + num_items
        self.config = config
        self.edge_index = edge_index
        self.embedding_dim = config['dim']
        self.user_embedding = nn.Embedding(num_users,self.embedding_dim)
        self.item_embedding = nn.Embedding(num_items,self.embedding_dim)
        self.dropout = nn.Dropout(p=world.dropout_rate)
        self.reset_parameters(config)
    
    def reset_parameters(self,config):
        if(config['init'] == 'normal'):
            nn.init.normal_(self.user_embedding.weight.data,std=config['init_weight'])
            nn.init.normal_(self.item_embedding.weight.data,std=config['init_weight'])
        else:
            nn.init.xavier_uniform_(self.user_embedding.weight.data,gain=config['init_weight'])
            nn.init.xavier_uniform_(self.item_embedding.weight.data,gain=config['init_weight'])
        self.f = nn.Sigmoid()

    def get_sparse_bipartite_graph(self,
                                    edge_index,
                                    use_value=False,
                                    value=None):
        num_users = self.num_users
        num_items = self.num_items
        r,c = edge_index
        if use_value:
            return SparseTensor(row=r,col=c,value=value,sparse_sizes=(num_users,num_items))
        else:
            return SparseTensor(row=r,col=c,sparse_sizes=(num_users,num_items))
        
    def get_sparse_graph(self,
                         edge_index,
                         use_value=False,
                         value=None):
        num_users = self.num_users
        num_nodes = self.num_nodes
        r,c = edge_index
        row = torch.cat([r , c + num_users])
        col = torch.cat([c + num_users , r])
        if use_value:
            value = torch.cat([value,value])
            return SparseTensor(row=row,col=col,value=value,sparse_sizes=(num_nodes,num_nodes))
        else:
            return SparseTensor(row=row,col=col,sparse_sizes=(num_nodes,num_nodes))
    
    def link_prediction(self,
                        src_index:Tensor=None,
                        dst_index:Tensor=None):
        out_u,out_i = self.forward(edge_index=self.edge_index)
        # out_u = F.normalize(out_u, dim=-1)
        # out_i = F.normalize(out_i, dim=-1)
        if src_index is None:
            src_index = torch.arange(self.num_users).long()
        if dst_index is None:
            dst_index = torch.arange(self.num_items).long()
        out_src = out_u[src_index]
        out_dst = out_i[dst_index]
        pred = out_src @ out_dst.t()
        return pred
    

    
    def get_sparse_bipartite_graph_transpose(self,
                                             edge_index:LongTensor,
                                             use_value=False,
                                             value=None):
        num_users = self.num_users
        num_items = self.num_items
        r,c = edge_index
        if use_value:
            return SparseTensor(row=c,col=r,value=value,sparse_sizes=(num_items,num_users))
        else:
            return SparseTensor(row=c,col=r,sparse_sizes=(num_items,num_users))
    
    def forward(self,edge_index:LongTensor=None):
        pass

    def bpr_loss(self,edge_label_index):
        user_emb,item_emb = self.forward(edge_index=self.edge_index)
        user_emb = user_emb[edge_label_index[0]]
        pos_item_emb = item_emb[edge_label_index[1]]
        neg_item_emb = item_emb[edge_label_index[2]]
        pos_rank = (user_emb * pos_item_emb).sum(dim=-1)
        neg_rank = (user_emb * neg_item_emb).sum(dim=-1)
        return F.softplus(neg_rank - pos_rank).mean()
    
    def get_loss(self,edge_label_index):
        pass 
    
    def l2_reg(self,edge_label_index):
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        embedding = torch.cat([user_emb[edge_label_index[0]],
                               item_emb[edge_label_index[1]],
                               item_emb[edge_label_index[2]]])
        regularization =  (1/2) * embedding.norm(p=2).pow(2)/ edge_label_index.size(1)
        return self.config['decay'] * regularization

    def alignment_loss(self,edge_label_index:LongTensor):
        user_emb,item_emb = self.forward(edge_index=self.edge_index)
        user_emb = user_emb[edge_label_index[0]]
        item_emb = item_emb[edge_label_index[1]]
        user_emb = F.normalize(user_emb, dim=-1)
        item_emb = F.normalize(item_emb, dim=-1)
        return (user_emb - item_emb).norm(dim=1).pow(2).mean()
    
    def uniformity(self,x, t=2):
        x = F.normalize(x, dim=-1)
        if x.size(0) < 2:
            return x.sum() * 0.0
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
    
    def uniformity_loss(self,edge_label_index:LongTensor):
        user_emb,item_emb = self.forward(edge_index=self.edge_index)
        user_emb = user_emb[edge_label_index[0]]
        item_emb = item_emb[edge_label_index[1]]
        return (
            self.uniformity(user_emb, t=self.config['au_uniformity_t'])
            + self.uniformity(item_emb, t=self.config['au_uniformity_t'])
        )

    def au_loss(self, edge_label_index:LongTensor):
        user_all, item_all = self.forward(edge_index=self.edge_index)
        user_emb = user_all[edge_label_index[0]]
        item_emb = item_all[edge_label_index[1]]
        user_normalized = F.normalize(user_emb, dim=-1)
        item_normalized = F.normalize(item_emb, dim=-1)
        alignment = (user_normalized - item_normalized).pow(2).sum(dim=-1).mean()
        uniformity = (
            self.uniformity(user_emb, t=self.config['au_uniformity_t'])
            + self.uniformity(item_emb, t=self.config['au_uniformity_t'])
        )
        return alignment + self.config['au_uniformity_weight'] * uniformity
    
    def message(self, x_j: Tensor) -> Tensor:
        return x_j
    
    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t,x)
    
class MF(RecModel):
    def __init__(self,
                 num_users:int,
                 num_items:int,
                 config,
                 edge_index:LongTensor):
        super().__init__(num_users,num_items,config,edge_index)
        self.user_degree = degree(self.edge_index[0], self.num_users)
        self.item_degree = degree(self.edge_index[1], self.num_items)
        
    
    def forward(self,edge_index=None):
        return self.user_embedding.weight, self.item_embedding.weight
    
    def get_loss(self,edge_label_index):
        rank_loss = self.bpr_loss(edge_label_index) + self.l2_reg(edge_label_index)
        return rank_loss 

class LightGCN(RecModel):
    def __init__(self,
                 num_users:int,
                 num_items:int,
                 config,
                 edge_index:LongTensor):
        super().__init__(num_users,num_items,config,edge_index)
        self.edge_index = self.get_sparse_graph(edge_index, use_value=False)
        self.edge_index = gcn_norm(self.edge_index)

    
    def forward(self,edge_index):
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        x = torch.cat([user_emb, item_emb], dim=0)
        out = [x]
        for i in range(self.config['K']):
            x = self.propagate(edge_index, x=x)
            out.append(x)
        out = torch.stack(out, dim=1)
        out = out.mean(dim=1)
        user_emb = out[:self.num_users]
        item_emb = out[self.num_users:]
        return user_emb, item_emb

    def get_loss(self,edge_label_index):
        rank_loss = self.bpr_loss(edge_label_index) + self.l2_reg(edge_label_index)
        return rank_loss 

    
class NRGCF(RecModel):
    def __init__(self,
                 num_users:int,
                 num_items:int,
                 config,
                 edge_index:LongTensor):
        super().__init__(num_users,num_items,config,edge_index)
        self.edge_index = self.get_sparse_graph(edge_index, use_value=False)
        self.edge_index = gcn_norm(self.edge_index)
        self.lambda_ = config['lambda']
        self.representation_modulation_mode = config.get(
            'representation_modulation_mode', 'original_stage_two'
        )
        if self.representation_modulation_mode == 'paper_stage_two':
            # Backward-compatible alias used by the first stage-two pilot.
            self.representation_modulation_mode = 'original_stage_two'
        valid_modulation_modes = {
            'none', 'legacy_always', 'original_always', 'blend_always',
            'original_stage_two', 'reliability_weighted_always',
            'reliability_weighted_stage_two',
        }
        if self.representation_modulation_mode not in valid_modulation_modes:
            raise ValueError(
                'Unsupported representation modulation mode: '
                + str(self.representation_modulation_mode)
            )
        if (self.representation_modulation_mode == 'blend_always'
                and not 0.0 <= float(self.lambda_) <= 1.0):
            raise ValueError(
                'blend_always requires lambda_ within [0, 1]'
            )
        self.modulation_ramp_epochs = int(
            config.get('representation_modulation_ramp_epochs', 0)
        )
        if self.modulation_ramp_epochs < 0:
            raise ValueError('representation modulation ramp cannot be negative')
        self.modulation_filtering_epoch = None
        self.modulation_active = (
            self.representation_modulation_mode in (
                'legacy_always', 'original_always', 'blend_always',
                'reliability_weighted_always',
            )
        )
        self.modulation_progress = 1.0 if self.modulation_active else 0.0
        self.last_user_block_rms = None
        self.last_item_block_rms = None
        self.last_modulation_layer_scales = []
        self.register_buffer(
            'user_modulation_weight', torch.ones(num_users, dtype=torch.float32)
        )
        self.register_buffer(
            'item_modulation_weight', torch.ones(num_items, dtype=torch.float32)
        )
        self.momentum_loss = torch.zeros(edge_index.size(1)).to(device)
        self.active_edge_count = int(edge_index.size(1))
        self.objective_message_dropout = float(
            config.get('objective_message_dropout', 0.0)
        )
        raw_user_degree = torch.bincount(
            edge_index[0], minlength=num_users
        ).to(torch.float32)
        if config.get('training_objective') == 'adap_tau':
            degree_quantile = float(
                config.get('adap_tau_degree_quantile', 0.2)
            )
            degree_threshold = torch.quantile(
                raw_user_degree, degree_quantile
            )
            high_degree_user_mask = raw_user_degree > degree_threshold
            high_edge_mask = high_degree_user_mask[edge_index[0]]
            self.adap_tau_high_degree_user_count = int(
                high_degree_user_mask.sum().item()
            )
            self.adap_tau_high_degree_interaction_count = int(
                high_edge_mask.sum().item()
            )
            initial_inverse_temperature = (
                initial_adap_tau_inverse_temperature(
                    high_degree_user_count=(
                        self.adap_tau_high_degree_user_count
                    ),
                    high_degree_interaction_count=(
                        self.adap_tau_high_degree_interaction_count
                    ),
                    num_items=num_items,
                    assumed_positive_gap=float(
                        config.get('adap_tau_initial_positive_gap', 0.7)
                    ),
                )
            )
        else:
            high_degree_user_mask = torch.zeros(
                num_users, dtype=torch.bool, device=edge_index.device
            )
            self.adap_tau_high_degree_user_count = 0
            self.adap_tau_high_degree_interaction_count = 0
            initial_inverse_temperature = 10.0
        self.register_buffer(
            'adap_tau_high_degree_user_mask', high_degree_user_mask
        )
        self.register_buffer(
            'adap_tau_memory',
            torch.full((num_users,), float(initial_inverse_temperature)),
        )
        self.register_buffer(
            'adap_tau_previous_user_loss',
            torch.zeros(num_users, dtype=torch.float32),
        )
        self.adap_tau_has_previous_user_loss = False
        self.adap_tau_initial_inverse_temperature = float(
            initial_inverse_temperature
        )
        self.adap_tau_current_positive_inverse_temperature = float(
            initial_inverse_temperature
        )
        self.last_objective_epoch_state = None

    def objective_metadata(self):
        name = str(self.config.get('training_objective', 'bpr'))
        initialization = {
            'name': (
                'normal' if self.config['init'] == 'normal'
                else 'xavier_uniform'
            ),
            'scale_or_gain': float(self.config['init_weight']),
        }
        if name == 'bpr':
            return {
                'name': 'bpr',
                'description': 'Mean pairwise BPR softplus plus ego-embedding L2.',
                'regularization': 'ego_embedding_l2',
                'embedding_initialization': initialization,
            }
        if name == 'ssm':
            return {
                'name': 'ssm',
                'description': (
                    'Adap_tau-reference SSM with B-1 in-batch negatives, a '
                    'negative-only denominator, and all-layer batch L2.'
                ),
                'configured_num_neg_ignored': int(self.config['num_neg']),
                'tau': float(self.config['tau']),
                'positive_in_denominator': False,
                'negative_sampling': 'other_positive_items_in_same_batch',
                'negative_count': 'batch_size_minus_one',
                'batch_order': 'numpy_shuffle_then_consecutive_slices',
                'regularization': 'selected_user_and_positive_item_all_layers_l2',
                'message_dropout': self.objective_message_dropout,
                'evaluation_scoring': 'raw_propagated_embedding_dot_product',
                'evaluation_protocol': (
                    'mask_training_edges_and_select_best_test_recall_at_20'
                ),
                'embedding_initialization': initialization,
            }
        if name == 'au':
            return {
                'name': 'au',
                'description': (
                    'Normalized positive alignment plus weighted sum of '
                    'within-batch user and item uniformity.'
                ),
                'uniformity_weight': float(
                    self.config['au_uniformity_weight']
                ),
                'uniformity_t': float(self.config['au_uniformity_t']),
                'uniformity_sides': 'user_plus_item',
                'regularization': 'none',
                'embedding_initialization': initialization,
            }
        if name == 'adap_tau':
            return {
                'name': 'adap_tau',
                'description': (
                    'Adap_tau-reference adaptive inverse-temperature SSM '
                    'with B-1 in-batch negatives and all-layer batch L2.'
                ),
                'mode': self.config['adap_tau_mode'],
                'configured_num_neg_ignored': int(self.config['num_neg']),
                'negative_sampling': 'other_positive_items_in_same_batch',
                'negative_count': 'batch_size_minus_one',
                'batch_order': 'numpy_shuffle_then_consecutive_slices',
                'positive_in_denominator': False,
                'temperature_2': float(
                    self.config['adap_tau_temperature_2']
                ),
                'loss_quantile': float(
                    self.config['adap_tau_loss_quantile']
                ),
                'recalibration_epoch_zero_based': int(
                    self.config['adap_tau_recalibration_epoch']
                ),
                'degree_quantile': float(
                    self.config['adap_tau_degree_quantile']
                ),
                'initial_positive_gap': float(
                    self.config['adap_tau_initial_positive_gap']
                ),
                'initial_inverse_temperature': (
                    self.adap_tau_initial_inverse_temperature
                ),
                'regularization': 'selected_user_and_positive_item_all_layers_l2',
                'message_dropout': self.objective_message_dropout,
                'evaluation_scoring': 'raw_propagated_embedding_dot_product',
                'evaluation_protocol': (
                    'mask_training_edges_and_select_best_test_recall_at_20'
                ),
                'lambert_w_implementation': (
                    'principal_real_branch_direct_halley_iterations'
                ),
                'reference_numeric_difference': (
                    'direct Lambert W replaces the reference discretized '
                    'SciPy lookup table; the mathematical mapping is unchanged'
                ),
                'embedding_initialization': initialization,
                'last_epoch_state': self.last_objective_epoch_state,
            }
        raise ValueError('Unsupported training objective: ' + name)

    @torch.no_grad()
    def set_training_graph(self, edge_index, edge_weight=None):
        """Rebuild and normalize the graph used by stage-two propagation."""
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if edge_index.size(1) == 0:
            raise ValueError("NR-GCF stage-two graph cannot be empty")
        edge_index = edge_index.to(self.user_embedding.weight.device)
        if edge_weight is not None:
            edge_weight = edge_weight.detach().to(
                device=edge_index.device,
                dtype=self.user_embedding.weight.dtype,
            )
            if edge_weight.dim() != 1 or edge_weight.numel() != edge_index.size(1):
                raise ValueError("edge_weight must contain one value per edge")
            if not bool(torch.isfinite(edge_weight).all()):
                raise ValueError("edge_weight contains NaN or Inf")
            if bool((edge_weight < 0).any()):
                raise ValueError("edge_weight must be non-negative")
            graph = self.get_sparse_graph(
                edge_index, use_value=True, value=edge_weight
            )
        else:
            graph = self.get_sparse_graph(edge_index, use_value=False)
        self.edge_index = gcn_norm(graph)
        self.active_edge_count = int(edge_index.size(1))

    
    @torch.no_grad()
    def activate_stage_two_modulation(
            self, filtering_epoch, user_weight=None, item_weight=None):
        """Activate the paper's post-filter modulation stage.

        Reliability weights are frozen diagnostics computed without labels.
        They affect only the global cross-type scale estimator, never BPR or
        message-passing edge weights.
        """
        if self.representation_modulation_mode in (
                'none', 'legacy_always', 'original_always', 'blend_always'):
            return
        weighted = (
            self.representation_modulation_mode in (
                'reliability_weighted_always',
                'reliability_weighted_stage_two',
            )
        )
        if weighted and (user_weight is None or item_weight is None):
            raise ValueError(
                'Reliability-weighted modulation requires user/item weights'
            )
        if not weighted:
            user_weight = torch.ones_like(self.user_modulation_weight)
            item_weight = torch.ones_like(self.item_modulation_weight)
        for name, value, target in (
                ('user_weight', user_weight, self.user_modulation_weight),
                ('item_weight', item_weight, self.item_modulation_weight)):
            value = value.detach().to(device=target.device, dtype=target.dtype)
            if value.shape != target.shape:
                raise ValueError(
                    '%s must have shape %s' % (name, tuple(target.shape))
                )
            if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
                raise ValueError('%s must be finite and non-negative' % name)
            if float(value.sum().item()) <= 0.0:
                raise ValueError('%s must contain positive mass' % name)
            target.copy_(value)
        self.modulation_filtering_epoch = int(filtering_epoch)
        self.modulation_active = True
        if self.representation_modulation_mode == 'reliability_weighted_always':
            # Cross norm is already active.  Filtering changes only the frozen
            # RMS estimator weights, so there is no on/off operator switch.
            self.modulation_progress = 1.0
            return
        # Stage two starts with the next optimization epoch.  This avoids
        # evaluating a newly switched operator before it has received an update.
        self.modulation_progress = 0.0

    @torch.no_grad()
    def set_training_epoch(self, epoch):
        """Update only the deterministic stage-two interpolation coefficient."""
        if not self.modulation_active:
            return
        if self.representation_modulation_mode in (
                'legacy_always', 'original_always', 'blend_always',
                'reliability_weighted_always'):
            self.modulation_progress = 1.0
            return
        elapsed = int(epoch) - int(self.modulation_filtering_epoch)
        if elapsed <= 0:
            self.modulation_progress = 0.0
        elif self.modulation_ramp_epochs == 0:
            self.modulation_progress = 1.0
        else:
            self.modulation_progress = min(
                1.0, float(elapsed) / float(self.modulation_ramp_epochs)
            )

    def _weighted_rms(self, embeddings, weight):
        # The released NR-GCF code uses this uncapped RMS.  The paper prints
        # min(mean_squared_norm, 1), which becomes an identity divisor whenever
        # the statistic is above one under the released std=1 initialization.
        # We preserve the executable, numerically active scale operation and
        # move it to the post-filter stage requested by this project.
        squared_norm = embeddings.pow(2).sum(dim=1)
        denominator = weight.sum().clamp_min(1e-12)
        mean_squared_norm = (squared_norm * weight).sum() / denominator
        return (mean_squared_norm + 1e-6).sqrt()

    def cross_norm(self,x):
        users,items = torch.split(x,[self.num_users,self.num_items])
        users_norm = self._weighted_rms(users, self.user_modulation_weight)
        items_norm = self._weighted_rms(items, self.item_modulation_weight)
        self.last_user_block_rms = users_norm.detach()
        self.last_item_block_rms = items_norm.detach()
        if not self.training:
            self.last_modulation_layer_scales.append(
                (users_norm.detach(), items_norm.detach())
            )
        users = users / (items_norm)
        items = items / (users_norm)
        x = torch.cat([users,items])
        return x

    @torch.no_grad()
    def modulation_snapshot(self):
        """Return compact evidence of whether modulation is numerically active."""
        def scalar_or_none(value):
            if value is None:
                return None
            return float(value.detach().cpu().item())

        user_rms = scalar_or_none(self.last_user_block_rms)
        item_rms = scalar_or_none(self.last_item_block_rms)
        layer_scales = []
        for layer, (layer_user_rms, layer_item_rms) in enumerate(
                self.last_modulation_layer_scales, 1):
            layer_scales.append({
                'layer': int(layer),
                'user_block_rms': scalar_or_none(layer_user_rms),
                'item_block_rms': scalar_or_none(layer_item_rms),
                'user_embedding_divisor': scalar_or_none(layer_item_rms),
                'item_embedding_divisor': scalar_or_none(layer_user_rms),
            })
        return {
            'active': bool(self.modulation_active),
            'progress': float(self.modulation_progress),
            'effective_strength': float(
                self.modulation_progress * (
                    self.lambda_
                    if self.representation_modulation_mode == 'blend_always'
                    else 1.0
                )
            ),
            'operator': (
                'crossnorm_propagation_blend'
                if self.representation_modulation_mode == 'blend_always'
                else 'direct_crossnorm'
            ),
            'crossnorm_blend_weight': (
                float(self.lambda_)
                if self.representation_modulation_mode == 'blend_always'
                else None
            ),
            'user_block_rms': user_rms,
            'item_block_rms': item_rms,
            'user_embedding_divisor': item_rms,
            'item_embedding_divisor': user_rms,
            'layer_scales': layer_scales,
        }
        
    def _forward_layers(self, edge_index, apply_message_dropout=True):
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        x = torch.cat([user_emb, item_emb], dim=0)
        if not self.training:
            self.last_modulation_layer_scales = []
        out = [x]
        for i in range(self.config['K']):
            x = self.propagate(edge_index, x=x)
            if (apply_message_dropout and self.training
                    and self.objective_message_dropout > 0.0):
                x = F.dropout(
                    x,
                    p=self.objective_message_dropout,
                    training=True,
                )
            modulation_strength = self.modulation_progress
            if modulation_strength != 0.0:
                x_c = self.cross_norm(x)
                if self.representation_modulation_mode == 'blend_always':
                    # Paper-style sensitivity operator.  This is opt-in and
                    # does not alter the main original_always direct operator.
                    x = self.lambda_ * x_c + (1.0 - self.lambda_) * x
                elif modulation_strength == 1.0:
                    # Exact released implementation requested by the project:
                    # propagate -> direct cross_norm -> layer aggregation.
                    x = x_c
                else:
                    # Optional transition only; recommended experiments use
                    # ramp_epochs=0 and therefore never enter this branch.
                    x = modulation_strength * x_c + (1 - modulation_strength) * x
            out.append(x)
        return torch.stack(out, dim=1)

    def forward(self,edge_index):
        out = self._forward_layers(edge_index)
        out = out.mean(dim=1)
        user_emb = out[:self.num_users]
        item_emb = out[self.num_users:]
        return user_emb, item_emb

    def _in_batch_layer_embeddings(self, edge_label_index):
        if edge_label_index.dim() != 2 or edge_label_index.size(0) != 2:
            raise ValueError(
                'reference in-batch objectives require [2, batch_size] '
                'positive interaction indices'
            )
        layers = self._forward_layers(self.edge_index)
        user_layers = layers[:self.num_users][edge_label_index[0]]
        positive_item_layers = layers[self.num_users:][edge_label_index[1]]
        return user_layers, positive_item_layers

    def _reference_all_layer_regularization(
            self, user_layers, positive_item_layers):
        batch_size = user_layers.size(0)
        return self.config['decay'] * 0.5 * (
            user_layers.pow(2).sum()
            + positive_item_layers.pow(2).sum()
        ) / batch_size

    def ssm_loss(self, edge_label_index:LongTensor, return_aux=False):
        """Adap_tau LightGCN SSM: no sampled items, B-1 batch negatives."""
        user_layers, positive_item_layers = (
            self._in_batch_layer_embeddings(edge_label_index)
        )
        user_embedding = user_layers.mean(dim=1)
        positive_item_embedding = positive_item_layers.mean(dim=1)
        instance_loss = ssm_in_batch_instance_loss(
            user_embedding,
            positive_item_embedding,
            temperature=self.config['tau'],
        )
        regularization = self._reference_all_layer_regularization(
            user_layers, positive_item_layers
        )
        total = instance_loss.mean() + regularization
        if return_aux:
            return total, {'instance_loss': instance_loss.detach()}
        return total

    def adap_tau_loss(self, edge_label_index:LongTensor, return_aux=False):
        user_layers, positive_item_layers = (
            self._in_batch_layer_embeddings(edge_label_index)
        )
        user_embedding = user_layers.mean(dim=1)
        positive_item_embedding = positive_item_layers.mean(dim=1)
        user_inverse_temperature = self.adap_tau_memory[
            edge_label_index[0]
        ].detach()
        instance_loss, unit_temperature_loss = (
            adap_tau_in_batch_instance_loss(
                user_embedding,
                positive_item_embedding,
                user_inverse_temperature=user_inverse_temperature,
                positive_inverse_temperature=(
                    self.adap_tau_current_positive_inverse_temperature
                ),
            )
        )
        regularization = self._reference_all_layer_regularization(
            user_layers, positive_item_layers
        )
        total = instance_loss.mean() + regularization
        if return_aux:
            return total, {
                'instance_loss': instance_loss.detach(),
                'unit_temperature_loss': unit_temperature_loss,
            }
        return total

    @torch.no_grad()
    def prepare_objective_epoch(self, train_edge_index, epoch):
        """Refresh Adap-tau state before the first update of an epoch."""
        if self.config.get('training_objective') != 'adap_tau':
            self.last_objective_epoch_state = None
            return
        source_epoch = int(epoch) - 1
        recalibration_epoch = int(
            self.config['adap_tau_recalibration_epoch']
        )
        inverse_temperature = self.adap_tau_initial_inverse_temperature
        calibration_source = 'reference_initial_positive_gap'
        if source_epoch >= recalibration_epoch:
            layers = self._forward_layers(
                self.edge_index, apply_message_dropout=False
            )
            pooled = layers.mean(dim=1)
            users = F.normalize(pooled[:self.num_users], dim=-1)
            items = F.normalize(pooled[self.num_users:], dim=-1)
            high_edge_mask = self.adap_tau_high_degree_user_mask[
                train_edge_index[0]
            ]
            positive_scores = (
                users[train_edge_index[0]]
                * items[train_edge_index[1]]
            ).sum(dim=-1)
            positive_mean = positive_scores[high_edge_mask].mean()
            mean_item = items.mean(dim=0, keepdim=True)
            all_item_mean_score = (
                users[self.adap_tau_high_degree_user_mask]
                @ mean_item.t()
            ).mean()
            positive_gap = positive_mean - all_item_mean_score
            if (not bool(torch.isfinite(positive_gap))
                    or float(positive_gap.item()) <= 0.0):
                raise RuntimeError(
                    'Adap-tau embedding calibration produced a non-positive '
                    'cosine gap'
                )
            c_value = 2.0 * (
                math.log(0.5)
                + math.log(
                    self.adap_tau_high_degree_user_count * self.num_items
                )
                - math.log(self.adap_tau_high_degree_interaction_count)
            )
            inverse_temperature = float(
                c_value / (2.0 * float(positive_gap.item()))
            )
            calibration_source = 'current_embedding_cosine_gap'
        self.adap_tau_current_positive_inverse_temperature = float(
            inverse_temperature
        )
        if self.adap_tau_has_previous_user_loss:
            memory = adap_tau_inverse_temperature(
                previous_user_loss=self.adap_tau_previous_user_loss,
                base_inverse_temperature=inverse_temperature,
                mode=self.config['adap_tau_mode'],
                temperature_2=self.config['adap_tau_temperature_2'],
                loss_quantile=self.config['adap_tau_loss_quantile'],
            )
        else:
            memory = torch.full_like(
                self.adap_tau_memory, float(inverse_temperature)
            )
        if not bool(torch.isfinite(memory).all()) or bool((memory <= 0).any()):
            raise RuntimeError('Adap-tau produced invalid inverse temperatures')
        self.adap_tau_memory.copy_(memory)
        self.last_objective_epoch_state = {
            'epoch': int(epoch),
            'source_epoch_zero_based': source_epoch,
            'positive_inverse_temperature': float(inverse_temperature),
            'user_inverse_temperature_min': float(memory.min().item()),
            'user_inverse_temperature_mean': float(memory.mean().item()),
            'user_inverse_temperature_max': float(memory.max().item()),
            'calibration_source': calibration_source,
            'uses_previous_epoch_user_loss': bool(
                self.adap_tau_has_previous_user_loss
            ),
        }

    @torch.no_grad()
    def finish_objective_epoch(self, loss_sum, observation_count):
        """Store mean unit-temperature loss per user for the next epoch."""
        if self.config.get('training_objective') != 'adap_tau':
            return
        observed = observation_count > 0
        if not bool(observed.any()):
            raise RuntimeError('Adap-tau did not observe any training users')
        user_loss = torch.empty_like(loss_sum)
        user_loss[observed] = (
            loss_sum[observed] / observation_count[observed]
        )
        fallback = user_loss[observed].mean()
        user_loss[~observed] = fallback
        if not bool(torch.isfinite(user_loss).all()):
            raise RuntimeError('Adap-tau per-user loss contains NaN or Inf')
        self.adap_tau_previous_user_loss.copy_(user_loss)
        self.adap_tau_has_previous_user_loss = True
        if self.last_objective_epoch_state is not None:
            self.last_objective_epoch_state.update({
                'observed_user_count': int(observed.sum().item()),
                'unobserved_user_count': int((~observed).sum().item()),
                'unit_temperature_user_loss_mean': float(
                    user_loss[observed].mean().item()
                ),
            })

    def objective_epoch_snapshot(self):
        if self.last_objective_epoch_state is None:
            return None
        return dict(self.last_objective_epoch_state)
    
    def get_loss(self,edge_label_index, return_aux=False):
        objective = self.config.get('training_objective', 'bpr')
        if objective == 'bpr':
            loss = self.bpr_loss(edge_label_index) + self.l2_reg(edge_label_index)
            return (loss, None) if return_aux else loss
        if objective == 'ssm':
            return self.ssm_loss(edge_label_index, return_aux=return_aux)
        if objective == 'au':
            loss = self.au_loss(edge_label_index)
            return (loss, None) if return_aux else loss
        if objective == 'adap_tau':
            return self.adap_tau_loss(
                edge_label_index, return_aux=return_aux
            )
        raise ValueError('Unsupported training objective: ' + str(objective))
    
    @torch.no_grad()
    def get_instance_loss(self,edge_label_index:LongTensor):
        user_emb,item_emb = self.forward(edge_index=self.edge_index)
        user_emb = user_emb[edge_label_index[0]]
        pos_item_emb = item_emb[edge_label_index[1]]
        neg_item_emb = item_emb[edge_label_index[2]]
        pos_rank = (user_emb * pos_item_emb).sum(dim=-1)
        neg_rank = (user_emb * neg_item_emb).sum(dim=-1)
        return F.softplus(neg_rank - pos_rank)
    
    @torch.no_grad()
    def update_momentum(self, index, instance_loss:torch.Tensor, epoch:int):
        r"""
        \mathcal{L}^h_{i,0} = \mathcal{L}_{i,0}
        \mathcal{L}^h_{i,t} = (t/T) * \mathcal{L}^h_{i,t-1} + (1 - t/T) * \mathcal{L}_{i,t}
        """
        if epoch == 0:
            self.momentum_loss[index] = instance_loss 
        else:
            w = epoch / 10 
            prev = self.momentum_loss[index]
            self.momentum_loss[index] = w * prev + (1.0 - w) * instance_loss

class NRGCL(RecModel):
    #InfoNCE + NRGCF
    #We use SGL as baseline to implement NR-GCL
    def __init__(self,
                 num_users:int,
                 num_items:int,
                 config,
                 edge_index:LongTensor):
        super().__init__(num_users,num_items,config,edge_index)
        self.edge_index1 = None
        self.edge_index2 = None
        self.aug_type = config['type']
        self.generate_graph(edge_index)
        self.edge_index = self.get_sparse_graph(edge_index, use_value=False)
        self.edge_index = gcn_norm(self.edge_index)
        self.ssl_tmp = config['ssl_tmp']
        self.ssl_decay = config['ssl_decay']
        self.lambda_ = config['lambda']
        self.momentum_loss = torch.zeros(edge_index.size(1)).to(device)

        
    def cross_norm(self,x):
        users,items = torch.split(x,[self.num_users,self.num_items])
        users_norm = (1e-6 + users.pow(2).sum(dim=1).mean()).sqrt()
        items_norm = (1e-6 + items.pow(2).sum(dim=1).mean()).sqrt()
        users = users / (items_norm)
        items = items / (users_norm)
        x = torch.cat([users,items])
        return x
    def forward(self,edge_index:SparseTensor):
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        x = torch.cat([user_emb, item_emb], dim=0)
        out = [x]
        for i in range(self.config['K']):
            x = self.propagate(edge_index, x=x)
            x_c = self.cross_norm(x)
            x = self.lambda_ * x_c + (1 - self.lambda_) * x
            out.append(x)
        out = torch.stack(out, dim=1)
        out = out.mean(dim=1)
        user_emb = out[:self.num_users]
        item_emb = out[self.num_users:]
        return user_emb, item_emb
    
    def generate_graph(self,edge_index:LongTensor):
        if self.aug_type == 'ED':
            self.edge_index1,_ = dropout_edge(edge_index=edge_index,
                                         p=self.config['drop_ratio'])
            self.edge_index2,_ = dropout_edge(edge_index=edge_index,
                                         p=self.config['drop_ratio'])
        if self.aug_type == 'ND':
            self.edge_index1 = dropout_node_bipartite(edge_index=edge_index,
                                                 num_users=self.num_users,
                                                 num_items=self.num_items,
                                                 p=self.config['drop_ratio']/2)
            self.edge_index2 = dropout_node_bipartite(edge_index=edge_index,
                                                 num_users=self.num_users,
                                                 num_items=self.num_items,
                                                 p=self.config['drop_ratio']/2)
        if self.aug_type == 'RW':
            self.edge_index1,_ = dropout_path(edge_index=edge_index,
                                         p=self.config['drop_ratio'])
            self.edge_index2,_ = dropout_path(edge_index=edge_index,
                                         p=self.config['drop_ratio'])
        self.edge_index1 = self.get_sparse_graph(self.edge_index1, use_value=False)
        self.edge_index2 = self.get_sparse_graph(self.edge_index2, use_value=False)
        self.edge_index1 = gcn_norm(self.edge_index1)
        self.edge_index2 = gcn_norm(self.edge_index2)
        
    def InfoNCE(self,
                edge_label_index:LongTensor):
        info_out_u_1,info_out_i_1 = self.forward(edge_index=self.edge_index1)
        info_out_u_2,info_out_i_2 = self.forward(edge_index=self.edge_index2)
        u_idx = torch.unique(edge_label_index[0])
        i_idx = torch.unique(edge_label_index[1])
        info_out_u1 = info_out_u_1[u_idx]
        info_out_u2 = info_out_u_2[u_idx]
        info_out_i1 = info_out_i_1[i_idx]
        info_out_i2 = info_out_i_2[i_idx]
        info_out_u1 = F.normalize(info_out_u1,dim=1)
        info_out_u2 = F.normalize(info_out_u2,dim=1)
        info_out_u_2 = F.normalize(info_out_u_2,dim=1)
        info_out_i_2 = F.normalize(info_out_i_2,dim=1)
        info_pos_user = (info_out_u1 * info_out_u2).sum(dim=1)/ self.ssl_tmp
        info_pos_user = torch.exp(info_pos_user)
        info_neg_user = (info_out_u1 @ info_out_u_2.t())/ self.ssl_tmp
        info_neg_user = torch.exp(info_neg_user)
        info_neg_user = torch.sum(info_neg_user,dim=1,keepdim=True)
        info_neg_user = info_neg_user.T
        ssl_logits_user = -torch.log(info_pos_user / info_neg_user).mean()
        info_out_i1 = F.normalize(info_out_i1,dim=1)
        info_out_i2 = F.normalize(info_out_i2,dim=1)
        info_pos_item = (info_out_i1 * info_out_i2).sum(dim=1)/ self.ssl_tmp
        info_neg_item = (info_out_i1 @ info_out_i_2.t())/ self.ssl_tmp
        info_pos_item = torch.exp(info_pos_item)
        info_neg_item = torch.exp(info_neg_item)
        info_neg_item = torch.sum(info_neg_item,dim=1,keepdim=True)
        info_neg_item = info_neg_item.T
        ssl_logits_item = -torch.log(info_pos_item / info_neg_item).mean()
        return self.ssl_decay * (ssl_logits_user + ssl_logits_item)

    def get_loss(self, edge_label_index):
        return self.bpr_loss(edge_label_index) + self.l2_reg(edge_label_index) + self.InfoNCE(edge_label_index)

            

    
    
