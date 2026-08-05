from torch import nn,Tensor,LongTensor
from torch_geometric.utils import degree
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import SparseTensor
from torch_sparse import SparseTensor,matmul
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch.nn.functional as F
import torch
import world
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

    def ssm_loss(self,edge_label_index:LongTensor):
        user_emb,item_emb = self.forward(edge_index=None)
        neg_edge_index = torch.randint(0, self.num_items,(edge_label_index[1].numel(),world.num_neg), device=device)
        embedding = torch.cat([user_emb[edge_label_index[0]],
                               item_emb[edge_label_index[1]],
                               item_emb[neg_edge_index].view(-1, item_emb.size(-1))])
        regularization = self.config['decay'] * (1/2) * embedding.norm(p=2).pow(2)/ edge_label_index.size(1)
        user_emb = user_emb[edge_label_index[0]]
        pos_item_emb = item_emb[edge_label_index[1]]
        neg_item_emb = item_emb[neg_edge_index]
        user_emb = F.normalize(user_emb, dim=-1)
        item_emb = torch.cat([pos_item_emb.unsqueeze(1), neg_item_emb], dim=1)
        item_emb = F.normalize(item_emb, dim=-1)
        # user_emb = self.dropout(user_emb)
        y_pred = torch.bmm(item_emb, user_emb.unsqueeze(-1)).squeeze(-1)
        pos_logits = torch.exp(y_pred[:, 0] / self.config['tau']) 
        neg_logits = torch.exp(y_pred[:, 1:]/ self.config['tau']) 
        Ng = neg_logits.sum(dim=-1)
        loss = (- torch.log(pos_logits / Ng))
        return loss.mean() + regularization
    
    def alignment_loss(self,edge_label_index:LongTensor):
        user_emb,item_emb = self.forward(edge_index=None)
        user_emb = user_emb[edge_label_index[0]]
        item_emb = item_emb[edge_label_index[1]]
        user_emb = F.normalize(user_emb, dim=-1)
        item_emb = F.normalize(item_emb, dim=-1)
        return (user_emb - item_emb).norm(dim=1).pow(2).mean()
    
    def uniformity(self,x, t=2):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
    
    def uniformity_loss(self,edge_label_index:LongTensor):
        user_emb,item_emb = self.forward(edge_index=None)
        user_emb = user_emb[edge_label_index[0]]
        item_emb = item_emb[edge_label_index[1]]
        return   (self.uniformity(user_emb) + self.uniformity(item_emb))
    
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
            'representation_modulation_mode', 'paper_stage_two'
        )
        valid_modulation_modes = {
            'none', 'legacy_always', 'paper_stage_two',
            'reliability_weighted_stage_two',
        }
        if self.representation_modulation_mode not in valid_modulation_modes:
            raise ValueError(
                'Unsupported representation modulation mode: '
                + str(self.representation_modulation_mode)
            )
        self.modulation_ramp_epochs = int(
            config.get('representation_modulation_ramp_epochs', 0)
        )
        if self.modulation_ramp_epochs < 0:
            raise ValueError('representation modulation ramp cannot be negative')
        self.modulation_filtering_epoch = None
        self.modulation_active = (
            self.representation_modulation_mode == 'legacy_always'
        )
        self.modulation_progress = 1.0 if self.modulation_active else 0.0
        self.register_buffer(
            'user_modulation_weight', torch.ones(num_users, dtype=torch.float32)
        )
        self.register_buffer(
            'item_modulation_weight', torch.ones(num_items, dtype=torch.float32)
        )
        self.momentum_loss = torch.zeros(edge_index.size(1)).to(device)
        self.active_edge_count = int(edge_index.size(1))

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
        if self.representation_modulation_mode in ('none', 'legacy_always'):
            return
        weighted = (
            self.representation_modulation_mode
            == 'reliability_weighted_stage_two'
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
        # Stage two starts with the next optimization epoch.  This avoids
        # evaluating a newly switched operator before it has received an update.
        self.modulation_progress = 0.0

    @torch.no_grad()
    def set_training_epoch(self, epoch):
        """Update only the deterministic stage-two interpolation coefficient."""
        if not self.modulation_active:
            return
        if self.representation_modulation_mode == 'legacy_always':
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

    def _weighted_rms(self, embeddings, weight, cap_at_one):
        squared_norm = embeddings.pow(2).sum(dim=1)
        denominator = weight.sum().clamp_min(1e-12)
        mean_squared_norm = (squared_norm * weight).sum() / denominator
        if cap_at_one:
            # Eq. 10 in the paper caps the cross-type statistic at one.  The
            # explicit legacy mode skips this branch to reproduce old code.
            mean_squared_norm = torch.clamp(mean_squared_norm, max=1.0)
        return (mean_squared_norm + 1e-6).sqrt()

    def cross_norm(self,x):
        users,items = torch.split(x,[self.num_users,self.num_items])
        cap_at_one = self.representation_modulation_mode != 'legacy_always'
        users_norm = self._weighted_rms(
            users, self.user_modulation_weight, cap_at_one
        )
        items_norm = self._weighted_rms(
            items, self.item_modulation_weight, cap_at_one
        )
        users = users / (items_norm)
        items = items / (users_norm)
        x = torch.cat([users,items])
        return x
        
    def forward(self,edge_index):
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        x = torch.cat([user_emb, item_emb], dim=0)
        out = [x]
        for i in range(self.config['K']):
            x = self.propagate(edge_index, x=x)
            modulation_strength = self.lambda_ * self.modulation_progress
            if modulation_strength != 0.0:
                x_c = self.cross_norm(x)
                x = modulation_strength * x_c + (1 - modulation_strength) * x
            out.append(x)
        out = torch.stack(out, dim=1)
        out = out.mean(dim=1)
        user_emb = out[:self.num_users]
        item_emb = out[self.num_users:]
        return user_emb, item_emb
    
    def get_loss(self,edge_label_index):
        rank_loss = self.bpr_loss(edge_label_index) + self.l2_reg(edge_label_index)
        return rank_loss 
    
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

            

    
    
