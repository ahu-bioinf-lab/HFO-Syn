import torch
from torch import nn


def _batch_to_gcn_args(drug_graph):
    x = drug_graph.x
    edge_index = drug_graph.edge_index
    if hasattr(drug_graph, 'batch') and drug_graph.batch is not None:
        ibatch = drug_graph.batch
    else:
        ibatch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
    return x, edge_index, ibatch


def _ctx_to_device(ctx, device):
    out = {}
    for k, v in ctx.items():
        if isinstance(v, list):
            out[k] = [t.to(device) for t in v]
        else:
            out[k] = v.to(device)
    return out


def build_drug_fea_extract(args, net_params, dataset=None):
    """GraphNet 分支依赖 dataset.gs_resource（GraphSynergyResource）。"""
    from .gcn_net import gcn

    model_gcn = gcn(args, net_params['gcn_drug'])
    model_gn = None
    if getattr(args, 'use_dual_drug_gcn', False) and dataset is not None:
        res = getattr(dataset, 'gs_resource', None)
        if res is not None:
            from .graph_net import GraphNet

            model_gn = GraphNet(args, res)
    return drug_fea_extract(args, model_gcn, model_gn)


class drug_fea_extract(nn.Module):
    def __init__(self, args, model_gcn, model_graph_net=None):
        super(drug_fea_extract, self).__init__()
        self.gcn = model_gcn
        self.graph_net = model_graph_net
        self.liner = nn.Linear(args.drug_dim * 2, args.drug_dim)
        if self.graph_net is not None:
            self.fuse = nn.Linear(args.drug_dim * 2, args.drug_dim)
        self.use_hetero = getattr(args, "use_hetero_graph_encoder", False)
        if self.use_hetero:
            self.fuse_hetero = nn.Linear(args.drug_dim * 2, args.drug_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.gcn.reset_parameters()
        if self.graph_net is not None:
            self.graph_net.reset_parameters()
        nn.init.xavier_uniform_(self.liner.weight)
        nn.init.zeros_(self.liner.bias)
        if self.graph_net is not None:
            nn.init.xavier_uniform_(self.fuse.weight)
            nn.init.zeros_(self.fuse.bias)
        if self.use_hetero:
            nn.init.xavier_uniform_(self.fuse_hetero.weight, gain=0.25)
            nn.init.zeros_(self.fuse_hetero.bias)

    @staticmethod
    def _pair_combine(druga_vec, drugb_vec, liner):
        dc_fea_add = torch.add(druga_vec, drugb_vec)
        dc_fea_min = torch.minimum(druga_vec, drugb_vec)
        return liner(torch.cat((dc_fea_add, dc_fea_min), dim=-1))

    def _encode_gcn(self, drug_batch):
        x, ei, ib = _batch_to_gcn_args(drug_batch)
        return self.gcn(x, ei, ib)

    def _fuse_hetero_gcn(self, gcn_vec, hetero_vec):
        return self.fuse_hetero(torch.cat((gcn_vec, hetero_vec), dim=-1))

    def forward(self, druga_fea, drugb_fea, hetero_drug_emb=None):
        druga_g = self._encode_gcn(druga_fea)
        drugb_g = self._encode_gcn(drugb_fea)
        if (
            self.use_hetero
            and hetero_drug_emb is not None
            and hasattr(druga_fea, "raw_drug_idx")
            and hasattr(drugb_fea, "raw_drug_idx")
        ):
            dev = druga_g.device
            ia = druga_fea.raw_drug_idx.to(dev)
            ib = drugb_fea.raw_drug_idx.to(dev)
            ha = hetero_drug_emb[ia]
            hb = hetero_drug_emb[ib]
            druga_g = self._fuse_hetero_gcn(druga_g, ha)
            drugb_g = self._fuse_hetero_gcn(drugb_g, hb)

        dc_gcn = self._pair_combine(druga_g, drugb_g, self.liner)

        if self.graph_net is None or not hasattr(druga_fea, 'gs_context') or druga_fea.gs_context is None:
            return dc_gcn

        ctx = _ctx_to_device(druga_fea.gs_context, dc_gcn.device)
        dc_gs = self.graph_net(ctx)
        return self.fuse(torch.cat((dc_gcn, dc_gs), dim=-1))
