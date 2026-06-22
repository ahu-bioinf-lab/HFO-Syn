import copy
import torch
import numpy as np
import pandas as pd
from torch.nn import BatchNorm2d, AdaptiveAvgPool2d, AdaptiveAvgPool1d, AdaptiveMaxPool2d, AdaptiveMaxPool1d
from torch import nn
#fzfrom CASynergy.utils.utils import reset
from ..layers.gcn_layer import GCNConv
from functools import partial

    
import torch
import torch.nn as nn
from torch.nn import AdaptiveAvgPool1d


def _no_cal_cline_outputs(cline_fea_x, batch_size: int, cline_dim: int):
    """跳过 CAL 时，将 [B, N, *] 细胞系张量池化为与 CAL 输出同形状的 cline_c/b/d。"""
    xv = cline_fea_x
    bs, cd = int(batch_size), int(cline_dim)
    if xv.dim() == 3:
        cline_vec = xv.squeeze(-1) if xv.size(-1) == 1 else xv.mean(dim=-1)
    else:
        cline_vec = xv.view(bs, cd, -1).mean(dim=-1)
    cline_c = cline_b = cline_d = cline_vec
    att = torch.zeros(bs, 1, device=xv.device, dtype=xv.dtype)
    loss1 = torch.tensor(0.0, device=xv.device, dtype=xv.dtype)
    return cline_c, cline_b, cline_d, att, loss1

class HFO_Syn(nn.Module):
    def __init__(
        self,
        args,
        drug_fea_extractor,
        cline_fea_extractor,
        cross_att,
        cal,
        predictor,
        hetero_encoder=None,
        cline_gcn_hidden=None,
    ):
        super(HFO_Syn, self).__init__()
        self.args = args
        self.drug_fea_extractor = drug_fea_extractor
        self.cline_fea_extractor = cline_fea_extractor
        self.cross_att = cross_att
        self.AvgPool = AdaptiveAvgPool1d(1)
        self.cal = cal  # args.use_cal=False 时为 None
        self.predictor = predictor
        self.hetero_encoder = hetero_encoder

        h_c = cline_gcn_hidden if cline_gcn_hidden is not None else 1
        if hetero_encoder is not None:
            d_h = hetero_encoder.out_channels
            self.hetero_cell_fuse = nn.Linear(h_c + d_h, h_c)
            nn.init.xavier_uniform_(self.hetero_cell_fuse.weight, gain=0.25)
            nn.init.zeros_(self.hetero_cell_fuse.bias)
        else:
            self.hetero_cell_fuse = None

        self.GCSM =GCSM(
            drug_dim=256, 
            cline_dim=args.cline_dim,
            #latent_dim=128
        )

    def reset_parameters(self):
        self.drug_fea_extractor.reset_parameters()
        self.cline_fea_extractor.reset_parameters()
        self.cross_att.reset_parameters()
        if self.cal is not None:
            self.cal.reset_parameters()
        self.predictor.reset_parameters()
        if self.hetero_encoder is not None:
            self.hetero_encoder.reset_parameters()
        if self.hetero_cell_fuse is not None:
            nn.init.xavier_uniform_(self.hetero_cell_fuse.weight, gain=0.25)
            nn.init.zeros_(self.hetero_cell_fuse.bias)

    def forward(self, druga_fea, drugb_fea, cline_fea, cline_mask, eval_random=True, edge_mask_dict=None):
        hetero_drug_emb = None
        hetero_cell_emb = None
        
        if self.hetero_encoder is not None:
            device = next(self.parameters()).device
            
            # ==== 核心修改：如果传入了 edge_mask_dict，透传给异构编码器 ====
            if edge_mask_dict is not None:
                emb = self.hetero_encoder(device, edge_mask_dict=edge_mask_dict)
            else:
                emb = self.hetero_encoder(device)
            # ==========================================================
            
            hetero_drug_emb = emb[str(self.hetero_encoder.drug_key)]
            hetero_cell_emb = emb[str(self.hetero_encoder.cell_key)]

        # 1. 药物特征提取（GCN 与异构图药物嵌入融合在 drug_fea_extractor 内）
        dc_fea = self.drug_fea_extractor(druga_fea, drugb_fea, hetero_drug_emb=hetero_drug_emb)

        # 2. 细胞系原始特征处理
        cline_fea.x = cline_fea.x.reshape(self.args.batch_size * self.args.cline_dim, -1)
        cline_fea.x = self.cline_fea_extractor(cline_fea).reshape(self.args.batch_size, self.args.cline_dim, -1)

        # 2b. 异构图细胞系嵌入与基因表达 GCN 后的节点特征融合
        if (
            self.hetero_cell_fuse is not None
            and hetero_cell_emb is not None
            and hasattr(cline_fea, "cell_line_idx")
        ):
            dev = cline_fea.x.device
            idx = cline_fea.cell_line_idx.to(dev)
            c_hetero = hetero_cell_emb[idx]
            c_exp = c_hetero.unsqueeze(1).expand(-1, cline_fea.x.size(1), -1)
            cline_fea.x = self.hetero_cell_fuse(torch.cat([cline_fea.x, c_exp], dim=-1))
            
        use_cal = getattr(self.args, "use_cal", False) and self.cal is not None
        if use_cal:
            cline_fea.x = cline_fea.x.reshape(self.args.batch_size * self.args.cline_dim, -1)
            cline_c, cline_b, cline_d, att, loss1 = self.cal(
                cline_fea, cline_mask, eval_random=eval_random
            )
        else:
            cline_c, cline_b, cline_d, att, loss1 = _no_cal_cline_outputs(
                cline_fea.x, self.args.batch_size, self.args.cline_dim
            )

        refined_c = self.GCSM(dc_fea, cline_c)

        score_c, score_d, score_b = self.predictor(refined_c, cline_b, cline_d, dc_fea)

        return score_c, score_d, score_b, att, loss1
    
class GCSM(nn.Module):
    def __init__(self, drug_dim, cline_dim, head_dim=128):
        super().__init__()
        
        self.gate_net = nn.Sequential(
            nn.Linear(drug_dim, cline_dim),
            nn.LayerNorm(cline_dim),
            nn.GELU(),
            nn.Linear(cline_dim, cline_dim),
            nn.Sigmoid()
        )
        
        self.query_net = nn.Linear(drug_dim, head_dim)
        self.key_net = nn.Linear(cline_dim, head_dim)
        self.value_net = nn.Linear(cline_dim, head_dim)
        
        self.output_net = nn.Sequential(
            nn.Linear(head_dim, cline_dim),
            nn.Dropout(0.2),
            nn.Tanh() 
        )
        
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.lmbda = nn.Parameter(torch.tensor(0.01))
        self.scale = head_dim ** 0.5

    def forward(self, dc_fea, cline_c):
        mask = self.gate_net(dc_fea)
       
        q = self.query_net(dc_fea).unsqueeze(1) 
        k = self.key_net(cline_c).unsqueeze(1)
        v = self.value_net(cline_c).unsqueeze(1)
        
    
        attn_weights = torch.softmax((q @ k.transpose(-2, -1)) / self.scale, dim=-1)
        redundant_features = self.output_net((attn_weights @ v).squeeze(1))
        
        residual = cline_c +self.alpha * redundant_features
        
        c_unique = torch.sign(residual) * torch.relu(torch.abs(residual) - self.lmbda)
   
        return c_unique * mask