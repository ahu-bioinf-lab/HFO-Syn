"""
异构图编码实现（独立模块名）。
--hetero_encoder hgt 时使用 SimpleHeteroScatterConv（torch_scatter，不经 PyG HeteroConv/SAGEConv 聚合栈）。
"""
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_scatter import scatter

_LOADER_TAG = "hetero_stack_enc_v1"


def _edge_type_key(et: Union[Tuple[str, str, str], Tuple]) -> str:
    return "{}||{}||{}".format(et[0], et[1], et[2])


# class SimpleHeteroScatterConv(nn.Module):
#     """
#     每条异构边：源端线性变换后按 edge_index 做 scatter-sum 聚合到目标类型节点；
#     输出 = 原特征（残差） + scale * 各关系聚合增量。避免旧版 PyG 里 HeteroConv/SAGEConv + aggregate(None) 崩溃。
#     消息线性用较小 Xavier gain；可学习 scale 抑制高度节点上的聚合爆炸，利于训练初期稳定。
#     """

#     def __init__(self, edge_types: List[Tuple], hidden_channels: int):
#         super().__init__()
#         self.edge_types = list(edge_types)
#         self.lin_src = nn.ModuleDict()
#         for et in self.edge_types:
#             self.lin_src[_edge_type_key(et)] = nn.Linear(hidden_channels, hidden_channels, bias=True)
#         # 初始较小，训练中学位调节各关系对残差的贡献，缓解随机初始化 + sum 聚合导致的发散
#         self.msg_scale = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
#         self.reset_parameters()

#     def reset_parameters(self):
#         for et in self.edge_types:
#             lin = self.lin_src[_edge_type_key(et)]
#             nn.init.xavier_uniform_(lin.weight, gain=0.5)
#             nn.init.zeros_(lin.bias)
#         self.msg_scale.data.fill_(0.25)

#     def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict) -> Dict[str, torch.Tensor]:
#         out = {k: v.clone() for k, v in x_dict.items()}
#         scale = self.msg_scale.to(dtype=out[next(iter(out))].dtype)
#         for et in self.edge_types:
#             ei = edge_index_dict.get(et)
#             if ei is None or ei.numel() == 0:
#                 continue
#             src_t, _, dst_t = et
#             row = ei[0].long()
#             col = ei[1].long()
#             x_src = x_dict[src_t]
#             x_dst = x_dict[dst_t]
#             msg = self.lin_src[_edge_type_key(et)](x_src)[row]
#             dim_size = int(x_dst.size(0))
#             acc = scatter(msg, col, dim=0, dim_size=dim_size, reduce="sum")
#             out[dst_t] = out[dst_t] + scale * acc
#         return out
class SimpleHeteroScatterConv(nn.Module):
    """
    【已改造版】：支持边掩码输入的异构图卷积层，用于梯度归因。
    """
    def __init__(self, edge_types: List[Tuple], hidden_channels: int):
        super().__init__()
        self.edge_types = list(edge_types)
        self.lin_src = nn.ModuleDict()
        for et in self.edge_types:
            self.lin_src[_edge_type_key(et)] = nn.Linear(hidden_channels, hidden_channels, bias=True)
        self.msg_scale = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self):
        for et in self.edge_types:
            lin = self.lin_src[_edge_type_key(et)]
            nn.init.xavier_uniform_(lin.weight, gain=0.5)
            nn.init.zeros_(lin.bias)
        self.msg_scale.data.fill_(0.25)

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict, edge_mask_dict: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        out = {k: v.clone() for k, v in x_dict.items()}
        scale = self.msg_scale.to(dtype=out[next(iter(out))].dtype)
        
        for et in self.edge_types:
            ei = edge_index_dict.get(et)
            if ei is None or ei.numel() == 0:
                continue
            src_t, _, dst_t = et
            row = ei[0].long()
            col = ei[1].long()
            
            x_src = x_dict[src_t]
            x_dst = x_dict[dst_t]
            
            # 1. 计算源节点的特征映射
            msg = self.lin_src[_edge_type_key(et)](x_src)[row]
            
            # ==== 核心归因改造：如果传入了边掩码，在此处与消息相乘 ====
            if edge_mask_dict is not None and et in edge_mask_dict:
                # mask = edge_mask_dict[et]  # 形状为 (num_edges,)
                # # mask 增加一个维度变为 (num_edges, 1)，通过广播机制应用到每条边的特征向量上
                # msg = msg * mask.unsqueeze(-1).to(dtype=msg.dtype)
                mask = edge_mask_dict[et]
    # 关键修复：确保 mask 完全在图中
                mask = mask.to(dtype=msg.dtype, device=msg.device)
                mask = mask.unsqueeze(-1)                    # (num_edges, 1)
                
                # 使用 .mul_() 或 clone 后乘，保证梯度流
                msg = msg * mask
            # ====================================================
            
            dim_size = int(x_dst.size(0))
            acc = scatter(msg, col, dim=0, dim_size=dim_size, reduce="sum")
            out[dst_t] = out[dst_t] + scale * acc
            
        return out

def _torch_load_pt(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _deep_find_heterodata(obj: Any, depth: int = 0, max_depth: int = 8) -> Optional[HeteroData]:
    if isinstance(obj, HeteroData):
        return obj
    if depth >= max_depth:
        return None
    if isinstance(obj, dict):
        for v in obj.values():
            h = _deep_find_heterodata(v, depth + 1, max_depth)
            if h is not None:
                return h
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            h = _deep_find_heterodata(v, depth + 1, max_depth)
            if h is not None:
                return h
    return None


def _extract_hetero_data(loaded, path: str, *, _nested: bool = False) -> HeteroData:
    if isinstance(loaded, HeteroData):
        return loaded
    if isinstance(loaded, dict):
        preferred_keys = (
            "data",
            "hetero",
            "hetero_data",
            "graph",
            "HeteroData",
            "heterograph",
            "hetero_graph",
            "g",
            "hetero_graph_data",
        )
        for k in preferred_keys:
            v = loaded.get(k)
            if isinstance(v, HeteroData):
                print(f"[{_LOADER_TAG}] unwrapped HeteroData from dict key '{k}'")
                return v
        for k, v in loaded.items():
            if isinstance(v, HeteroData):
                print(f"[{_LOADER_TAG}] unwrapped HeteroData from dict key '{k}'")
                return v
        if not _nested:
            for v in loaded.values():
                if isinstance(v, dict):
                    try:
                        return _extract_hetero_data(v, path, _nested=True)
                    except TypeError:
                        continue
        deep = _deep_find_heterodata(loaded)
        if deep is not None:
            print(f"[{_LOADER_TAG}] found nested HeteroData via deep scan")
            return deep
        if hasattr(HeteroData, "from_dict"):
            try:
                h = HeteroData.from_dict(loaded)
                nts = list(getattr(h, "node_types", []) or [])
                if len(nts) > 0:
                    print(f"[{_LOADER_TAG}] reconstructed HeteroData via from_dict()")
                    return h
            except Exception:
                pass
        raise TypeError(
            f"hetero_graph.pt 是 dict，但无法得到 HeteroData。顶层键: {list(loaded.keys())}。"
        )
    raise TypeError(f"Expected HeteroData or dict in {path}, got {type(loaded)}")


def _guess_type(names: List[str], keywords: Tuple[str, ...]) -> Optional[str]:
    for n in names:
        low = n.lower()
        for k in keywords:
            if k in low:
                return n
    return None


def resolve_entity_keys(node_types: List[str]) -> Tuple[str, str]:
    drug_key = _guess_type(list(node_types), ("drug", "compound", "chem"))
    cell_key = _guess_type(list(node_types), ("cell", "cline", "line"))
    if drug_key is None:
        drug_key = node_types[0]
    if cell_key is None:
        cell_key = node_types[1] if len(node_types) > 1 else node_types[0]
    if drug_key == cell_key and len(node_types) > 1:
        cell_key = node_types[1]
    return drug_key, cell_key


def _looks_like_cell_node_type(name: str) -> bool:
    low = name.lower()
    return any(k in low for k in ("cell", "cline", "line"))


def _pathway_cell_fusable(main_enc: "HeteroGraphEncoder", pathway_enc: "HeteroGraphEncoder") -> bool:
    """仅当通路图里解析出的 cell 类型语义像细胞系且节点数与主图一致时，才拼接融合细胞嵌入。"""
    ck_p = str(pathway_enc.cell_key)
    if not _looks_like_cell_node_type(ck_p):
        return False
    ck_m = str(main_enc.cell_key)
    nm = int(main_enc.data_cpu[ck_m].x.size(0))
    np_ = int(pathway_enc.data_cpu[ck_p].x.size(0))
    return nm == np_


class DualFusedHeteroGraphEncoder(nn.Module):
    """
    主异构图 (hetero_graph.pt) + 通路异构图 (pathway_drug_hetero.pt) 各用一套 HeteroGraphEncoder，
    在嵌入空间 concat 后经 Linear 压回 out_channels；对外接口与单路 HeteroGraphEncoder 一致，供 CAESynergy 调用。
    """

    def __init__(self, enc_main: "HeteroGraphEncoder", enc_pathway: "HeteroGraphEncoder"):
        super().__init__()
        self.enc_main = enc_main
        self.enc_pathway = enc_pathway
        oc = int(enc_main.out_channels)
        self.out_channels = oc
        self.drug_key = enc_main.drug_key
        self.cell_key = enc_main.cell_key
        self.drug_fuse = nn.Linear(2 * oc, oc)
        self.use_pathway_cell = _pathway_cell_fusable(enc_main, enc_pathway)
        self.cell_fuse: Optional[nn.Linear]
        if self.use_pathway_cell:
            self.cell_fuse = nn.Linear(2 * oc, oc)
            print(
                f"[{_LOADER_TAG}] dual hetero: fuse drug + cell (pathway cell type={enc_pathway.cell_key!r})"
            )
        else:
            self.cell_fuse = None
            print(f"[{_LOADER_TAG}] dual hetero: fuse drug only (pathway cell fusion skipped)")
        nn.init.xavier_uniform_(self.drug_fuse.weight, gain=0.5)
        nn.init.zeros_(self.drug_fuse.bias)
        if self.cell_fuse is not None:
            nn.init.xavier_uniform_(self.cell_fuse.weight, gain=0.5)
            nn.init.zeros_(self.cell_fuse.bias)

    def reset_parameters(self):
        self.enc_main.reset_parameters()
        self.enc_pathway.reset_parameters()
        nn.init.xavier_uniform_(self.drug_fuse.weight, gain=0.5)
        nn.init.zeros_(self.drug_fuse.bias)
        if self.cell_fuse is not None:
            nn.init.xavier_uniform_(self.cell_fuse.weight, gain=0.5)
            nn.init.zeros_(self.cell_fuse.bias)

    # def forward(self, device: torch.device) -> Dict[str, torch.Tensor]:
    #     emb_m = self.enc_main(device)
    #     emb_p = self.enc_pathway(device)
    #     dk_m = str(self.enc_main.drug_key)
    #     dk_p = str(self.enc_pathway.drug_key)
    #     nd_m = emb_m[dk_m].size(0)
    #     nd_p = emb_p[dk_p].size(0)
    #     if nd_m != nd_p:
    #         raise ValueError(
    #             f"主图与通路图的 drug 节点数不一致: {dk_m} n={nd_m} vs {dk_p} n={nd_p}，请保证两图药物对齐。"
    #         )
    #     fused_d = self.drug_fuse(torch.cat([emb_m[dk_m], emb_p[dk_p]], dim=-1))
    #     out: Dict[str, torch.Tensor] = {str(k): v for k, v in emb_m.items()}
    #     out[dk_m] = fused_d
    #     if self.cell_fuse is not None:
    #         ck_m = str(self.enc_main.cell_key)
    #         ck_p = str(self.enc_pathway.cell_key)
    #         out[ck_m] = self.cell_fuse(torch.cat([emb_m[ck_m], emb_p[ck_p]], dim=-1))
    #     return out

    # def drug_cell_embeddings(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    #     emb = self.forward(device)
    #     return emb[str(self.drug_key)], emb[str(self.cell_key)]
    # 修改后的 forward，添加可选参数 edge_mask_dict
    def forward(self, device: torch.device, edge_mask_dict: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        # 主图保持正常前向传播
        emb_m = self.enc_main(device)
        
        # ==== 核心修改：将梯度掩码精准透传给通路图编码器 ====
        if edge_mask_dict is not None:
            emb_p = self.enc_pathway(device, edge_mask_dict=edge_mask_dict)
        else:
            emb_p = self.enc_pathway(device)
        # ====================================================
        
        dk_m = str(self.enc_main.drug_key)
        dk_p = str(self.enc_pathway.drug_key)
        nd_m = emb_m[dk_m].size(0)
        nd_p = emb_p[dk_p].size(0)
        if nd_m != nd_p:
            raise ValueError(
                f"主图与通路图的 drug 节点数不一致: {dk_m} n={nd_m} vs {dk_p} n={nd_p}，请保证两图药物对齐。"
            )
        fused_d = self.drug_fuse(torch.cat([emb_m[dk_m], emb_p[dk_p]], dim=-1))
        out: Dict[str, torch.Tensor] = {str(k): v for k, v in emb_m.items()}
        out[dk_m] = fused_d
        if self.cell_fuse is not None:
            ck_m = str(self.enc_main.cell_key)
            ck_p = str(self.enc_pathway.cell_key)
            out[ck_m] = self.cell_fuse(torch.cat([emb_m[ck_m], emb_p[ck_p]], dim=-1))
        return out

    # 同步适配包装接口
    def drug_cell_embeddings(self, device: torch.device, edge_mask_dict: Optional[Dict] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.forward(device, edge_mask_dict=edge_mask_dict)
        return emb[str(self.drug_key)], emb[str(self.cell_key)]


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv
from typing import Dict, List, Tuple, Optional

class HeteroGraphEncoder0(nn.Module):
    def __init__(
        self,
        hetero_path: str,
        hidden_channels: int,
        out_channels: int,
        encoder: str = "hgt",
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.2,
        han_metapath_json: Optional[str] = None,
    ):
        super().__init__()
        print(f"[{_LOADER_TAG}] using implementation file: hetero_stack_enc.py")
        
        if not os.path.isfile(hetero_path):
            raise FileNotFoundError(f"hetero graph not found: {hetero_path}")
            
        print(f"[{_LOADER_TAG}] loading {os.path.abspath(hetero_path)}")
        loaded = _torch_load_pt(hetero_path)
        raw = _extract_hetero_data(loaded, hetero_path)
        
        print(
            f"[{_LOADER_TAG}] ok | node_types={list(raw.node_types)} | "
            f"edge_types={len(raw.edge_types)}"
        )

        encoder = encoder.lower().strip()
        
        # HAN 模式保持不变
        if encoder == "han":
            from torch_geometric.transforms import AddMetaPaths
            mp_path = han_metapath_json or os.path.join(
                os.path.dirname(hetero_path), "han_metapaths.json"
            )
            if not os.path.isfile(mp_path):
                raise FileNotFoundError("HAN 需要 han_metapaths.json 文件")
            
            with open(mp_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            metapaths = [[tuple(t) for t in mp_chain] for mp_chain in meta["metapaths"]]
            self.data_cpu = AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=False)(raw)
        else:
            self.data_cpu = raw

        self.metadata = self.data_cpu.metadata()
        self.node_types = list(self.data_cpu.node_types)
        self.edge_types = list(self.data_cpu.edge_types)
        
        self.drug_key, self.cell_key = resolve_entity_keys(self.node_types)
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.encoder = encoder
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.heads = heads

        # 特征投影层
        self.x_projs = nn.ModuleDict()
        for nt in self.node_types:
            if getattr(self.data_cpu[nt], "x", None) is None:
                raise ValueError(f"Node type {nt} has no feature x!")
            d = int(self.data_cpu[nt].x.size(-1))
            self.x_projs[str(nt)] = nn.Linear(d, hidden_channels)

        # 输出投影层
        self.out_lin = nn.ModuleDict({
            str(nt): nn.Linear(hidden_channels, out_channels) 
            for nt in self.node_types
        })

        # ==================== 核心修改：使用标准 HGTConv ====================
        if self.encoder == "hgt":
            print(f"[{_LOADER_TAG}] encoder = Standard HGTConv (layers={num_layers}, heads={heads})")
            self.convs = nn.ModuleList()
            for _ in range(num_layers):
                self.convs.append(
                    HGTConv(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        metadata=self.metadata,      # 关键：传入 metadata
                        heads=heads,
                        group='sum',                 # 或 'mean'
                        # dropout=dropout,           # 可选
                    )
                )
        elif self.encoder == "han":
            from torch_geometric.nn import HANConv
            print(f"[{_LOADER_TAG}] encoder = HANConv (layers={num_layers}, heads={heads})")
            self.convs = nn.ModuleList()
            for _ in range(num_layers):
                self.convs.append(
                    HANConv(hidden_channels, hidden_channels, heads=heads, metadata=self.metadata)
                )
        else:
            raise ValueError(f"Unknown encoder: {encoder}. Use 'hgt' or 'han'.")

        # LayerNorm
        self.conv_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_norms.append(
                nn.ModuleDict({str(nt): nn.LayerNorm(hidden_channels) for nt in self.node_types})
            )

        self._init_hetero_weights()

    def _init_hetero_weights(self):
        for m in self.x_projs.values():
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        for m in self.out_lin.values():
            nn.init.xavier_uniform_(m.weight, gain=0.75)
            nn.init.zeros_(m.bias)

    def reset_parameters(self):
        self._init_hetero_weights()
        for conv in self.convs:
            if hasattr(conv, 'reset_parameters'):
                conv.reset_parameters()
        for norm_layer in self.conv_norms:
            for ln in norm_layer.values():
                ln.reset_parameters()

    def _to_x_dict(self, device: torch.device) -> Dict[str, torch.Tensor]:
        out = {}
        for nt in self.node_types:
            x = self.data_cpu[nt].x.to(device=device, dtype=torch.float32)
            out[str(nt)] = self.x_projs[str(nt)](x)
        return out

    def _to_edge_index_dict(self, device: torch.device) -> Dict:
        ed = {}
        for et in self.edge_types:
            ei = self.data_cpu[et].edge_index
            ed[et] = ei.to(device=device, dtype=torch.long)
        return ed

    def forward(self, device: torch.device, edge_mask_dict: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        x_dict = self._to_x_dict(device)
        edge_index_dict = self._to_edge_index_dict(device)

        for i, conv in enumerate(self.convs):
            # 标准 HGTConv 的调用方式
            x_dict = conv(x_dict, edge_index_dict)
            
            # Norm + Activation + Dropout
            norms = self.conv_norms[i]
            x_dict = {k: norms[k](v) for k, v in x_dict.items()}
            x_dict = {k: F.gelu(v) for k, v in x_dict.items()}
            x_dict = {k: F.dropout(v, p=self.dropout_p, training=self.training) 
                     for k, v in x_dict.items()}

        # 输出投影
        emb = {}
        for nt in self.node_types:
            k = str(nt)
            emb[k] = self.out_lin[k](x_dict[k])

        return emb

    def drug_cell_embeddings(self, device: torch.device, edge_mask_dict: Optional[Dict] = None):
        emb = self.forward(device, edge_mask_dict=edge_mask_dict)
        return emb[str(self.drug_key)], emb[str(self.cell_key)]

class HeteroGraphEncoder(nn.Module):
    def __init__(
        self,
        hetero_path: str,
        hidden_channels: int,
        out_channels: int,
        encoder: str = "hgt",
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.2,
        han_metapath_json: Optional[str] = None,
    ):
        super().__init__()
        print(f"[{_LOADER_TAG}] using implementation file: hetero_stack_enc.py")

        if not os.path.isfile(hetero_path):
            raise FileNotFoundError(f"hetero graph not found: {hetero_path}")

        print(f"[{_LOADER_TAG}] loading {os.path.abspath(hetero_path)}")
        loaded = _torch_load_pt(hetero_path)
        raw = _extract_hetero_data(loaded, hetero_path)
        print(
            f"[{_LOADER_TAG}] ok | node_types={list(raw.node_types)} | "
            f"edge_types={len(raw.edge_types)}"
        )

        encoder = encoder.lower().strip()
        if encoder == "han":
            from torch_geometric.transforms import AddMetaPaths

            mp_path = han_metapath_json or os.path.join(
                os.path.dirname(hetero_path), "han_metapaths.json"
            )
            if not os.path.isfile(mp_path):
                raise FileNotFoundError(
                    "HAN 需要 metapath 定义。请在数据目录放置 han_metapaths.json，"
                    "或通过参数 hetero_han_metapath_json 指定。"
                )
            with open(mp_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            metapaths: List[List[Tuple[str, str, str]]] = [
                [tuple(t) for t in mp_chain] for mp_chain in meta["metapaths"]
            ]
            self.data_cpu = AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=False)(raw)
        else:
            self.data_cpu = raw

        self.metadata = self.data_cpu.metadata()
        self.node_types = list(self.data_cpu.node_types)
        self.edge_types = list(self.data_cpu.edge_types)
        self.drug_key, self.cell_key = resolve_entity_keys(self.node_types)

        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.encoder = encoder
        self.num_layers = num_layers
        self.dropout_p = dropout

        self.x_projs = nn.ModuleDict()
        for nt in self.node_types:
            if getattr(self.data_cpu[nt], "x", None) is None:
                raise ValueError(f"Node type {nt} has no x; provide features in HeteroData.")
            d = int(self.data_cpu[nt].x.size(-1))
            self.x_projs[str(nt)] = nn.Linear(d, hidden_channels)

        self.out_lin = nn.ModuleDict(
            {str(nt): nn.Linear(hidden_channels, out_channels) for nt in self.node_types}
        )

        if self.encoder == "hgt":
            print(
                f"[{_LOADER_TAG}] encoder=hgt -> SimpleHeteroScatterConv "
                f"(layers={num_layers}); hetero_heads ignored"
            )
            self.convs = nn.ModuleList()
            for _ in range(num_layers):
                self.convs.append(SimpleHeteroScatterConv(self.edge_types, hidden_channels))
        elif self.encoder == "han":
            from torch_geometric.nn import HANConv

            self.convs = nn.ModuleList()
            for _ in range(num_layers):
                self.convs.append(
                    HANConv(
                        hidden_channels,
                        hidden_channels,
                        heads=heads,
                        metadata=self.metadata,
                    )
                )
        else:
            raise ValueError(f"Unknown hetero encoder: {encoder}. Use 'hgt' or 'han'.")

        self.conv_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_norms.append(
                nn.ModuleDict({str(nt): nn.LayerNorm(hidden_channels) for nt in self.node_types})
            )

        self._init_hetero_weights()

    def _init_hetero_weights(self):
        """输入维各异时用 Xavier；输出投影略保守，减轻与 GCN/下游融合初期的尺度冲突。"""
        for m in self.x_projs.values():
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        for m in self.out_lin.values():
            nn.init.xavier_uniform_(m.weight, gain=0.75)
            nn.init.zeros_(m.bias)

    def reset_parameters(self):
        """必须由 CAESynergy 调用 hetero_encoder.reset_parameters()；勿仅遍历子模块以免跳过此入口。"""
        self._init_hetero_weights()
        for conv in self.convs:
            if isinstance(conv, SimpleHeteroScatterConv):
                conv.reset_parameters()
            else:
                r = getattr(conv, "reset_parameters", None)
                if callable(r):
                    r()
        for norm_layer in self.conv_norms:
            for ln in norm_layer.values():
                ln.reset_parameters()

    def _to_x_dict(self, device: torch.device) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for nt in self.node_types:
            x = self.data_cpu[nt].x.to(device=device, dtype=torch.float32)
            out[str(nt)] = self.x_projs[str(nt)](x)
        return out

    def _to_edge_index_dict(self, device: torch.device) -> Dict:
        ed: Dict = {}
        for et in self.edge_types:
            ei = self.data_cpu[et].edge_index
            ed[et] = ei.to(device=device, dtype=torch.long)
        return ed

    # def forward(self, device: torch.device) -> Dict[str, torch.Tensor]:
    #     x_dict = self._to_x_dict(device)
    #     edge_index_dict = self._to_edge_index_dict(device)

    #     for i, conv in enumerate(self.convs):
    #         x_dict = conv(x_dict, edge_index_dict)
    #         norms = self.conv_norms[i]
    #         x_dict = {k: norms[k](v) for k, v in x_dict.items()}
    #         x_dict = {k: F.gelu(v) for k, v in x_dict.items()}
    #         x_dict = {
    #             k: F.dropout(v, p=self.dropout_p, training=self.training)
    #             for k, v in x_dict.items()
    #         }

    #     emb: Dict[str, torch.Tensor] = {}
    #     for nt in self.node_types:
    #         k = str(nt)
    #         if k not in x_dict:
    #             continue
    #         emb[k] = self.out_lin[k](x_dict[k])
    #     return emb

    # def drug_cell_embeddings(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    #     emb = self.forward(device)
    #     return emb[str(self.drug_key)], emb[str(self.cell_key)]
    def forward(self, device: torch.device, edge_mask_dict: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        x_dict = self._to_x_dict(device)
        edge_index_dict = self._to_edge_index_dict(device)

        for i, conv in enumerate(self.convs):
            # ---- 核心修改：如果传入了 edge_mask_dict，将其传递给你的自定义卷积层 ----
            if edge_mask_dict is not None and isinstance(conv, SimpleHeteroScatterConv):
                # 提示：你需要确保你的 SimpleHeteroScatterConv 能够接收或乘以 edge_mask
                # 如果你的 conv 没写这个参数，可以在这里手动把 mask 应用到特征或消息上
                x_dict = conv(x_dict, edge_index_dict, edge_mask_dict=edge_mask_dict)
            else:
                x_dict = conv(x_dict, edge_index_dict)
                
            norms = self.conv_norms[i]
            x_dict = {k: norms[k](v) for k, v in x_dict.items()}
            x_dict = {k: F.gelu(v) for k, v in x_dict.items()}
            x_dict = {
                k: F.dropout(v, p=self.dropout_p, training=self.training)
                for k, v in x_dict.items()
            }

        emb: Dict[str, torch.Tensor] = {}
        for nt in self.node_types:
            k = str(nt)
            if k not in x_dict:
                continue
            emb[k] = self.out_lin[k](x_dict[k])
        return emb

    # 同时适配这个包装函数
    def drug_cell_embeddings(self, device: torch.device, edge_mask_dict: Optional[Dict] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.forward(device, edge_mask_dict=edge_mask_dict)
        return emb[str(self.drug_key)], emb[str(self.cell_key)]


def _make_hetero_encoder(
    hetero_path: str,
    args,
    out_dim: int,
    han_json: Optional[str],
) -> HeteroGraphEncoder:
    return HeteroGraphEncoder(
        hetero_path=hetero_path,
        hidden_channels=args.hetero_hidden_dim,
        out_channels=out_dim,
        encoder=args.hetero_encoder,
        num_layers=args.hetero_num_layers,
        heads=args.hetero_heads,
        dropout=args.hetero_dropout,
        han_metapath_json=han_json,
    )


def build_hetero_encoder_if_enabled(
    args, out_dim: int
) -> Optional[Union[HeteroGraphEncoder, DualFusedHeteroGraphEncoder]]:
    if not getattr(args, "use_hetero_graph_encoder", False):
        return None
    path = (getattr(args, "hetero_graph_path", "") or "").strip()
    if not path:
        path = os.path.join(
            os.path.normpath(args.data_dir),
            args.dataset_name,
            "hetero_graph.pt",
        )
    han_json = (getattr(args, "hetero_han_metapath_json", "") or "").strip() or None
    main_enc = _make_hetero_encoder(path, args, out_dim, han_json)

    if not getattr(args, "use_pathway_drug_hetero", False):
        return main_enc

    pw = (getattr(args, "pathway_drug_hetero_path", "") or "").strip()
    if not pw:
        pw = os.path.join(
            os.path.normpath(args.data_dir),
            args.dataset_name,
            "pathway_drug_hetero.pt",
        )
    if not os.path.isfile(pw):
        print(f"[{_LOADER_TAG}] pathway_drug_hetero not found, use main hetero only: {os.path.abspath(pw)}")
        return main_enc

    # HAN 时优先使用通路图目录下的 han_metapaths.json（与主图结构不同时应分文件配置）
    han_pw: Optional[str] = han_json
    if (getattr(args, "hetero_encoder", "hgt") or "").lower().strip() == "han":
        local_mp = os.path.join(os.path.dirname(pw), "han_metapaths.json")
        if os.path.isfile(local_mp):
            han_pw = local_mp

    pathway_enc = _make_hetero_encoder(pw, args, out_dim, han_pw)
    print(f"[{_LOADER_TAG}] dual hetero: main={os.path.abspath(path)} | pathway={os.path.abspath(pw)}")
    return DualFusedHeteroGraphEncoder(main_enc, pathway_enc)
