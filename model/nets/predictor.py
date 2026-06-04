from typing import Any, Dict

import torch
from torch import nn


def unified_decoder_net_params(args, net_params: Dict[str, Any]) -> Dict[str, int]:
    """单路 decoder：拼接因果 / 偏置 / debias 分支表征，输入维 = drug_dim + 3 * cline_dim。"""
    return {
        "input_size": int(args.drug_dim) + int(args.cline_dim),
        "hidden_dim": int(net_params["decoder_no_cross"]["hidden_dim"]),
    }


class Decoder_no_cross(nn.Module):
    """单个 MLP 解码器；仍返回 (score_c, score_d, score_b) 三个相同预测量以保持上游接口不变。"""

    def __init__(self, args, decoder):
        super(Decoder_no_cross, self).__init__()
        self.args = args
        self.decoder = decoder
        self.reset_parameters()

    def reset_parameters(self):
        self.decoder.reset_parameters()

    def forward(self, cline_c, cline_b, cline_d, dc_fea):
        merge = torch.cat([dc_fea, cline_c], dim=-1)
        score = self.decoder(merge)
        return score, score, score

class predictor_ablation(nn.Module):

    def __init__(self, args, decoder):
        super(predictor_ablation, self).__init__()
        self.args = args
        # decoder 模块
        self.decoder = decoder  # causal decoder
        self.reset_parameters()  # 初始化模型参数

    # 重置模型参数的函数
    def reset_parameters(self):
        self.decoder.reset_parameters()

    # 前向传播函数 cline_emb [bs, gene_dim]
    def forward(self, cline, dc_fea):
        # 对三个部分解码
        merge_embed = torch.cat([dc_fea, cline], dim=-1)
        score = self.decoder(merge_embed)

        return score