import argparse
import numpy as np
import random
import torch
from itertools import product

'''
opts.py文件通常是用来处理命令行参数和配置模型超参数的Python模块。
它通常定义了一个命令行参数解析器和一些默认参数值，以便用户可以通过命令行或配置文件修改模型的参数。
这种方式使得模型的使用更加灵活，并且可以更方便地进行超参数调整和实验比较。
'''

def parse_args():
    str2bool = lambda x: x.lower() == "true"
    # 创建一个解析对象  科普parse就是解析的意思
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotion', type=str, default='')
    parser.add_argument('--L2', type=float, default=1e-8)
    parser.add_argument('--dataset_name', type=str, default='OncologyScreen', help="DrugCombDB or OncologyScreen")
    parser.add_argument('--out_dir', type=str, default='./result/')
    parser.add_argument('--config', type=str, default='./config/HFO_Syn.json')
    parser.add_argument('--data_dir', type=str, default='./data/Synergy/')
    parser.add_argument('--out_dir_different_params', type=str, default='OncologyScreen')    # 不同的模型参数对于的结果路径
    parser.add_argument('--method_name', type=str, default='HFO_Syn', help="HFO_Syn or DeepSynergy or GraphSynergy") # 对比试验的方法
    parser.add_argument(
        '--Independent_Testing', type=str2bool, default=False,
        help='False: 固定 9:1 训练/测试；训练部分再划验证集选模型，测试集仅最终评估。True: 5 折交叉验证（全量 data 上折）',
    )

    ####################    参数     #######################
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--device', type=str, default='cuda:0', help="cuda:0 or cpu")
    parser.add_argument('--cal_name', type=str, default='cal', help="cal")
    parser.add_argument('--use_cal', type=str2bool, default=False,
                        help='True: 使用 causal_attention_learning 的 CAL 解耦；False: 跳过 CAL，直接用细胞系表征进预测器')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--cline_dim', type=int, default=2865, help="2865 or 7217")    # 种子值为0 效果比较好
    parser.add_argument('--cline_hidden', type=int, default=2048, help="2048 or 4096")    # 种子值为0 效果比较好
    parser.add_argument('--drug_dim', type=int, default=256, help="256 or ")    # 种子值为0 效果比较好
    parser.add_argument('--mode', type=int, default=0, help="0 或 1 代表分类和回归")
    parser.add_argument('--batch_size', type=int, default=200, help="256")
    parser.add_argument('--alpha', type=float, default=0.4)
    parser.add_argument('--beta', type=float, default=0.6)
    parser.add_argument('--seed', type=int, default=42)    # 种子值为0 效果比较好

    ####################     cal     #######################
    parser.add_argument('--with_random', type=str2bool, default=True)
    parser.add_argument('--without_node_attention', type=str2bool, default=False)
    parser.add_argument('--without_edge_attention', type=str2bool, default=False)
    parser.add_argument('--cat_or_add', type=str, default="add")    # random add 的方式
    parser.add_argument('--global_pool', type=str, default="sum")
    parser.add_argument('--using_specific_ppi', type=bool, default=True, help="True or False") # 是否使用特异性PPI

    ####################     gcn     #######################
    parser.add_argument('--use_GMP', type=bool, default=True)       # gcn中是否使用全局最大池化（global max pooling）
    parser.add_argument('--use_dual_drug_gcn', type=str2bool, default=False,
                        help='True: gcn + graph_net(GraphNet) 双路编码药物图并融合；False: 仅 gcn（不改 gcn_net）')

    parser.add_argument('--use_hetero_graph_encoder', type=str2bool, default=True,
                        help='True: 加载 hetero_graph.pt，用 HGT/HAN 学习药物与细胞系嵌入，并与分子 GCN、基因表达分支融合；False 关闭')
    parser.add_argument('--hetero_graph_path', type=str, default='',
                        help='HeteroData 的 .pt 路径；留空则使用 data_dir/dataset_name/hetero_graph.pt')
    parser.add_argument('--hetero_encoder', type=str, default='hgt', help="hgt 或 han")
    parser.add_argument('--hetero_hidden_dim', type=int, default=256)
    parser.add_argument('--hetero_num_layers', type=int, default=2)
    parser.add_argument('--hetero_heads', type=int, default=4)
    parser.add_argument('--hetero_dropout', type=float, default=0.2)
    parser.add_argument('--hetero_han_metapath_json', type=str, default='',
                        help="HAN 时 metapath 的 JSON 路径；默认同目录 han_metapaths.json")
    parser.add_argument('--use_pathway_drug_hetero', type=str2bool, default=True,
                        help='True: 若存在 pathway_drug_hetero.pt，则用与 hetero_graph 相同的 HeteroGraphEncoder 编码后再与主异构图嵌入融合；文件不存在时自动仅用主图')
    parser.add_argument('--pathway_drug_hetero_path', type=str, default='',
                        help='通路药物异构图 .pt；留空则使用 data_dir/dataset_name/pathway_drug_hetero.pt')

    ####################     decoder_name     #######################
    parser.add_argument('--decoder_name', type=str, default='decoder_mlp', help='decoder_lstm or decoder_mlp')
    ####################     ablation     #######################
    parser.add_argument('--ablation_cross_att',  type=bool, default=False, help='True or False')

    args = parser.parse_args()  # 属性给予实例args
    print_args(args)            # 打印实例args中的各个参数
    setup_seed(args.seed)       # 设置随机种子为args.seed
    return args

# 打印参数
def print_args(args, str_num=80):
    for arg, val in args.__dict__.items():
        print(arg + '.' * (str_num - len(arg) - len(str(val))) + str(val))
    print()

# 设定python标准库、torch库、numpy库的随机种子，保证代码可以重现
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
