import json
import opts
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import warnings
import os

from tqdm import tqdm

from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR       # 学习率调度器
from model.layers.decoder_layer import Decoder_mlp
from model.nets.cross_attention_net import cross_attention
from model.nets.cline_extract import cline_fea_extract
from model.nets.drug_extract import build_drug_fea_extract
from model.nets.select_model import cal_model, decoder_model
from model.nets.predictor import Decoder_no_cross, unified_decoder_net_params
from model.nets.GCA import ca  # gated cross attention
from model.nets.HFO_Syn import HFO_Syn
from model.nets.hetero_stack_enc import build_hetero_encoder_if_enabled

from time import time as time
from torch.utils.data import DataLoader, Subset
from data.data import LoadData
from train.train import train_epoch, eval_epoch
from utils.utils import result_save, result_save_i, result_save_independent, test_result_save
from sklearn.model_selection import KFold, train_test_split

warnings.filterwarnings("ignore")

def train0(args, dataset, out_file, net_params):
    t0 = time()
    # 预处理
    print("[II] preprocess ...")
    # 设置 gpu
    device = torch.device(args.device)

    # define model：药物分子图（可选 gcn + graph_net 双路融合）
    model_drug_extract = build_drug_fea_extract(args, net_params, dataset)
    model_cline_extract = cline_fea_extract(args, net_params['gcn_cline'])
    hetero_enc = build_hetero_encoder_if_enabled(args, net_params["gcn_drug"]["output"])

    # 使用cross attention融合数据
    ca_d2c_i = ca(args, net_params['ca'])
    model_cline_e_decoder = cross_attention(args, ca_d2c_i)

    # 使用cal解耦细胞系ppi网络（可选）
    model_cal = cal_model(args, net_params) if getattr(args, "use_cal", False) else None

    # 解码层
    decoder = Decoder_mlp(args, unified_decoder_net_params(args, net_params))
    model_Decoder_nocross = Decoder_no_cross(args, decoder)

    model_HFO_Syn = HFO_Syn(
        args,
        model_drug_extract,
        model_cline_extract,
        model_cline_e_decoder,
        model_cal,
        model_Decoder_nocross,
        hetero_encoder=hetero_enc,
        cline_gcn_hidden=net_params["gcn_cline"]["hidden"],
    ).to(device)

    # Load dataset..
    trainset, testset, dataset_all = dataset.train_set, dataset.test_set, dataset.dataset_all

    # 使用独立验证方式训练
    if(args.Independent_Testing == True):
        n_train_pool = len(trainset)
        if n_train_pool < 3:
            raise ValueError("独立测试模式下训练池样本过少，无法划分验证集（至少需要 3 条）。")
        idx_all = np.arange(n_train_pool)
        fit_idx, val_idx = train_test_split(
            idx_all, test_size=0.1, random_state=args.seed, shuffle=True
        )
        if len(fit_idx) == 0 or len(val_idx) == 0:
            fit_idx, val_idx = idx_all[:-1], idx_all[-1:]
        train_fit_loader = DataLoader(
            Subset(trainset, fit_idx.tolist()),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=dataset.collate,
        )
        val_loader = DataLoader(
            Subset(trainset, val_idx.tolist()),
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=dataset.collate,
        )
        test_loader = DataLoader(testset,batch_size=args.batch_size,shuffle=False,drop_last=True, collate_fn=dataset.collate)

        # 创建保存结果的csv文件，独立验证集
        if args.mode == 0:
            df = pd.DataFrame(columns=['epoch', 'loss', 'loss_c', 'AUC', 'AUPR', 'F1', 'ACC'])
            df_val = pd.DataFrame(columns=['epoch', 'loss', 'loss_c', 'AUC', 'AUPR', 'F1', 'ACC'])
            df_hold = pd.DataFrame(columns=['epoch', 'loss', 'loss_c', 'AUC', 'AUPR', 'F1', 'ACC'])
            df.to_csv(out_file + '/i_train.csv', index=False)
            df_val.to_csv(out_file + '/i_val.csv', index=False)
            df_hold.to_csv(out_file + '/i_test.csv', index=False)
            best_auc, best_aupr, best_f1, best_acc = 0, 0, 0, 0
        else:
            df = pd.DataFrame(columns=['epoch', 'loss', 'loss_c', 'rmse', 'r2', 'r'])
            df_val = pd.DataFrame(columns=['epoch', 'loss', 'loss_c', 'rmse', 'r2', 'r'])
            df_hold = pd.DataFrame(columns=['epoch', 'loss', 'loss_c', 'rmse', 'r2', 'r'])
            df.to_csv(out_file + '/i_train_r.csv', index=False)
            df_val.to_csv(out_file + '/i_val_r.csv', index=False)
            df_hold.to_csv(out_file + '/i_test_r.csv', index=False)
            best_rmse, best_r2, best_r = 1000, -1000, -1000

        # 初始化模型参数
        model_HFO_Syn.reset_parameters()

        # 优化器放在循环外
        optimizer = optim.Adam(model_HFO_Syn.parameters(), lr=args.learning_rate, weight_decay=args.L2)

        print("preprocess done! cost time:[{:.2f} s]".format(time()-t0))
        print(f"独立划分: 训练池 {n_train_pool:d} -> 拟合 {len(fit_idx):d} / 验证 {len(val_idx):d}；held-out 测试 {len(testset):d}")

        best_ckpt_cls = './model/save_model/best_model_' + args.out_dir_different_params + '.pth'
        best_ckpt_reg = './model/save_model/reg_best_model_' + args.out_dir_different_params + '.pth'

        print("start train ...")
        for epoch in tqdm(range(args.epochs)):
            t0 = time()
            if args.mode == 0:  # 分类
                train_loss, loss_c, train_auc, train_aupr, train_f1, train_acc = train_epoch(
                    model_HFO_Syn, train_fit_loader, optimizer, args, device
                )
                val_loss, val_auc, val_aupr, val_f1, val_acc, att_list, pre_ls = eval_epoch(
                    model_HFO_Syn, val_loader, args, device
                )
                train_result = [epoch, train_loss, loss_c, train_auc, train_aupr, train_f1, train_acc]
                val_result = [epoch, val_loss, val_loss, val_auc, val_aupr, val_f1, val_acc]
                if val_acc + val_auc > best_acc + best_auc:
                    best_auc, best_aupr, best_f1, best_acc = val_auc, val_aupr, val_f1, val_acc
                    torch.save(model_HFO_Syn.state_dict(), best_ckpt_cls)
            else:               # 回归
                train_loss, loss_c, train_rmse, train_r2, train_r = train_epoch(
                    model_HFO_Syn, train_fit_loader, optimizer, args, device
                )
                val_loss, val_rmse, val_r2, val_r, att_list, pre_ls = eval_epoch(
                    model_HFO_Syn, val_loader, args, device
                )
                train_result = [epoch, train_loss, loss_c, train_rmse, train_r2, train_r]
                val_result = [epoch, val_loss, val_loss, val_rmse, val_r2, val_r]
                if val_rmse < best_rmse:
                    best_rmse, best_r2, best_r = val_rmse, val_r2, val_r
                    torch.save(model_HFO_Syn.state_dict(), best_ckpt_reg)

            result_save_independent(args, out_file, train_result, val_result)

        # 最终在测试集上评估
        if args.mode == 0:
            if os.path.isfile(best_ckpt_cls):
                model_HFO_Syn.load_state_dict(torch.load(best_ckpt_cls, map_location=device))
            model_HFO_Syn.eval()
            test_loss, test_auc, test_aupr, test_f1, test_acc, _, _ = eval_epoch(
                model_HFO_Syn, test_loader, args, device
            )
            pd.DataFrame([[-1, test_loss, test_loss, test_auc, test_aupr, test_f1, test_acc]]).to_csv(
                out_file + '/i_test.csv', mode='a', header=False, index=False
            )
        else:
            if os.path.isfile(best_ckpt_reg):
                model_HFO_Syn.load_state_dict(torch.load(best_ckpt_reg, map_location=device))
            model_HFO_Syn.eval()
            test_loss, test_rmse, test_r2, test_r, _, _ = eval_epoch(
                model_HFO_Syn, test_loader, args, device
            )
            pd.DataFrame([[-1, test_loss, test_loss, test_rmse, test_r2, test_r]]).to_csv(
                out_file + '/i_test_r.csv', mode='a', header=False, index=False
            )
            
    else:
        print("5-fold cross validation start ...")
        full_dataset = dataset.dataset_all

        # 平均性能初始化
        if args.mode == 0:
            all_auc, all_aupr, all_f1, all_acc = 0, 0, 0, 0
        else:
            all_rmse, all_r2, all_r = 0, 0, 0

        kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
        indices = np.arange(len(full_dataset))
        folds = list(kf.split(indices))

        for fold, (train_idx, val_idx) in enumerate(folds):
            print(f"\n========== Fold {fold} ==========")

            trainset_kf = Subset(full_dataset, train_idx)
            valset_kf = Subset(full_dataset, val_idx)

            train_loader = DataLoader(trainset_kf, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=dataset.collate)
            val_loader = DataLoader(valset_kf, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn=dataset.collate)

            # 初始化每折结果文件，创建表头防止覆盖
            if args.mode == 0:
                pd.DataFrame(columns=['epoch', 'loss', 'loss_c', 'AUC', 'AUPR', 'F1', 'ACC']).to_csv(out_file + f'/train_{fold}.csv', index=False)
                pd.DataFrame(columns=['epoch', 'loss', 'AUC', 'AUPR', 'F1', 'ACC']).to_csv(out_file + f'/val_{fold}.csv', index=False)
                best_auc, best_aupr, best_f1, best_acc = 0, 0, 0, 0
                best_model_path = f'./model/save_model/best_model_fold{fold}_{args.out_dir_different_params}.pth'
            else:
                pd.DataFrame(columns=['epoch', 'loss', 'loss_c', 'rmse', 'r2', 'r']).to_csv(out_file + f'/train_r_{fold}.csv', index=False)
                pd.DataFrame(columns=['epoch', 'loss', 'rmse', 'r2', 'r']).to_csv(out_file + f'/val_r_{fold}.csv', index=False)
                best_rmse, best_r2, best_r = 1e9, -1e9, -1e9
                best_model_path = f'./model/save_model/reg_best_model_fold{fold}_{args.out_dir_different_params}.pth'

            # 每 fold 重新初始化模型
            model_HFO_Syn.reset_parameters()
            optimizer = optim.Adam(model_HFO_Syn.parameters(), lr=args.learning_rate, weight_decay=args.L2)

            for epoch in tqdm(range(args.epochs)):
                t_epoch = time()

                if args.mode == 0:  # 分类
                    train_loss, loss_c, train_auc, train_aupr, train_f1, train_acc = train_epoch(
                        model_HFO_Syn, train_loader, optimizer, args, device
                    )
                    val_loss, val_auc, val_aupr, val_f1, val_acc, _, _ = eval_epoch(
                        model_HFO_Syn, val_loader, args, device
                    )

                    # 实时写入保存
                    pd.DataFrame([[epoch, train_loss, loss_c, train_auc, train_aupr, train_f1, train_acc]]).to_csv(out_file + f'/train_{fold}.csv', mode='a', header=False, index=False)
                    pd.DataFrame([[epoch, val_loss, val_auc, val_aupr, val_f1, val_acc]]).to_csv(out_file + f'/val_{fold}.csv', mode='a', header=False, index=False)

                    if val_auc + val_acc > best_auc + best_acc:
                        best_auc, best_aupr, best_f1, best_acc = val_auc, val_aupr, val_f1, val_acc
                        torch.save(model_HFO_Syn.state_dict(), best_model_path)
                else:               # 回归
                    train_loss, loss_c, train_rmse, train_r2, train_r = train_epoch(
                        model_HFO_Syn, train_loader, optimizer, args, device
                    )
                    val_loss, val_rmse, val_r2, val_r, _, _ = eval_epoch(
                        model_HFO_Syn, val_loader, args, device
                    )

                    # 实时写入保存
                    pd.DataFrame([[epoch, train_loss, loss_c, train_rmse, train_r2, train_r]]).to_csv(out_file + f'/train_r_{fold}.csv', mode='a', header=False, index=False)
                    pd.DataFrame([[epoch, val_loss, val_rmse, val_r2, val_r]]).to_csv(out_file + f'/val_r_{fold}.csv', mode='a', header=False, index=False)

                    if val_rmse < best_rmse:
                        best_rmse, best_r2, best_r = val_rmse, val_r2, val_r
                        torch.save(model_HFO_Syn.state_dict(), best_model_path)

            # 累加每一折的最优表现
            if args.mode == 0:
                all_auc += best_auc
                all_aupr += best_aupr
                all_f1 += best_f1
                all_acc += best_acc
                # 追加最佳标记到该折的验证集日志尾部
                pd.DataFrame([["best_result", 0, best_auc, best_aupr, best_f1, best_acc]]).to_csv(out_file + f'/val_{fold}.csv', mode='a', header=False, index=False)
            else:
                all_rmse += best_rmse
                all_r2 += best_r2
                all_r += best_r
                pd.DataFrame([["best_result", 0, best_rmse, best_r2, best_r]]).to_csv(out_file + f'/val_r_{fold}.csv', mode='a', header=False, index=False)

        # 五折平均结果统计并持久化
        print("\n========== Final 5-Fold CV Result ==========")
        if args.mode == 0:
            avg_res = ["avg_result", all_auc/5, all_aupr/5, all_f1/5, all_acc/5]
            pd.DataFrame([avg_res], columns=["type", "AUC", "AUPR", "F1", "ACC"]).to_csv(out_file + '/avg.csv', index=False)
            print(f"AVG AUC  : {all_auc/5:.4f} | AVG AUPR : {all_aupr/5:.4f} | AVG F1   : {all_f1/5:.4f} | AVG ACC  : {all_acc/5:.4f}")
        else:
            avg_res = ["avg_result", all_rmse/5, all_r2/5, all_r/5]
            pd.DataFrame([avg_res], columns=["type", "RMSE", "R2", "R"]).to_csv(out_file + '/avg_r.csv', index=False)
            print(f"AVG RMSE : {all_rmse/5:.4f} | AVG R2   : {all_r2/5:.4f} | AVG R    : {all_r/5:.4f}")
# 正常实验分批次训练
def train(args, dataset, out_file, net_params):
    t0 = time()
    # 预处理
    print("[II] preprocess ...")
    # 设置 gpu
    device = torch.device(args.device)

    # define model：药物分子图（可选 gcn + graph_net 双路融合）
    model_drug_extract = build_drug_fea_extract(args, net_params, dataset)
    model_cline_extract = cline_fea_extract(args, net_params['gcn_cline'])
    hetero_enc = build_hetero_encoder_if_enabled(args, net_params["gcn_drug"]["output"])

    # 使用cross attention融合数据
    ca_d2c_i = ca(args, net_params['ca'])
    model_cline_e_decoder = cross_attention(args, ca_d2c_i)

    # 使用cal解耦细胞系ppi网络（可选）
    model_cal = cal_model(args, net_params) if getattr(args, "use_cal", False) else None

    # 解码层
    decoder = Decoder_mlp(args, unified_decoder_net_params(args, net_params))
    model_Decoder_nocross = Decoder_no_cross(args, decoder)

    model_HFO_Syn = HFO_Syn(
        args,
        model_drug_extract,
        model_cline_extract,
        model_cline_e_decoder,
        model_cal,
        model_Decoder_nocross,
        hetero_encoder=hetero_enc,
        cline_gcn_hidden=net_params["gcn_cline"]["hidden"],
    ).to(device)
    trainset, testset, dataset_all = dataset.train_set, dataset.test_set, dataset.dataset_all

    # 使用独立验证方式训练
    if(args.Independent_Testing == True):#TODO 独立测试集模式和五折交叉模式
        n_train_pool = len(trainset)
        if n_train_pool < 3:
            raise ValueError("独立测试模式下训练池样本过少，无法划分验证集（至少需要 3 条）。")
        idx_all = np.arange(n_train_pool)
        fit_idx, val_idx = train_test_split(
            idx_all, test_size=0.1, random_state=args.seed, shuffle=True
        )
        if len(fit_idx) == 0 or len(val_idx) == 0:
            fit_idx, val_idx = idx_all[:-1], idx_all[-1:]
        train_fit_loader = DataLoader(
            Subset(trainset, fit_idx.tolist()),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=dataset.collate,
        )
        val_loader = DataLoader(
            Subset(trainset, val_idx.tolist()),
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=dataset.collate,
        )
        test_loader = DataLoader(testset,batch_size=args.batch_size,shuffle=False,drop_last=True, collate_fn=dataset.collate)

        # 创建保存结果的csv文件，独立验证集
        # 分类
        if args.mode == 0:
            df = pd.DataFrame(columns=['epoch', 'loss', 'loss_c', 'AUC', 'AUPR', 'F1', 'ACC'])
            df_val = pd.DataFrame(columns=['epoch', 'loss', 'loss_c', 'AUC', 'AUPR', 'F1', 'ACC'])
            df_hold = pd.DataFrame(columns=['epoch', 'loss', 'loss_c', 'AUC', 'AUPR', 'F1', 'ACC'])
            df.to_csv(out_file + '/i_train', index=False)
            df_val.to_csv(out_file + '/i_val', index=False)
            df_hold.to_csv(out_file + '/i_test', index=False)
            best_auc, best_aupr, best_f1, best_acc = 0, 0, 0, 0
        # 回归
        else:
            df = pd.DataFrame(columns=['epoch', 'loss', 'loss_c', 'rmse', 'r2', 'r'])
            df_val = pd.DataFrame(columns=['epoch', 'loss', 'loss_c', 'rmse', 'r2', 'r'])
            df_hold = pd.DataFrame(columns=['epoch', 'loss', 'loss_c', 'rmse', 'r2', 'r'])
            df.to_csv(out_file + '/i_train_r', index=False)
            df_val.to_csv(out_file + '/i_val_r', index=False)
            df_hold.to_csv(out_file + '/i_test_r', index=False)
            best_rmse, best_r2, best_r = 1000, 1000, 1000

        # 初始化模型参数
        model_HFO_Syn.reset_parameters()

        # 打印预处理使用时间
        print("preprocess done! cost time:[{:.2f} s]".format(time()-t0))
        print(
            "独立划分: 训练池 {:d} -> 拟合 {:d} / 验证 {:d}；held-out 测试 {:d}".format(
                n_train_pool, len(fit_idx), len(val_idx), len(testset)
            )
        )

        best_ckpt_cls = './model/save_model/best_model_' + args.out_dir_different_params + '.pth'
        best_ckpt_reg = './model/save_model/reg_best_model_' + args.out_dir_different_params + '.pth'

        print("start train ...")
        for epoch in tqdm(range(args.epochs)):
            t0 = time()
            optimizer = optim.Adam(model_HFO_Syn.parameters(), lr = args.learning_rate, weight_decay= args.L2)
            train_result = []
            att_list = []
            if args.mode == 0:  # 分类
                train_loss, loss_c, train_auc, train_aupr, train_f1, train_acc = train_epoch(
                    model_HFO_Syn, train_fit_loader, optimizer, args, device
                )
                val_loss, val_auc, val_aupr, val_f1, val_acc, att_list, pre_ls = eval_epoch(
                    model_HFO_Syn, val_loader, args, device
                )
                train_result = [epoch, train_loss, loss_c, train_auc, train_aupr, train_f1, train_acc]
                val_result = [epoch, val_loss, val_loss, val_auc, val_aupr, val_f1, val_acc]
                if val_acc + val_auc > best_acc + best_auc:
                    best_auc, best_aupr, best_f1, best_acc = val_auc, val_aupr, val_f1, val_acc
                    torch.save(model_HFO_Syn.state_dict(), best_ckpt_cls)
            else:               # 回归
                train_loss, loss_c, train_rmse, train_r2, train_r = train_epoch(
                    model_HFO_Syn, train_fit_loader, optimizer, args, device
                )
                val_loss, val_rmse, val_r2, val_r, att_list, pre_ls = eval_epoch(
                    model_HFO_Syn, val_loader, args, device
                )
                train_result = [epoch, train_loss, loss_c, train_rmse, train_r2, train_r]
                val_result = [epoch, val_loss, val_loss, val_rmse, val_r2, val_r]
                if val_rmse < best_rmse:
                    best_rmse, best_r2, best_r = val_rmse, val_r2, val_r
                    torch.save(model_HFO_Syn.state_dict(), best_ckpt_reg)

            result_save_independent(args, out_file, train_result, val_result)
            print("training done! cost time:[{:.2f}s]".format(time()-t0))

        if args.mode == 0:
            if os.path.isfile(best_ckpt_cls):
                model_HFO_Syn.load_state_dict(torch.load(best_ckpt_cls, map_location=device))
            model_HFO_Syn.eval()
            test_loss, test_auc, test_aupr, test_f1, test_acc, _, _ = eval_epoch(
                model_HFO_Syn, test_loader, args, device
            )
            pd.DataFrame([[-1, test_loss, test_loss, test_auc, test_aupr, test_f1, test_acc]]).to_csv(
                out_file + '/i_test', mode='a', header=False, index=False
            )
        else:
            if os.path.isfile(best_ckpt_reg):
                model_HFO_Syn.load_state_dict(torch.load(best_ckpt_reg, map_location=device))
            model_HFO_Syn.eval()
            test_loss, test_rmse, test_r2, test_r, _, _ = eval_epoch(
                model_HFO_Syn, test_loader, args, device
            )
            pd.DataFrame([[-1, test_loss, test_loss, test_rmse, test_r2, test_r]]).to_csv(
                out_file + '/i_test_r', mode='a', header=False, index=False
            )
    else:
        print("cross_val start train ...")
       
        if args.mode == 0:
            all_auc, all_aupr, all_f1, all_acc = 0, 0, 0, 0
        else :
            all_rmse, all_r2, all_r = 0, 0, 0

      
        kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
       
        indices = np.arange(len(dataset_all))

        folds = list(kf.split(indices))

        for fold, (trainset_index, valset_index) in enumerate(folds):
            trainset_kf = Subset(dataset_all, trainset_index)
            valset_kf = Subset(dataset_all, valset_index)
            train_loader = DataLoader(trainset_kf,batch_size=args.batch_size,shuffle=True,drop_last=True, collate_fn=dataset.collate)
            val_loader = DataLoader(valset_kf,batch_size=args.batch_size,shuffle=False,drop_last=True, collate_fn=dataset.collate)

            if args.mode == 0:
                df = pd.DataFrame(columns=['epoch', 'loss', 'loss_c', 'AUC', 'AUPR', 'F1', 'ACC'])
                df_val = pd.DataFrame(columns=['fold', 'loss', 'AUC', 'AUPR', 'F1', 'ACC'])
                df.to_csv(out_file + '/train_' +repr(fold), index=False)
                df_val.to_csv(out_file + '/val_' + repr(fold), index=False)
                best_auc, best_aupr, best_f1, best_acc = 0, 0, 0, 0
            else:
                df = pd.DataFrame(columns=['epoch', 'loss', 'loss_c', 'rmse', 'r2', 'r'])
                df_val = pd.DataFrame(columns=['fold', 'loss', 'rmse', 'r2', 'r'])
                df.to_csv(out_file + '/train_r_' +repr(fold), index=False)
                df_val.to_csv(out_file + '/val_r_' + repr(fold), index=False)
                best_rmse, best_r2, best_r = 1000, 1000, 1000

            model_HFO_Syn.reset_parameters()

            print("preprocess done! cost time:[{:.2f} s]".format(time()-t0))

            test_result = []
            # 训练过程
            print("fold_" + repr(fold) + "train ...")
            print("start train ...")
            for epoch in tqdm(range(args.epochs)):
                t0 = time()
                train_result = []
                val_result = []
                optimizer = optim.Adam(model_HFO_Syn.parameters(), lr = args.learning_rate, weight_decay= args.L2)
                if args.mode == 0:  # 分类
                    train_loss, loss_c, train_auc, train_aupr, train_f1, train_acc = train_epoch(model_HFO_Syn, train_loader, optimizer, args, device)
                    val_loss, val_auc, val_aupr, val_f1, val_acc, _, _ = eval_epoch(model_HFO_Syn, val_loader, args, device)
                    
                    train_result = [epoch, train_loss, loss_c, train_auc, train_aupr, train_f1, train_acc]
                    val_result = [epoch, val_loss, val_auc, val_aupr, val_f1, val_acc]
                
                    if val_acc + val_auc > best_acc + best_auc:
                        best_auc =  val_auc
                        best_aupr = val_aupr
                        best_f1 = val_f1
                        best_acc = val_acc
                        print("best_auc:" + str(best_auc) + " | " +"best_aupr:" + str(best_aupr) +" | ""best_f1:" + str(best_f1) +" | ""best_acc:" + str(best_acc) +" | ")
                        torch.save(model_HFO_Syn.state_dict(), './model/save_model/best_model_' + repr(fold) + '_' + args.out_dir_different_params + '.pth')
                else:               
                    train_loss, loss_c, train_rmse, train_r2, train_r = train_epoch(model_HFO_Syn, train_loader, optimizer, args, device)
                    val_loss, val_rmse, val_r2, val_r, _, _ = eval_epoch(model_HFO_Syn, val_loader, args, device)
                    train_result = [epoch, train_loss, loss_c, train_rmse, train_r2, train_r]
                    val_result = [epoch, val_loss, val_rmse, val_r2, val_r]
                    if val_rmse < best_rmse:
                        best_rmse =  val_rmse
                        best_r2 = val_r2
                        best_r = val_r
                        print("best_rmse:" + str(best_rmse) + " | " +"best_r2:" + str(best_r2) +" | ""best_r:" + str(best_r) )
                        torch.save(model_HFO_Syn.state_dict(), './model/save_model/reg_best_model_' + repr(fold) + '_' + args.out_dir_different_params + '.pth')

                
                result_save(args, epoch, out_file, fold, train_result, val_result)
                print("training done! cost time:[{:.2f}s]".format(time()-t0))
               

            if args.mode == 0:
                best_result = ["best_result", best_auc, best_aupr, best_f1, best_acc]
                all_auc += best_auc
                all_aupr += best_aupr
                all_f1 += best_f1
                all_acc += best_acc
                best_result_data = pd.DataFrame([best_result])
                best_result_data.to_csv(out_file + '/val_' + repr(fold), mode='a', header=False, index=False)
                if fold == 4:
                    all_auc, all_aupr, all_f1, all_acc = all_auc/5, all_aupr/5, all_f1/5, all_acc/5
                    all_result = ["avg_result", all_auc, all_aupr, all_f1, all_acc]
                    all_result_data = pd.DataFrame([all_result])
                    all_result_data.to_csv(out_file + '/avg', mode='a', header=False, index=False)
            else :
                best_result = ["best_result", best_rmse, best_r2, best_r]
                all_rmse += best_rmse
                all_r2 += best_r2
                all_r += best_r
                best_result_data = pd.DataFrame([best_result])
                best_result_data.to_csv(out_file + '/val_r_' + repr(fold), mode='a', header=False, index=False)
                if fold == 4:
                    all_rmse, all_r2, all_r = all_rmse/5, all_r2/5, all_r/5
                    all_result = ["avg_result", all_rmse, all_r2, all_r]
                    all_result_data = pd.DataFrame([all_result])
                    all_result_data.to_csv(out_file + '/avg_r', mode='a', header=False, index=False)

def main():
    args = opts.parse_args()
    opts.setup_seed(args.seed)

    with open(r'E:\HFO代码\HFO-Syn\config\HFO-Syn.json') as f:
        config = json.load(f)

    DATASET_NAME = args.dataset_name
    print("DATASET_NAME:" , DATASET_NAME)
    dataset = LoadData(DATASET_NAME, args)
    net_params = config[DATASET_NAME]  # 根据不同的数据集读取对应的模型参数

    out_dir_different_params = args.out_dir_different_params
        # 结果保存路径
    out_file = args.out_dir + DATASET_NAME + '/' + out_dir_different_params
    # 如果不存在文件夹则创建文件夹
    if not os.path.exists(out_file):
        os.mkdir(out_file)
    # 使用指定的方法训练
    if(args.method_name == 'HFO_Syn'):
        train(args, dataset, out_file, net_params)
    else:
        print("method_name_error!")

if __name__ == '__main__':
     main()
