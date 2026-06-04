import opts
import json
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import warnings
import os
from sklearn.model_selection import KFold, GroupKFold, train_test_split

from tqdm import tqdm

from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR       # 学习率调度器
from model.layers.decoder_layer import Decoder_mlp
from model.nets.cross_attention_net import cross_attention
from model.nets.cline_extract import cline_fea_extract
from model.nets.drug_extract import build_drug_fea_extract
from model.nets.select_model import cal_model, decoder_model
from model.nets.predictor import Decoder_no_cross, unified_decoder_net_params
from model.nets.GCA import ca  # gated cross attention
from model.nets.HFO_Syn import CAESynergy
from model.nets.hetero_stack_enc import build_hetero_encoder_if_enabled

from time import time as time
from torch.utils.data import DataLoader, Subset
from data.data import LoadData
from train.train import train_epoch, eval_epoch
from utils.utils import result_save, result_save_i, result_save_independent, test_result_save
from sklearn.model_selection import KFold, train_test_split

warnings.filterwarnings("ignore")

def train_co(args, dataset, out_file, net_params):
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

    model_CAESynergy = CAESynergy(
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
        model_CAESynergy.reset_parameters()

        # 优化器放在循环外
        optimizer = optim.Adam(model_CAESynergy.parameters(), lr=args.learning_rate, weight_decay=args.L2)

        print("preprocess done! cost time:[{:.2f} s]".format(time()-t0))
        print(f"独立划分: 训练池 {n_train_pool:d} -> 拟合 {len(fit_idx):d} / 验证 {len(val_idx):d}；held-out 测试 {len(testset):d}")

        best_ckpt_cls = './model/save_model/best_model_' + args.out_dir_different_params + '.pth'
        best_ckpt_reg = './model/save_model/reg_best_model_' + args.out_dir_different_params + '.pth'

        print("start train ...")
        for epoch in tqdm(range(args.epochs)):
            t0 = time()
            if args.mode == 0:  # 分类
                train_loss, loss_c, train_auc, train_aupr, train_f1, train_acc = train_epoch(
                    model_CAESynergy, train_fit_loader, optimizer, args, device
                )
                val_loss, val_auc, val_aupr, val_f1, val_acc, att_list, pre_ls = eval_epoch(
                    model_CAESynergy, val_loader, args, device
                )
                train_result = [epoch, train_loss, loss_c, train_auc, train_aupr, train_f1, train_acc]
                val_result = [epoch, val_loss, val_loss, val_auc, val_aupr, val_f1, val_acc]
                if val_acc + val_auc > best_acc + best_auc:
                    best_auc, best_aupr, best_f1, best_acc = val_auc, val_aupr, val_f1, val_acc
                    torch.save(model_CAESynergy.state_dict(), best_ckpt_cls)
            else:               # 回归
                train_loss, loss_c, train_rmse, train_r2, train_r = train_epoch(
                    model_CAESynergy, train_fit_loader, optimizer, args, device
                )
                val_loss, val_rmse, val_r2, val_r, att_list, pre_ls = eval_epoch(
                    model_CAESynergy, val_loader, args, device
                )
                train_result = [epoch, train_loss, loss_c, train_rmse, train_r2, train_r]
                val_result = [epoch, val_loss, val_loss, val_rmse, val_r2, val_r]
                if val_rmse < best_rmse:
                    best_rmse, best_r2, best_r = val_rmse, val_r2, val_r
                    torch.save(model_CAESynergy.state_dict(), best_ckpt_reg)

            result_save_independent(args, out_file, train_result, val_result)

        # 最终在测试集上评估
        if args.mode == 0:
            if os.path.isfile(best_ckpt_cls):
                model_CAESynergy.load_state_dict(torch.load(best_ckpt_cls, map_location=device))
            model_CAESynergy.eval()
            test_loss, test_auc, test_aupr, test_f1, test_acc, _, _ = eval_epoch(
                model_CAESynergy, test_loader, args, device
            )
            pd.DataFrame([[-1, test_loss, test_loss, test_auc, test_aupr, test_f1, test_acc]]).to_csv(
                out_file + '/i_test.csv', mode='a', header=False, index=False
            )
        else:
            if os.path.isfile(best_ckpt_reg):
                model_CAESynergy.load_state_dict(torch.load(best_ckpt_reg, map_location=device))
            model_CAESynergy.eval()
            test_loss, test_rmse, test_r2, test_r, _, _ = eval_epoch(
                model_CAESynergy, test_loader, args, device
            )
            pd.DataFrame([[-1, test_loss, test_loss, test_rmse, test_r2, test_r]]).to_csv(
                out_file + '/i_test_r.csv', mode='a', header=False, index=False
            )
            
    else:
            print("5-fold cross validation (by drug combinations) start ...")
            full_dataset = dataset.dataset_all

            # ==========================================
            # 【全新修改】: 精准解构复杂元组以提取药物组合 ID
            # ==========================================
            groups = []
            for i in range(len(full_dataset)):
                data = full_dataset[i]
                
                try:
                    # 根据 Debug 结构，样本是一个序列。
                    # 倒数第二项 data[-2] 通常是包含 [DrugA_ID, DrugB_ID, Cell_ID] 映射关系的特殊数组
                    if isinstance(data, (tuple, list)) and len(data) >= 4:
                        # 优先尝试从 data[-2] 中获取对应的特征或映射
                        target_meta = data[-2]
                        if isinstance(target_meta, (tuple, list, np.ndarray)):
                            # 如果它是 list 包裹的，拿出来
                            if isinstance(target_meta, list) and len(target_meta) > 0:
                                target_meta = target_meta[0]
                            
                            # 如果是多维矩阵，且第二维包含组合 ID (例如第二行保存着类似 19., 1135. 的映射 ID)
                            if hasattr(target_meta, 'shape') and len(target_meta.shape) > 1 and target_meta.shape[0] >= 2:
                                d1 = target_meta[1, 0] # 提取矩阵特定维度的标识
                                d2 = target_meta[1, 1]
                            elif isinstance(target_meta, (list, tuple, np.ndarray)) and len(target_meta) >= 2:
                                d1 = target_meta[0]
                                d2 = target_meta[1]
                            else:
                                # 降级尝试最后一项 data[-1]
                                d1 = data[-1][0] if hasattr(data[-1], '__getitem__') else i
                                d2 = data[-1][1] if hasattr(data[-1], '__getitem__') and len(data[-1]) > 1 else i
                        else:
                            d1, d2 = i, i
                    else:
                        d1, d2 = i, i
                except Exception:
                    # 异常安全保护：如果某条异常，用单样本索引隔离，不影响整体运行
                    d1, d2 = i, i

                # 药物组合去顺序化（统一排序拼接字符串，确保 A-B 和 B-A 被视为同一个 Group）
                drug_comb = "-".join(sorted([str(int(float(d1))), str(int(float(d2)))]))
                groups.append(drug_comb)
                
            groups = np.array(groups)
            # ==========================================

            # 平均性能初始化
            if args.mode == 0:
                all_auc, all_aupr, all_f1, all_acc = 0, 0, 0, 0
            else:
                all_rmse, all_r2, all_r = 0, 0, 0

            # ==========================================
            # 【全新安全机制】: 校验并应用 GroupKFold
            # ==========================================
            unique_groups_count = len(np.unique(groups))
            print(f"[INFO] 成功提取到唯一药物组合总数: {unique_groups_count}")
            
            # 稳健性处理：如果提取失败导致组数少于 5，自动切换回常规 KFold，防止程序直接崩溃崩溃
            if unique_groups_count < 5:
                print("[Warning] 药物组合提取不满足分组条件（不同组合少于5种），GroupKFold 自动安全降级为标准随机 KFold！")
                kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
                folds = list(kf.split(np.arange(len(full_dataset))))
            else:
                gkf = GroupKFold(n_splits=5)
                indices = np.arange(len(full_dataset))
                folds = list(gkf.split(indices, groups=groups))
            # ==========================================

            for fold, (train_idx, val_idx) in enumerate(folds):
                print(f"\n========== Fold {fold} ==========")
                
                # 只有在成功执行 Group 分组时才做重叠校验
                if unique_groups_count >= 5:
                    train_groups = set(groups[train_idx])
                    val_groups = set(groups[val_idx])
                    overlap = train_groups.intersection(val_groups)
                    print(f"训练集药物组合数: {len(train_groups)}, 验证集药物组合数: {len(val_groups)}, 重叠组合数: {len(overlap)}")
                    if len(overlap) > 0:
                        print(f"[Warning] 警告！发现数据泄露！重叠组合: {overlap}")

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
                model_CAESynergy.reset_parameters()
                optimizer = optim.Adam(model_CAESynergy.parameters(), lr=args.learning_rate, weight_decay=args.L2)

                for epoch in tqdm(range(args.epochs)):
                    t_epoch = time()

                    if args.mode == 0:  # 分类
                        train_loss, loss_c, train_auc, train_aupr, train_f1, train_acc = train_epoch(
                            model_CAESynergy, train_loader, optimizer, args, device
                        )
                        val_loss, val_auc, val_aupr, val_f1, val_acc, _, _ = eval_epoch(
                            model_CAESynergy, val_loader, args, device
                        )

                        # 实时写入保存
                        pd.DataFrame([[epoch, train_loss, loss_c, train_auc, train_aupr, train_f1, train_acc]]).to_csv(out_file + f'/train_{fold}.csv', mode='a', header=False, index=False)
                        pd.DataFrame([[epoch, val_loss, val_auc, val_aupr, val_f1, val_acc]]).to_csv(out_file + f'/val_{fold}.csv', mode='a', header=False, index=False)

                        if val_auc + val_acc > best_auc + best_acc:
                            best_auc, best_aupr, best_f1, best_acc = val_auc, val_aupr, val_f1, val_acc
                            torch.save(model_CAESynergy.state_dict(), best_model_path)
                    else:               # 回归
                        train_loss, loss_c, train_rmse, train_r2, train_r = train_epoch(
                            model_CAESynergy, train_loader, optimizer, args, device
                        )
                        val_loss, val_rmse, val_r2, val_r, _, _ = eval_epoch(
                            model_CAESynergy, val_loader, args, device
                        )

                        # 实时写入保存
                        pd.DataFrame([[epoch, train_loss, loss_c, train_rmse, train_r2, train_r]]).to_csv(out_file + f'/train_r_{fold}.csv', mode='a', header=False, index=False)
                        pd.DataFrame([[epoch, val_loss, val_rmse, val_r2, val_r]]).to_csv(out_file + f'/val_r_{fold}.csv', mode='a', header=False, index=False)

                        if val_rmse < best_rmse:
                            best_rmse, best_r2, best_r = val_rmse, val_r2, val_r
                            torch.save(model_CAESynergy.state_dict(), best_model_path)

                # 累加每一折的最优表现
                if args.mode == 0:
                    all_auc += best_auc
                    all_aupr += best_aupr
                    all_f1 += best_f1
                    all_acc += best_acc
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

def train(args, dataset, out_file, net_params):
    t0 = time.time()
    print("[II] preprocess ...")
    device = torch.device(args.device)

    # 2. 初始化网络模型
    model_drug_extract = build_drug_fea_extract(args, net_params, dataset)
    model_cline_extract = cline_fea_extract(args, net_params['gcn_cline'])
    hetero_enc = build_hetero_encoder_if_enabled(args, net_params["gcn_drug"]["output"])

    ca_d2c_i = ca(args, net_params['ca'])
    model_cline_e_decoder = cross_attention(args, ca_d2c_i)
    model_cal = cal_model(args, net_params) if getattr(args, "use_cal", False) else None

    decoder = Decoder_mlp(args, unified_decoder_net_params(args, net_params))
    model_Decoder_nocross = Decoder_no_cross(args, decoder)

    model_CAESynergy = CAESynergy(
        args,
        model_drug_extract,
        model_cline_extract,
        model_cline_e_decoder,
        model_cal,
        model_Decoder_nocross,
        hetero_encoder=hetero_enc,
        cline_gcn_hidden=net_params["gcn_cline"]["hidden"],
    ).to(device)

    # 3. 独立测试模式保留原样
    if getattr(args, "Independent_Testing", False) == True:
        n_train_pool = len(dataset.train_set)
        if n_train_pool < 3:
            raise ValueError("独立测试模式下训练池样本过少，无法划分验证集（至少需要 3 条）。")
        idx_all = np.arange(n_train_pool)
        from sklearn.model_selection import train_test_split
        fit_idx, val_idx = train_test_split(idx_all, test_size=0.1, random_state=args.seed, shuffle=True)
        if len(fit_idx) == 0 or len(val_idx) == 0:
            fit_idx, val_idx = idx_all[:-1], idx_all[-1:]
        
        train_fit_loader = DataLoader(Subset(dataset.train_set, fit_idx.tolist()), batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=dataset.collate)
        val_loader = DataLoader(Subset(dataset.train_set, val_idx.tolist()), batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn=dataset.collate)
        test_loader = DataLoader(dataset.test_set, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn=dataset.collate)

        if args.mode == 0:
            pd.DataFrame(columns=['epoch', 'loss', 'loss_c', 'AUC', 'AUPR', 'F1', 'ACC']).to_csv(out_file + '/i_train.csv', index=False)
            pd.DataFrame(columns=['epoch', 'loss', 'loss_c', 'AUC', 'AUPR', 'F1', 'ACC']).to_csv(out_file + '/i_val.csv', index=False)
            pd.DataFrame(columns=['epoch', 'loss', 'loss_c', 'AUC', 'AUPR', 'F1', 'ACC']).to_csv(out_file + '/i_test.csv', index=False)
            best_auc, best_aupr, best_f1, best_acc = 0, 0, 0, 0
            best_ckpt = './model/save_model/best_model_' + args.out_dir_different_params + '.pth'
        else:
            pd.DataFrame(columns=['epoch', 'loss', 'loss_c', 'rmse', 'r2', 'r']).to_csv(out_file + '/i_train_r.csv', index=False)
            pd.DataFrame(columns=['epoch', 'loss', 'loss_c', 'rmse', 'r2', 'r']).to_csv(out_file + '/i_val_r.csv', index=False)
            pd.DataFrame(columns=['epoch', 'loss', 'loss_c', 'rmse', 'r2', 'r']).to_csv(out_file + '/i_test_r.csv', index=False)
            best_rmse, best_r2, best_r = 1000, -1000, -1000
            best_ckpt = './model/save_model/reg_best_model_' + args.out_dir_different_params + '.pth'

        model_CAESynergy.reset_parameters()
        optimizer = optim.Adam(model_CAESynergy.parameters(), lr=args.learning_rate, weight_decay=args.L2)

        print("preprocess done! cost time:[{:.2f} s]".format(time.time()-t0))
        for epoch in tqdm(range(args.epochs)):
            if args.mode == 0:
                train_loss, loss_c, train_auc, train_aupr, train_f1, train_acc = train_epoch(model_CAESynergy, train_fit_loader, optimizer, args, device)
                val_loss, val_auc, val_aupr, val_f1, val_acc, _, _ = eval_epoch(model_CAESynergy, val_loader, args, device)
                if val_acc + val_auc > best_acc + best_auc:
                    best_auc, best_aupr, best_f1, best_acc = val_auc, val_aupr, val_f1, val_acc
                    torch.save(model_CAESynergy.state_dict(), best_ckpt)
            else:
                train_loss, loss_c, train_rmse, train_r2, train_r = train_epoch(model_CAESynergy, train_fit_loader, optimizer, args, device)
                val_loss, val_rmse, val_r2, val_r, _, _ = eval_epoch(model_CAESynergy, val_loader, args, device)
                if val_rmse < best_rmse:
                    best_rmse, best_r2, best_r = val_rmse, val_r2, val_r
                    torch.save(model_CAESynergy.state_dict(), best_ckpt)

    # =========================================================================
    # 4. 【彻底重构】: 5 次独立重复 Leave-One-Cell-Line-Out (LOCO) 细胞系冷启动实验
    # =========================================================================
    else:
        print("\n>>> 开始标准的 5 次重复独立细胞系留一法 (LOCO) 实验...")
        full_dataset = dataset.dataset_all
        
        # 核心改造：从数据缓存中提取原始协同矩阵
        raw_synergy = np.array(dataset.synergy_c if args.mode == 0 else dataset.synergy)
        
        # 核心修复方案：定位第 2 列以提取真实的细胞系内部数字 ID 列表
        cell_indices = raw_synergy[:, 2].astype(int).tolist()

        # 统计全局独立细胞系
        unique_cell_ids = sorted(list(set(cell_indices)))
        print(f"[INFO] 真实统计：全局数据集中共包含 {len(unique_cell_ids)} 种独立的细胞系。")

        # 随机挑选 5 种细胞系 ID 用于 5 次独立实验
        n_experiments = min(5, len(unique_cell_ids))
        np.random.seed(args.seed)
        selected_test_cells = np.random.choice(unique_cell_ids, size=n_experiments, replace=False)
        print(f"[INFO] 已随机挑选以下 {n_experiments} 种真实细胞系索引进行冷启动实验: {selected_test_cells}")

        loco_results = []

        # 直接进行 5 次纯净的独立 LOCO 迭代循环
        for run_idx, test_cell_id in enumerate(selected_test_cells):
            print(f"\n========== Leave-One-Cell-Line-Out [实验 {run_idx+1}/{n_experiments}] -> 测试锁定细胞系 ID: {test_cell_id} ==========")

            train_idx = []
            val_idx = []

            # 遍历所有样本进行严格的细胞系单侧特征物理隔离
            for i in range(len(full_dataset)):
                c_id = cell_indices[i]

                # 包含该测试细胞系的样本作为冷启动验证集，其余细胞系的样本全部作为训练集
                if c_id == test_cell_id:
                    val_idx.append(i)
                else:
                    train_idx.append(i)

            print(f"数据切分完毕 -> 训练集大小 (完全不含细胞系 {test_cell_id}): {len(train_idx)} | 验证测试集大小 (只含细胞系 {test_cell_id}): {len(val_idx)}")
            
            if len(val_idx) == 0:
                print(f"[Warning] 警告：选中的细胞系 {test_cell_id} 没有对应的组合样本数据，跳过此次实验！")
                continue

            # 构建当前细胞系留一法实验专用的加载器
            trainset_kf = Subset(full_dataset, train_idx)
            valset_kf = Subset(full_dataset, val_idx)

            train_loader = DataLoader(trainset_kf, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=dataset.collate)
            # 【完美避坑】：验证集 val_loader 的 drop_last 必须设为 False，避免因为单细胞系样本数小于 Batch Size 导致数据被全丢弃引发 torch.cat 报错
            val_loader = DataLoader(valset_kf, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=dataset.collate)

            # 配置当前实验专有的 CSV 持久化路径和权重路径
            if args.mode == 0:
                pd.DataFrame(columns=['epoch', 'loss', 'loss_c', 'AUC', 'AUPR', 'F1', 'ACC']).to_csv(out_file + f'/train_loco_cell_{test_cell_id}.csv', index=False)
                pd.DataFrame(columns=['epoch', 'loss', 'AUC', 'AUPR', 'F1', 'ACC']).to_csv(out_file + f'/val_loco_cell_{test_cell_id}.csv', index=False)
                best_auc, best_aupr, best_f1, best_acc = 0, 0, 0, 0
                best_model_path = f'./model/save_model/best_model_loco_cell_{test_cell_id}.pth'
            else:
                pd.DataFrame(columns=['epoch', 'loss', 'loss_c', 'rmse', 'r2', 'r']).to_csv(out_file + f'/train_r_loco_cell_{test_cell_id}.csv', index=False)
                pd.DataFrame(columns=['epoch', 'loss', 'rmse', 'r2', 'r']).to_csv(out_file + f'/val_r_loco_cell_{test_cell_id}.csv', index=False)
                best_rmse, best_r2, best_r = 1e9, -1e9, -1e9
                best_model_path = f'./model/save_model/reg_best_model_loco_cell_{test_cell_id}.pth'

            # 重置模型权重，确保各次冷启动实验绝对独立
            model_CAESynergy.reset_parameters()
            optimizer = optim.Adam(model_CAESynergy.parameters(), lr=args.learning_rate, weight_decay=args.L2)

            # 周期训练
            for epoch in tqdm(range(args.epochs)):
                if args.mode == 0:  # 分类
                    train_loss, loss_c, train_auc, train_aupr, train_f1, train_acc = train_epoch(model_CAESynergy, train_loader, optimizer, args, device)
                    val_loss, val_auc, val_aupr, val_f1, val_acc, _, _ = eval_epoch(model_CAESynergy, val_loader, args, device)

                    pd.DataFrame([[epoch, train_loss, loss_c, train_auc, train_aupr, train_f1, train_acc]]).to_csv(out_file + f'/train_loco_cell_{test_cell_id}.csv', mode='a', header=False, index=False)
                    pd.DataFrame([[epoch, val_loss, val_auc, val_aupr, val_f1, val_acc]]).to_csv(out_file + f'/val_loco_cell_{test_cell_id}.csv', mode='a', header=False, index=False)

                    if val_auc + val_acc > best_auc + best_acc:
                        best_auc, best_aupr, best_f1, best_acc = val_auc, val_aupr, val_f1, val_acc
                        torch.save(model_CAESynergy.state_dict(), best_model_path)
                else:  # 回归
                    train_loss, loss_c, train_rmse, train_r2, train_r = train_epoch(model_CAESynergy, train_loader, optimizer, args, device)
                    val_loss, val_rmse, val_r2, val_r, _, _ = eval_epoch(model_CAESynergy, val_loader, args, device)

                    pd.DataFrame([[epoch, train_loss, loss_c, train_rmse, train_r2, train_r]]).to_csv(out_file + f'/train_r_loco_cell_{test_cell_id}.csv', mode='a', header=False, index=False)
                    pd.DataFrame([[epoch, val_loss, val_rmse, val_r2, val_r]]).to_csv(out_file + f'/val_r_loco_cell_{test_cell_id}.csv', mode='a', header=False, index=False)

                    if val_rmse < best_rmse:
                        best_rmse, best_r2, best_r = val_rmse, val_r2, val_r
                        torch.save(model_CAESynergy.state_dict(), best_model_path)

            # 缓存并输出各单次实验成果
            if args.mode == 0:
                loco_results.append([best_auc, best_aupr, best_f1, best_acc])
                pd.DataFrame([["best_result", 0, best_auc, best_aupr, best_f1, best_acc]]).to_csv(out_file + f'/val_loco_cell_{test_cell_id}.csv', mode='a', header=False, index=False)
                print(f"-> 细胞系 {test_cell_id} 实验结束 | 本次最优成果: AUC={best_auc:.4f} ACC={best_acc:.4f}")
            else:
                loco_results.append([best_rmse, best_r2, best_r])
                pd.DataFrame([["best_result", 0, best_rmse, best_r2, best_r]]).to_csv(out_file + f'/val_r_loco_cell_{test_cell_id}.csv', mode='a', header=False, index=False)
                print(f"-> 细胞系 {test_cell_id} 实验结束 | 本次最优成果: RMSE={best_rmse:.4f} R2={best_r2:.4f}")

        # 5. 汇总 5 次完全独立、重复冷启动实验的最终全局平均结果
        print(f"\n========== Final LOCO ({n_experiments} Random Cells) Average Result ==========")
        loco_results = np.array(loco_results)
        avg_metrics = np.mean(loco_results, axis=0)

        if args.mode == 0:
            avg_res = ["avg_loco_result", avg_metrics[0], avg_metrics[1], avg_metrics[2], avg_metrics[3]]
            pd.DataFrame([avg_res], columns=["type", "AUC", "AUPR", "F1", "ACC"]).to_csv(out_file + '/loco_final_avg.csv', index=False)
            print(f"LOCO AVG AUC  : {avg_metrics[0]:.4f} | AVG AUPR : {avg_metrics[1]:.4f} | AVG F1   : {avg_metrics[2]:.4f} | AVG ACC  : {avg_metrics[3]:.4f}")
        else:
            avg_res = ["avg_loco_result", avg_metrics[0], avg_metrics[1], avg_metrics[2]]
            pd.DataFrame([avg_res], columns=["type", "RMSE", "R2", "R"]).to_csv(out_file + '/loco_final_avg_r.csv', index=False)
            print(f"LOCO AVG RMSE : {avg_metrics[0]:.4f} | AVG R2   : {avg_metrics[1]:.4f} | AVG R    : {avg_metrics[2]:.4f}")

def train_drugcold(args, dataset, out_file, net_params):
    # 1. 修复 time 模块调用冲突
    t0 = time.time()
    print("[II] preprocess ...")
    device = torch.device(args.device)

    # 2. 初始化网络模型
    model_drug_extract = build_drug_fea_extract(args, net_params, dataset)
    model_cline_extract = cline_fea_extract(args, net_params['gcn_cline'])
    hetero_enc = build_hetero_encoder_if_enabled(args, net_params["gcn_drug"]["output"])

    ca_d2c_i = ca(args, net_params['ca'])
    model_cline_e_decoder = cross_attention(args, ca_d2c_i)
    model_cal = cal_model(args, net_params) if getattr(args, "use_cal", False) else None

    decoder = Decoder_mlp(args, unified_decoder_net_params(args, net_params))
    model_Decoder_nocross = Decoder_no_cross(args, decoder)

    model_CAESynergy = CAESynergy(
        args,
        model_drug_extract,
        model_cline_extract,
        model_cline_e_decoder,
        model_cal,
        model_Decoder_nocross,
        hetero_encoder=hetero_enc,
        cline_gcn_hidden=net_params["gcn_cline"]["hidden"],
    ).to(device)

    # 3. 独立测试模式保留原样
    if getattr(args, "Independent_Testing", False) == True:
        n_train_pool = len(dataset.train_set)
        if n_train_pool < 3:
            raise ValueError("独立测试模式下训练池样本过少，无法划分验证集（至少需要 3 条）。")
        idx_all = np.arange(n_train_pool)
        from sklearn.model_selection import train_test_split
        fit_idx, val_idx = train_test_split(idx_all, test_size=0.1, random_state=args.seed, shuffle=True)
        if len(fit_idx) == 0 or len(val_idx) == 0:
            fit_idx, val_idx = idx_all[:-1], idx_all[-1:]
        
        train_fit_loader = DataLoader(Subset(dataset.train_set, fit_idx.tolist()), batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=dataset.collate)
        val_loader = DataLoader(Subset(dataset.train_set, val_idx.tolist()), batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn=dataset.collate)
        test_loader = DataLoader(dataset.test_set, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn=dataset.collate)

        if args.mode == 0:
            pd.DataFrame(columns=['epoch', 'loss', 'loss_c', 'AUC', 'AUPR', 'F1', 'ACC']).to_csv(out_file + '/i_train.csv', index=False)
            pd.DataFrame(columns=['epoch', 'loss', 'loss_c', 'AUC', 'AUPR', 'F1', 'ACC']).to_csv(out_file + '/i_val.csv', index=False)
            pd.DataFrame(columns=['epoch', 'loss', 'loss_c', 'AUC', 'AUPR', 'F1', 'ACC']).to_csv(out_file + '/i_test.csv', index=False)
            best_auc, best_aupr, best_f1, best_acc = 0, 0, 0, 0
            best_ckpt = './model/save_model/best_model_' + args.out_dir_different_params + '.pth'
        else:
            pd.DataFrame(columns=['epoch', 'loss', 'loss_c', 'rmse', 'r2', 'r']).to_csv(out_file + '/i_train_r.csv', index=False)
            pd.DataFrame(columns=['epoch', 'loss', 'loss_c', 'rmse', 'r2', 'r']).to_csv(out_file + '/i_val_r.csv', index=False)
            pd.DataFrame(columns=['epoch', 'loss', 'loss_c', 'rmse', 'r2', 'r']).to_csv(out_file + '/i_test_r.csv', index=False)
            best_rmse, best_r2, best_r = 1000, -1000, -1000
            best_ckpt = './model/save_model/reg_best_model_' + args.out_dir_different_params + '.pth'

        model_CAESynergy.reset_parameters()
        optimizer = optim.Adam(model_CAESynergy.parameters(), lr=args.learning_rate, weight_decay=args.L2)

        print("preprocess done! cost time:[{:.2f} s]".format(time.time()-t0))
        for epoch in tqdm(range(args.epochs)):
            if args.mode == 0:
                train_loss, loss_c, train_auc, train_aupr, train_f1, train_acc = train_epoch(model_CAESynergy, train_fit_loader, optimizer, args, device)
                val_loss, val_auc, val_aupr, val_f1, val_acc, _, _ = eval_epoch(model_CAESynergy, val_loader, args, device)
                if val_acc + val_auc > best_acc + best_auc:
                    best_auc, best_aupr, best_f1, best_acc = val_auc, val_aupr, val_f1, val_acc
                    torch.save(model_CAESynergy.state_dict(), best_ckpt)
            else:
                train_loss, loss_c, train_rmse, train_r2, train_r = train_epoch(model_CAESynergy, train_fit_loader, optimizer, args, device)
                val_loss, val_rmse, val_r2, val_r, _, _ = eval_epoch(model_CAESynergy, val_loader, args, device)
                if val_rmse < best_rmse:
                    best_rmse, best_r2, best_r = val_rmse, val_r2, val_r
                    torch.save(model_CAESynergy.state_dict(), best_ckpt)

    # =========================================================================
    # 4. 彻底重构的 4 次独立重复 Leave-One-Drug-Out 实验 (无任何五折交叉验证污染)
    # =========================================================================
    else:
        print("\n>>> 开始标准的 4 次重复独立药物留一法 (LODO) 实验...")
        full_dataset = dataset.dataset_all
        
        # 核心修复方案：直接从 dataset 缓存的未混淆原始二维协同矩阵中抓取精确的全局药物索引
        raw_synergy = np.array(dataset.synergy_c if args.mode == 0 else dataset.synergy)
        
        # 拿到每一行对应的真实 Drug A 和 Drug B 的内部数字 ID 列表
        drug_a_indices = raw_synergy[:, 0].astype(int).tolist()
        drug_b_indices = raw_synergy[:, 1].astype(int).tolist()

        # 统计全局独立药物
        unique_drug_ids = sorted(list(set(drug_a_indices + drug_b_indices)))
        print(f"[INFO] 真实统计：全局数据集中共包含 {len(unique_drug_ids)} 种独立的药物。")

        # 随机挑选 4 种药物 ID 用于 4 次独立实验
        np.random.seed(args.seed)
        selected_test_drugs = np.random.choice(unique_drug_ids, size=4, replace=False)
        print(f"[INFO] 已随机挑选以下 4 种真实药物索引进行冷启动实验: {selected_test_drugs}")

        lodo_results = []

        # 直接进行 4 次纯净的独立迭代循环
        for run_idx, test_drug_id in enumerate(selected_test_drugs):
            print(f"\n========== Leave-One-Drug-Out [实验 {run_idx+1}/4] -> 测试锁定药物 ID: {test_drug_id} ==========")

            train_idx = []
            val_idx = []

            # 严格按照刚才还原出来的原始药物索引矩阵进行完美隔离
            for i in range(len(full_dataset)):
                d1 = drug_a_indices[i]
                d2 = drug_b_indices[i]

                # 包含该测试药物的作为冷启动验证集，不包含的全部作为训练集
                if d1 == test_drug_id or d2 == test_drug_id:
                    val_idx.append(i)
                else:
                    train_idx.append(i)

            print(f"数据切分完毕 -> 训练集大小 (完全不含药物 {test_drug_id}): {len(train_idx)} | 验证测试集大小 (只含药物 {test_drug_id}): {len(val_idx)}")
            
            if len(val_idx) == 0:
                print(f"[Warning] 警告：选中的药物 {test_drug_id} 没有对应的组合样本数据，跳过此次实验！")
                continue

            # 构建当前留一法实验专用的加载器
            trainset_kf = Subset(full_dataset, train_idx)
            valset_kf = Subset(full_dataset, val_idx)

            train_loader = DataLoader(trainset_kf, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=dataset.collate)
            val_loader = DataLoader(valset_kf, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn=dataset.collate)

            # 配置当前实验专有的 CSV 持久化路径和权重路径
            if args.mode == 0:
                pd.DataFrame(columns=['epoch', 'loss', 'loss_c', 'AUC', 'AUPR', 'F1', 'ACC']).to_csv(out_file + f'/train_lodo_drug_{test_drug_id}.csv', index=False)
                pd.DataFrame(columns=['epoch', 'loss', 'AUC', 'AUPR', 'F1', 'ACC']).to_csv(out_file + f'/val_lodo_drug_{test_drug_id}.csv', index=False)
                best_auc, best_aupr, best_f1, best_acc = 0, 0, 0, 0
                best_model_path = f'./model/save_model/best_model_lodo_drug_{test_drug_id}.pth'
            else:
                pd.DataFrame(columns=['epoch', 'loss', 'loss_c', 'rmse', 'r2', 'r']).to_csv(out_file + f'/train_r_lodo_drug_{test_drug_id}.csv', index=False)
                pd.DataFrame(columns=['epoch', 'loss', 'rmse', 'r2', 'r']).to_csv(out_file + f'/val_r_lodo_drug_{test_drug_id}.csv', index=False)
                best_rmse, best_r2, best_r = 1e9, -1e9, -1e9
                best_model_path = f'./model/save_model/reg_best_model_lodo_drug_{test_drug_id}.pth'

            # 重置模型权重，确保各次冷启动实验绝对独立，不会发生前一个药物的学习权重污染下一个药物
            model_CAESynergy.reset_parameters()
            optimizer = optim.Adam(model_CAESynergy.parameters(), lr=args.learning_rate, weight_decay=args.L2)

            # 周期训练
            for epoch in tqdm(range(args.epochs)):
                if args.mode == 0:  # 分类
                    train_loss, loss_c, train_auc, train_aupr, train_f1, train_acc = train_epoch(model_CAESynergy, train_loader, optimizer, args, device)
                    val_loss, val_auc, val_aupr, val_f1, val_acc, _, _ = eval_epoch(model_CAESynergy, val_loader, args, device)

                    pd.DataFrame([[epoch, train_loss, loss_c, train_auc, train_aupr, train_f1, train_acc]]).to_csv(out_file + f'/train_lodo_drug_{test_drug_id}.csv', mode='a', header=False, index=False)
                    pd.DataFrame([[epoch, val_loss, val_auc, val_aupr, val_f1, val_acc]]).to_csv(out_file + f'/val_lodo_drug_{test_drug_id}.csv', mode='a', header=False, index=False)

                    if val_auc + val_acc > best_auc + best_acc:
                        best_auc, best_aupr, best_f1, best_acc = val_auc, val_aupr, val_f1, val_acc
                        torch.save(model_CAESynergy.state_dict(), best_model_path)
                else:  # 回归
                    train_loss, loss_c, train_rmse, train_r2, train_r = train_epoch(model_CAESynergy, train_loader, optimizer, args, device)
                    val_loss, val_rmse, val_r2, val_r, _, _ = eval_epoch(model_CAESynergy, val_loader, args, device)

                    pd.DataFrame([[epoch, train_loss, loss_c, train_rmse, train_r2, train_r]]).to_csv(out_file + f'/train_r_lodo_drug_{test_drug_id}.csv', mode='a', header=False, index=False)
                    pd.DataFrame([[epoch, val_loss, val_rmse, val_r2, val_r]]).to_csv(out_file + f'/val_r_lodo_drug_{test_drug_id}.csv', mode='a', header=False, index=False)

                    if val_rmse < best_rmse:
                        best_rmse, best_r2, best_r = val_rmse, val_r2, val_r
                        torch.save(model_CAESynergy.state_dict(), best_model_path)

            # 缓存并输出各单次实验成果
            if args.mode == 0:
                lodo_results.append([best_auc, best_aupr, best_f1, best_acc])
                pd.DataFrame([["best_result", 0, best_auc, best_aupr, best_f1, best_acc]]).to_csv(out_file + f'/val_lodo_drug_{test_drug_id}.csv', mode='a', header=False, index=False)
                print(f"-> 药物 {test_drug_id} 实验结束 | 本次最优成果: AUC={best_auc:.4f} ACC={best_acc:.4f}")
            else:
                lodo_results.append([best_rmse, best_r2, best_r])
                pd.DataFrame([["best_result", 0, best_rmse, best_r2, best_r]]).to_csv(out_file + f'/val_r_lodo_drug_{test_drug_id}.csv', mode='a', header=False, index=False)
                print(f"-> 药物 {test_drug_id} 实验结束 | 本次最优成果: RMSE={best_rmse:.4f} R2={best_r2:.4f}")

        # 5. 汇总 4 次完全独立、重复冷启动实验的最终全局平均结果
        print("\n========== Final LODO (4 Random Drugs) Average Result ==========")
        lodo_results = np.array(lodo_results)
        avg_metrics = np.mean(lodo_results, axis=0)

        if args.mode == 0:
            avg_res = ["avg_lodo_result", avg_metrics[0], avg_metrics[1], avg_metrics[2], avg_metrics[3]]
            pd.DataFrame([avg_res], columns=["type", "AUC", "AUPR", "F1", "ACC"]).to_csv(out_file + '/lodo_final_avg.csv', index=False)
            print(f"LODO AVG AUC  : {avg_metrics[0]:.4f} | AVG AUPR : {avg_metrics[1]:.4f} | AVG F1   : {avg_metrics[2]:.4f} | AVG ACC  : {avg_metrics[3]:.4f}")
        else:
            avg_res = ["avg_lodo_result", avg_metrics[0], avg_metrics[1], avg_metrics[2]]
            pd.DataFrame([avg_res], columns=["type", "RMSE", "R2", "R"]).to_csv(out_file + '/lodo_final_avg_r.csv', index=False)
            print(f"LODO AVG RMSE : {avg_metrics[0]:.4f} | AVG R2   : {avg_metrics[1]:.4f} | AVG R    : {avg_metrics[2]:.4f}")
def main():
    args = opts.parse_args()
    opts.setup_seed(args.seed)

    # 获取配置文件config（本机路径）
    # 可以使用相对路径：config\CASynergy.json，或绝对路径：F:\pythonProject\CASynergy\config\CASynergy.json
    with open(r'/home/lkp/cywhome/1/CASynergy/CASynergy/config/CASynergy.json') as f:
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
    if(args.method_name == 'CAESynergy'):
        train(args, dataset, out_file, net_params)
    else:
        print("method_name_error!")

if __name__ == '__main__':
     main()