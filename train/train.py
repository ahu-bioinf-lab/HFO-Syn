import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import BCELoss,MSELoss,KLDivLoss

from utils.utils import get_mutual
from .metrics import metrics_graph, regression_metric

def train_epoch(model_CAESnergy, train_loader, optimizer, args, device):
    model_CAESnergy.train()
    print("[III] training ...")
    # 计算metrics
    true_ls, pre_ls = [], []
    loss_train = 0
    loss_1_train = 0
    criterion = BCELoss()
    criterion_r = MSELoss()
    # 分类
    if args.mode == 0:
        # druga_se = druga_side_effect
        for batch, (druga, drugb, cline, cline_mask, label) in enumerate(train_loader):
            label = label.to(device)
            # 返回三个预测得分和节点的注意力得分
            score_c, score_d, score_b, att,conflict_loss = model_CAESnergy(druga.to(device), drugb.to(device), cline.to(device), cline_mask.to(device))
            loss = criterion(score_c, label) + conflict_loss * 0
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            # 指标计算
            loss_train += loss.item()
            loss_1_train += loss.item()
            pre_ls += score_c.cpu().detach().numpy().tolist()
            true_ls += label.cpu().detach().numpy().tolist()
        auc, aupr, f1, acc = metrics_graph(pre_ls, true_ls)
        return loss_train, loss_1_train, auc, aupr, f1, acc
    # 回归
    else:
        for batch, (druga, drugb, cline, label) in enumerate(train_loader):
            label = label.to(device)
            # 返回三个预测得分和节点的注意力得分
            score_c, score_d, score_b, att= model_CAESnergy(druga.to(device), drugb.to(device), cline.to(device))
            loss = criterion_r(score_c, label)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            loss_1_train += loss.item()
            pre_ls += score_c.cpu().detach().numpy().tolist()
            true_ls += label.cpu().detach().numpy().tolist()
        rmse, r2, r = regression_metric(pre_ls, true_ls)
        return loss_train, loss_1_train, rmse, r2, r

def eval_epoch(model_CAESnergy, eval_loader, args, device):
    model_CAESnergy.eval()
    # 计算metrics
    true_ls, pre_ls = [], []
    att_ls = []                 # 注意力分数的列表
    loss_train = 0
    criterion = BCELoss()
    criterion_r = MSELoss()
    with torch.no_grad():
        if args.mode == 0:
            for batch, (druga, drugb, cline, cline_mask, label) in enumerate(eval_loader):
                score_c, _, _, att = model_CAESnergy(druga.to(device), drugb.to(device), cline.to(device), cline_mask.to(device))
                label = label.to(device)
                loss = criterion(score_c, label)
                pre_ls += score_c.cpu().detach().numpy().tolist()
                true_ls += label.cpu().detach().numpy().tolist()
                att_ls += att.cpu().detach().numpy().tolist()
                loss_train += loss.item()
            auc, aupr, f1, acc = metrics_graph(pre_ls, true_ls)
            return loss_train, auc, aupr, f1, acc, att_ls, pre_ls
        else:
            for batch, (druga, drugb, cline, label) in enumerate(eval_loader):
                score_c, _, _, att = model_CAESnergy(druga.to(device), drugb.to(device), cline.to(device))
                label = label.to(device)
                loss = criterion_r(score_c,label)
                true_ls += label.cpu().detach().numpy().tolist()
                pre_ls += score_c.cpu().detach().numpy().tolist()
                att_ls += att.cpu().detach().numpy().tolist()
                loss_train += loss.item()
            rmse, r2, r = regression_metric(pre_ls, true_ls)
            return loss_train, rmse, r2, r, att_ls, pre_ls

def inter_qa_epoch0(model_CAESnergy, eval_loader, args, device):
    model_CAESnergy.eval()
    # 计算metrics
    true_ls, pre_ls = [], []
    att_ls = []                 # 注意力分数的列表
    loss_train = 0
    criterion = BCELoss()
    criterion_r = MSELoss()
    with torch.no_grad():
        if args.mode == 0:
            for batch, (druga, drugb, cline, cline_mask, label) in enumerate(eval_loader):
                # origin_feature causal_feature noncausal_feature
                o_fea, c_fea, nc_fea = model_CAESnergy(druga.to(device), drugb.to(device), cline.to(device), cline_mask.to(device))
                label = label.to(device)
                mutual_list = get_mutual(c_fea, nc_fea, o_fea, label, device)
        else:
            for batch, (druga, drugb, cline, label) in enumerate(eval_loader):
                score_c, _, _, att = model_CAESnergy(druga.to(device), drugb.to(device), cline.to(device))
                label = label.to(device)
                loss = criterion_r(score_c,label)
                true_ls += label.cpu().detach().numpy().tolist()
                pre_ls += score_c.cpu().detach().numpy().tolist()
                att_ls += att.cpu().detach().numpy().tolist()
                loss_train += loss.item()
            rmse, r2, r = regression_metric(pre_ls, true_ls)
            return loss_train, rmse, r2, r, att_ls, pre_ls
        
def eval_epoch(model_CAESnergy, eval_loader, args, device):
    model_CAESnergy.eval()
    true_ls, pre_ls = [], []
    att_ls = []
    loss_train = 0
    loss_conflict_total = 0  # 新增：用于记录测试集的冲突程度
    criterion = BCELoss()
    criterion_r = MSELoss()
    
    with torch.no_grad():
        if args.mode == 0:
            for batch, (druga, drugb, cline, cline_mask, label) in enumerate(eval_loader):
                # 1. 必须接收 5 个值，即使后几个在 eval 中不全用
                score_c, score_d, score_b, att, conflict_loss = model_CAESnergy(
                    druga.to(device), drugb.to(device), cline.to(device), cline_mask.to(device)
                )
                
                label = label.to(device)
                # 2. 这里的 loss 建议只计算主任务 loss，或者同步 train 的公式
                loss = criterion(score_c, label) 
                
                pre_ls += score_c.cpu().detach().numpy().tolist()
                true_ls += label.cpu().detach().numpy().tolist()
                att_ls += att.cpu().detach().numpy().tolist()
                
                loss_train += loss.item()
                loss_conflict_total += conflict_loss.item() # 记录冲突情况

            auc, aupr, f1, acc = metrics_graph(pre_ls, true_ls)
            
            # 建议返回增加 loss_conflict_total / len(eval_loader) 以便观察
            return loss_train, auc, aupr, f1, acc, att_ls, pre_ls

def inter_epoch(model_CAESnergy, eval_loader, args, device):
    model_CAESnergy.eval()
    # 计算metrics
    true_ls, pre_ls = [], []
    att_ls = []                 # 注意力分数的列表
    loss_train = 0
    criterion = BCELoss()
    criterion_r = MSELoss()
    with torch.no_grad():
        if args.mode == 0:
            # druga_se = druga_side_effect
            for batch, (druga, drugb, cline, cline_mask, label) in enumerate(eval_loader):
                score_c, _, _, att = model_CAESnergy(druga.to(device), drugb.to(device), cline.to(device), cline_mask.to(device))
                label = label.to(device)
                loss = criterion(score_c, label)
                pre_ls += score_c.cpu().detach().numpy().tolist()
                true_ls += label.cpu().detach().numpy().tolist()
                att_ls += att.cpu().detach().numpy().tolist()
                loss_train += loss.item()
            auc, aupr, f1, acc = metrics_graph(pre_ls, true_ls)
            return loss_train, auc, aupr, f1, acc, att_ls, pre_ls
        else:
            for batch, (druga, drugb, cline, label) in enumerate(eval_loader):
                score_c, _, _, att = model_CAESnergy(druga.to(device), drugb.to(device), cline.to(device))
                label = label.to(device)
                loss = criterion_r(score_c,label)
                true_ls += label.cpu().detach().numpy().tolist()
                pre_ls += score_c.cpu().detach().numpy().tolist()
                att_ls += att.cpu().detach().numpy().tolist()
                loss_train += loss.item()
            rmse, r2, r = regression_metric(pre_ls, true_ls)
            return loss_train, rmse, r2, r, att_ls, pre_ls

# 案例分析实验
def case_study_epoch(model_CAESnergy, eval_loader, args, device):
    model_CAESnergy.eval()
    # 计算metrics
    true_ls, pre_ls = [], []
    att_ls = []                 # 注意力分数的列表
    loss_train = 0
    criterion = BCELoss()
    criterion_r = MSELoss()
    with torch.no_grad():
        if args.mode == 0:
            # druga_se = druga_side_effect
            for batch, (druga, drugb, cline, cline_mask, label) in enumerate(eval_loader):
                score_c, _, _, att = model_CAESnergy(druga.to(device), drugb.to(device), cline.to(device), cline_mask.to(device))
                pre_ls += score_c.cpu().detach().numpy().tolist()
            return loss_train, pre_ls
        else:
            for batch, (druga, drugb, cline, label) in enumerate(eval_loader):
                score_c, _, _, att = model_CAESnergy(druga.to(device), drugb.to(device), cline.to(device))
                label = label.to(device)
                loss = criterion_r(score_c,label)
                true_ls += label.cpu().detach().numpy().tolist()
                pre_ls += score_c.cpu().detach().numpy().tolist()
                loss_train += loss.item()
            return loss_train, pre_ls

# 消融实验训练
def train_ablation_epoch(model_CAESnergy, train_loader, optimizer, args, device):
    model_CAESnergy.train()
    print("[III] training ...")
    # 计算metrics
    true_ls, pre_ls = [], []
    loss_train = 0
    loss_1_train = 0
    criterion = BCELoss()
    criterion_r = MSELoss()
    criterion_kl = KLDivLoss(reduction='batchmean')
    # 分类
    if args.mode == 0:
        # druga_se = druga_side_effect
        for batch, (druga, drugb, cline, cline_mask, label) in enumerate(train_loader):
            label = label.to(device)
            # 返回三个预测得分和节点的注意力得分
            score = model_CAESnergy(druga.to(device), drugb.to(device), cline.to(device), cline_mask.to(device))
            loss = criterion(score, label)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            # 指标计算
            loss_train += loss.item()
            pre_ls += score.cpu().detach().numpy().tolist()
            true_ls += label.cpu().detach().numpy().tolist()
        auc, aupr, f1, acc = metrics_graph(pre_ls, true_ls)
        return loss_train, auc, aupr, f1, acc
    # 回归
    else:
        for batch, (druga, drugb, cline, label) in enumerate(train_loader):
            label = label.to(device)
            # 返回三个预测得分和节点的注意力得分
            score = model_CAESnergy(druga.to(device), drugb.to(device), cline.to(device))
            loss = criterion_r(score, label)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            pre_ls += score.cpu().detach().numpy().tolist()
            true_ls += label.cpu().detach().numpy().tolist()
        rmse, r2, r = regression_metric(pre_ls, true_ls)
        return loss_train, loss_1_train, rmse, r2, r

def eval_ablation_epoch(model_CAESnergy, eval_loader, args, device):
    model_CAESnergy.eval()
    # 计算metrics
    true_ls, pre_ls = [], []
    att_ls = []                 # 注意力分数的列表
    loss_train = 0
    criterion = BCELoss()
    criterion_r = MSELoss()
    with torch.no_grad():
        if args.mode == 0:
            for batch, (druga, drugb, cline, cline_mask, label) in enumerate(eval_loader):
                score = model_CAESnergy(druga.to(device), drugb.to(device), cline.to(device), cline_mask.to(device))
                label = label.to(device)
                loss = criterion(score, label)
                pre_ls += score.cpu().detach().numpy().tolist()
                true_ls += label.cpu().detach().numpy().tolist()
                loss_train += loss.item()
            auc, aupr, f1, acc = metrics_graph(pre_ls, true_ls)
            return loss_train, auc, aupr, f1, acc, pre_ls
        else:
            for batch, (druga, drugb, cline, label) in enumerate(eval_loader):
                score = model_CAESnergy(druga.to(device), drugb.to(device), cline.to(device))
                label = label.to(device)
                loss = criterion_r(score,label)
                true_ls += label.cpu().detach().numpy().tolist()
                pre_ls += score.cpu().detach().numpy().tolist()
                loss_train += loss.item()
            rmse, r2, r = regression_metric(pre_ls, true_ls)
            return loss_train, rmse, r2, r, pre_ls