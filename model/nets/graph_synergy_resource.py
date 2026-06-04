import os

import networkx as nx
import numpy as np
import pandas as pd
import torch


class GraphSynergyResource:
    """
    依据 cell_protein.csv +（若存在）protein-protein_network.xlsx 构图，
    按 GraphSynergy 原 data_loaders 的 get_neighbor_set 方式采样多跳蛋白邻居。
    无 drug_protein 时：药物分支的 hop0 邻域用该细胞系关联蛋白随机子集近似（与细胞上下文一致）。
    """

    def __init__(self, data_dir, drug_num, cell_names, n_hop=2, n_memory=32, seed=0):
        self.data_dir = os.path.normpath(data_dir)
        self.n_hop = int(n_hop)
        self.n_memory = int(n_memory)
        self.rng_master = np.random.RandomState(int(seed))
        self.drug_num = int(drug_num)
        self.cell_names = list(cell_names)

        cpi = pd.read_csv(os.path.join(self.data_dir, 'cell_protein.csv'))
        cpi_ids = cpi['target_id'].astype(np.int64).values
        cpi_ids = cpi_ids[cpi_ids != -666]
        prot_ids = np.unique(cpi_ids)
        ppi_path = os.path.join(self.data_dir, 'protein-protein_network.xlsx')
        if os.path.isfile(ppi_path):
            ppi = pd.read_excel(ppi_path)
            pa = pd.to_numeric(ppi['protein_a'], errors='coerce').dropna().astype(np.int64).values
            pb = pd.to_numeric(ppi['protein_b'], errors='coerce').dropna().astype(np.int64).values
            prot_ids = np.unique(np.concatenate([prot_ids, pa, pb]))
        prot_ids = sorted(int(x) for x in prot_ids)
        self.protein_id_to_ix = {pid: ix for ix, pid in enumerate(prot_ids)}
        self.num_proteins = len(self.protein_id_to_ix)
        self.default_protein_ix = 0

        self.graph = nx.Graph()
        if os.path.isfile(ppi_path):
            ppi = pd.read_excel(ppi_path)
            for _, row in ppi.iterrows():
                a, b = row['protein_a'], row['protein_b']
                if a not in self.protein_id_to_ix or b not in self.protein_id_to_ix:
                    continue
                self.graph.add_edge(self.protein_id_to_ix[int(a)], self.protein_id_to_ix[int(b)])

        self.cell_protein_ix = {}
        for cell, g in cpi.groupby('cell'):
            pids = []
            for tid in g['target_id'].astype(np.int64).values:
                if int(tid) == -666:
                    continue
                if int(tid) in self.protein_id_to_ix:
                    pids.append(self.protein_id_to_ix[int(tid)])
            self.cell_protein_ix[str(cell)] = list(set(pids)) if pids else [self.default_protein_ix]

    def _sample_neighbors(self, item_protein_ix_list, rng):
        """item_protein_ix_list: 当前 hop 起点蛋白索引列表（长度 n_memory）。"""
        neighbor_set = []
        cur = list(item_protein_ix_list)
        for hop in range(self.n_hop):
            if hop == 0:
                replace = len(cur) < self.n_memory
                target_list = list(rng.choice(cur, size=self.n_memory, replace=replace))
            else:
                neighbors = []
                for node in cur:
                    if node in self.graph:
                        neighbors.extend(list(self.graph.neighbors(node)))
                if len(neighbors) == 0:
                    neighbors = list(range(self.num_proteins))
                replace = len(neighbors) < self.n_memory
                target_list = list(rng.choice(neighbors, size=self.n_memory, replace=replace))
            neighbor_set.append(target_list)
            cur = target_list
        return neighbor_set

    def _stub_to_neighbor_lists(self, drug_i, drug_j, cell_k, batch_position):
        cell_name = self.cell_names[int(cell_k)]
        proteins = self.cell_protein_ix.get(str(cell_name), [self.default_protein_ix])

        rng_c = np.random.RandomState(int(self.rng_master.randint(0, 2 ** 31 - 1)))
        rng_d1 = np.random.RandomState((int(self.rng_master.randint(0, 2 ** 31 - 1)) ^ (int(drug_i) * 9973)) % (2 ** 31 - 1))
        rng_d2 = np.random.RandomState((int(self.rng_master.randint(0, 2 ** 31 - 1)) ^ (int(drug_j) * 7919)) % (2 ** 31 - 1))

        cell_lists = self._sample_neighbors(proteins, rng_c)
        d1_lists = self._sample_neighbors(proteins, rng_d1)
        d2_lists = self._sample_neighbors(proteins, rng_d2)
        return cell_lists, d1_lists, d2_lists

    def batch_context(self, stubs, device):
        """
        stubs: list of (drug_i, drug_j, cell_k) 与 Dataset.build_data 中保存的一致。
        返回 GraphNet.forward 所需 ctx（张量在 device 上）。
        """
        b = len(stubs)
        cell_nei = [torch.zeros(b, self.n_memory, dtype=torch.long) for _ in range(self.n_hop)]
        d1_nei = [torch.zeros(b, self.n_memory, dtype=torch.long) for _ in range(self.n_hop)]
        d2_nei = [torch.zeros(b, self.n_memory, dtype=torch.long) for _ in range(self.n_hop)]
        drug1 = torch.zeros(b, dtype=torch.long)
        drug2 = torch.zeros(b, dtype=torch.long)
        cells = torch.zeros(b, dtype=torch.long)
        for bi, (di, dj, ck) in enumerate(stubs):
            drug1[bi] = int(di)
            drug2[bi] = int(dj)
            cells[bi] = int(ck)
            c_l, a_l, b_l = self._stub_to_neighbor_lists(di, dj, ck, bi)
            for h in range(self.n_hop):
                cell_nei[h][bi] = torch.tensor(c_l[h], dtype=torch.long)
                d1_nei[h][bi] = torch.tensor(a_l[h], dtype=torch.long)
                d2_nei[h][bi] = torch.tensor(b_l[h], dtype=torch.long)
        return {
            'cells': cells.to(device),
            'drug1': drug1.to(device),
            'drug2': drug2.to(device),
            'cell_neighbors': [t.to(device) for t in cell_nei],
            'drug1_neighbors': [t.to(device) for t in d1_nei],
            'drug2_neighbors': [t.to(device) for t in d2_nei],
        }

    @staticmethod
    def from_dataset(dataset_root, drug_num, cell_names, n_hop=2, n_memory=32, seed=0):
        return GraphSynergyResource(dataset_root, drug_num, cell_names, n_hop=n_hop, n_memory=n_memory, seed=seed)
