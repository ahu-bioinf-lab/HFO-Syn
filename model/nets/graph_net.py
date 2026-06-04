import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphNet(nn.Module):
    def __init__(self, args, resource, l1_decay=1e-5):
        super(GraphNet, self).__init__()
        self.resource = resource
        protein_num = resource.num_proteins
        cell_num = len(resource.cell_names)
        drug_num = resource.drug_num
        emb_dim = max(32, args.drug_dim // 2)
        n_hop = resource.n_hop

        self.protein_num = int(protein_num)
        self.cell_num = int(cell_num)
        self.drug_num = int(drug_num)
        self.emb_dim = int(emb_dim)
        self.n_hop = int(n_hop)
        self.l1_decay = float(l1_decay)
        self.out_dim = int(args.drug_dim)

        self.protein_embedding = nn.Embedding(self.protein_num, self.emb_dim)
        self.cell_embedding = nn.Embedding(self.cell_num, self.emb_dim)
        self.drug_embedding = nn.Embedding(self.drug_num, self.emb_dim)
        self.aggregation_function = nn.Linear(self.emb_dim * self.n_hop, self.emb_dim)
        self.to_dc = nn.Linear(self.emb_dim * 3, self.out_dim)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)) and hasattr(m, 'reset_parameters'):
                m.reset_parameters()

    def _get_neighbor_emb(self, neighbors):
        out = []
        for hop in range(self.n_hop):
            out.append(self.protein_embedding(neighbors[hop]))
        return out

    def _interaction_aggregation(self, item_embeddings, neighbors_emb_list):
        interact_list = []
        for hop in range(self.n_hop):
            neighbor_emb = neighbors_emb_list[hop]
            contributions = (neighbor_emb * item_embeddings.unsqueeze(1)).sum(dim=-1)
            contributions_normalized = F.softmax(contributions, dim=1)
            i = (neighbor_emb * contributions_normalized.unsqueeze(2)).sum(dim=1)
            item_embeddings = i
            interact_list.append(i)
        return interact_list

    def _aggregation(self, item_i_list):
        item_i_concat = torch.cat(item_i_list, dim=1)
        return self.aggregation_function(item_i_concat)

    def forward(self, ctx):
        cells = ctx['cells']
        drug1 = ctx['drug1']
        drug2 = ctx['drug2']
        cell_neighbors = ctx['cell_neighbors']
        drug1_neighbors = ctx['drug1_neighbors']
        drug2_neighbors = ctx['drug2_neighbors']

        cell_embeddings = self.cell_embedding(cells)
        drug1_embeddings = self.drug_embedding(drug1)
        drug2_embeddings = self.drug_embedding(drug2)

        cell_neighbors_emb_list = self._get_neighbor_emb(cell_neighbors)
        drug1_neighbors_emb_list = self._get_neighbor_emb(drug1_neighbors)
        drug2_neighbors_emb_list = self._get_neighbor_emb(drug2_neighbors)

        cell_i_list = self._interaction_aggregation(cell_embeddings, cell_neighbors_emb_list)
        drug1_i_list = self._interaction_aggregation(drug1_embeddings, drug1_neighbors_emb_list)
        drug2_i_list = self._interaction_aggregation(drug2_embeddings, drug2_neighbors_emb_list)

        cell_vec = self._aggregation(cell_i_list)
        drug1_vec = self._aggregation(drug1_i_list)
        drug2_vec = self._aggregation(drug2_i_list)

        h = torch.cat([drug1_vec, drug2_vec, cell_vec], dim=-1)
        return self.to_dc(h)
