from typing import Dict


class DictBasedAdjacencyMatrix:
    def __init__(self, adjacencies=None):
        if adjacencies is None:
            adjacencies = {}
        self.adjacencies: Dict[str, Dict[str, float]] = adjacencies

    def add_node(self, p_id, p_neighbors: Dict[str, float] = None):
        if p_neighbors is None:
            p_neighbors = {}
        self.adjacencies[p_id] = p_neighbors

    def update_weight(self, p_id, n_id, delta_weight, do_normalize=False):
        if self.adjacencies.get(p_id) is None:
            self.add_node(p_id)
        prev_weight = self.adjacencies[p_id].get(n_id)
        self.adjacencies[p_id][n_id] = delta_weight if prev_weight is None else prev_weight + delta_weight
        # print(f'Partition {p_id}, increase weight for {n_id} = {prev_weight+delta_weight}')
        if do_normalize:
            adj = self.adjacencies[p_id]
            total = sum(adj.values(), 0.0)
            adj = {k: v / total for k, v in adj.items()}
            self.adjacencies[p_id] = adj
    
    def update_weight_with_time_effect(self, p_id, n_ids, delta_weight, par_cnt):
        if self.adjacencies.get(p_id) is None:
            self.add_node(p_id)
            for n_id in n_ids:
                self.adjacencies[p_id][n_id] = 1/par_cnt
            self.adjacencies[p_id]['default'] = 1/par_cnt
        else:
            for n_id in n_ids:
                if n_id not in self.adjacencies[p_id]:
                    self.adjacencies[p_id][n_id] = self.adjacencies[p_id]['default']

        for n_id in self.adjacencies[p_id]:
            prev_prob = self.adjacencies[p_id][n_id]
            if n_id in n_ids:
                self.adjacencies[p_id][n_id] = delta_weight + ((1-delta_weight)*prev_prob)
            else:
                self.adjacencies[p_id][n_id] = (1-delta_weight)*prev_prob

    def normalize_adjacencies(self):
        for p_id in self.adjacencies:
            adj = self.adjacencies[p_id]
            total = sum(adj.values(), 0.0)
            adj = {k: v / total for k, v in adj.items()}
            self.adjacencies[p_id] = adj
    
    def __str__(self):
        return f'{self.adjacencies}'

    def get_edge_weight(self, s, e):
        w = 0
        if self.adjacencies.get(s) is None:
            return w
        weight = self.adjacencies[s].get(e)
        if weight is not None:
            w = weight
        return w

    def get_weighted_neighbors(self, p_id):
        return self.adjacencies.get(p_id)

