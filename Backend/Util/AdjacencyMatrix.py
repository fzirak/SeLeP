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

    def update_weight(self, p_id, n_id, delta_weight):
        if self.adjacencies.get(p_id) is None:
            self.add_node(p_id)
        prev_weight = self.adjacencies[p_id].get(n_id)
        self.adjacencies[p_id][n_id] = delta_weight if prev_weight is None else prev_weight + delta_weight
        # print(f'Partition {p_id}, increase weight for {n_id} = {prev_weight+delta_weight}')

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

