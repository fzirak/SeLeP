from . import AffinityMatrix
from .AdjacencyMatrix import DictBasedAdjacencyMatrix
from .BackendUtilFunctions import *
from Configuration.config import Config
from typing import Dict
import numpy as np
import pandas as pd

from ..Database.LRUCache import LRUCache


class Partition:
    def __init__(self, p_id, blocks=None):
        if blocks is None:
            blocks = []
        self.partition_id = p_id
        blocks_ = []
        for bid in blocks:
            if bid.rsplit('_', 1)[0] in Config.table_lookup:
                blocks_.append(bid)
        self.blocks = blocks_
        self.encoding = None
        self.load = 0

    def calculate_encoding(self, table_manager, num_of_tbs, encoding_length=32):
        if len(self.blocks) == 0:
            # self.encoding = np.zeros((num_of_tbs, 2 * encoding_length))
            self.encoding = np.zeros((num_of_tbs, encoding_length))
            return
        block_encodings = []
        tbnames = set([item.rsplit('_', 1)[0] for item in self.blocks])
        for block in self.blocks:
            erb = calculate_block_matrix_encoding(block, encoding_length, num_of_tbs, table_manager)
            if erb is None:
                continue
            block_encodings.append(erb)
        self.encoding = get_encoded_block_aggregation(block_encodings)

    def __str__(self):
        res = '[id: {}, blocks: {}]'.format(self.partition_id, self.blocks)
        return res

    def get_size(self):
        return len(self.blocks)

    def add_block_and_update_enc(self, block_id, block_encoding):
        self.blocks.append(block_id)
        self.update_encoding(block_encoding)

    def add_block(self, block_id):
        self.blocks.append(block_id)

    def update_encoding(self, block_encoding):
        if self.encoding is None:
            self.encoding = np.zeros((len(Config.table_list), Config.encoding_length))
            # self.encoding = np.zeros((len(Config.table_list), 2 * Config.encoding_length))

        self.encoding = update_encoding_aggregation(self.encoding, block_encoding, len(self.blocks) - 1)
    
        
    def calculate_load(self, a_matrix:AffinityMatrix.AffinityMatrix, k):
        load = 0
        for bid in self.blocks:
            block_affinityEntry = a_matrix.get_block_aff_entry(bid)
            if block_affinityEntry is None:
                continue

            for key, value in block_affinityEntry.freqs.items():
                if key in self.blocks:
                    continue
                load += k * value

        return load
    
    def get_hottest_block(self, a_matrix:AffinityMatrix.AffinityMatrix):
        max_exit_freq = 0
        res = ""
        for block in self.blocks:
            block_affinity_entry = a_matrix.get_block_affinities(block)
            exit_freq = 0
            if block_affinity_entry is None:
                continue
            for key, value in block_affinity_entry.freqs.items():
                if key in self.blocks:
                    continue
                exit_freq += value
            if exit_freq > max_exit_freq:
                max_exit_freq = exit_freq
                res = block
        return res


class PartitionManager:
    def __init__(self):
        self.partitions: Dict[str, Partition] = {}
        self.loads = {}
        self.increasing_index = 0
        self.partition_graph: DictBasedAdjacencyMatrix = DictBasedAdjacencyMatrix()
        loads = {}

    def read_partitions_from_file(self, file_path):
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line == '':
                    continue
                if ',' not in line:
                    print(f'no comma in {line}')
                    return
                partition_id = line[1:line.index(',')]
                # if partition_id == 'p1231':
                #     print('partition 1231 was created')
                blocks_str = line[line.index('[') + 1:line.index(']')]
                blocks = blocks_str.split(', ')
                if blocks_str == '':
                    blocks = {}
                self.partitions[partition_id] = Partition(partition_id, blocks)
        self.increasing_index = len(self.partitions)

    def get_max_pid(self):
        numbers = [int(key[1:]) for key in self.partitions.keys()]
        return max(numbers)

    def create_partition_graph_from_aff_matrix(self, aff_matrix: AffinityMatrix.AffinityMatrix):
        for partition in self.partitions.values():
            partition_out_weights = {}
            for block in partition.blocks:
                for neighbor_id, neighbor_freq in aff_matrix.get_block_affinities(block).items():
                    neighbor_partition_id = aff_matrix.get_block_partition_id(neighbor_id)
                    weight = partition_out_weights.get(neighbor_partition_id)
                    if weight is not None:
                        partition_out_weights[neighbor_partition_id] = weight + neighbor_freq
                    else:
                        partition_out_weights[neighbor_partition_id] = neighbor_freq
            self.partition_graph.add_node(partition.partition_id, partition_out_weights)

    def calculate_partition_encodings(self, table_manager):
        num_of_tbs = len(Config.table_list)
        encoding_length = Config.encoding_length
        for partition in self.partitions.values():
            partition.calculate_encoding(table_manager, num_of_tbs, encoding_length)

    def add_neighbors_to_pred_list(self, pred_partitions, k):
        if k < 1:
            return pred_partitions
        prev_pred_list = pred_partitions.copy()
        for pid in prev_pred_list:
            pred_partitions = self.add_partition_neighbors(pid, pred_partitions, k)
        return pred_partitions

    def add_partition_neighbors(self, p_id, pred_partitions, k):
        if p_id not in self.partition_graph.adjacencies:
            return pred_partitions
        sorted_dict = sorted(self.partition_graph.adjacencies[p_id].items(), key=lambda x: x[1], reverse=True)
        count = 0
        for key, value in sorted_dict:
            if key == p_id:
                continue
            if key not in pred_partitions:
                pred_partitions.append(key)
                count += 1
            if count >= k:
                break
        return pred_partitions

    def get_partition_encoding(self, p_id):
        if p_id not in self.partitions:
            print(f'partition id {p_id} is missing in partitionManager')
        if self.partitions.get(p_id) is None:
            print(f'partition id {p_id} is None in partitionManager')
        return self.partitions.get(p_id).encoding

    def get_partition_encodings(self):
        default_encoding = np.zeros((len(Config.table_list), Config.encoding_length))
        # default_encoding = np.zeros((len(Config.table_list), 2 * Config.encoding_length))
        sorted_pid = sorted(self.partitions.keys())
        partition_encodings = [self.partitions[pid].encoding if self.partitions[pid].encoding is not None
                               else default_encoding for pid in sorted_pid]

        return np.array(partition_encodings)

    def __str__(self):
        res = ''
        for par in self.partitions.values():
            res += str(par)
            res += '\n'
        return res

    def get_partition(self, p_id) -> Partition:
        return self.partitions.get(p_id)

    def get_increasing_index(self):
        self.increasing_index += 1
        return self.increasing_index

    def add_partition(self, new_partition: Partition):
        self.partitions[new_partition.partition_id] = new_partition

    def add_partition_with_load(self, new_partition: Partition, load):
        new_partition.load = load
        self.partitions[new_partition.partition_id] = new_partition

    def has_partition(self, partition_id):
        return partition_id in self.partitions

    def update_partition_graph(self, requested_partitions):
        delta = 1 / len(requested_partitions)
        for p_id in requested_partitions:
            for cp_id in requested_partitions:
                if p_id == cp_id:
                    continue
                self.partition_graph.update_weight(p_id, cp_id, delta)

    def check_neighborhood(self, p_id, partitions):
        max_weight = 0
        for partition in partitions:
            w = self.partition_graph.get_edge_weight(s=partition, e=p_id)
            max_weight = max(w, max_weight)
        return max_weight

    def put_partition_in_cache(self, p_id, cache: LRUCache):
        p_block_list = self.partitions.get(p_id).blocks
        for b_id in p_block_list:
            cache.put(b_id, increase_hit=False, insert_type='p')

    def put_partition_in_list(self, p_id, lst):
        p_block_list = self.partitions.get(p_id).blocks
        for b_id in p_block_list:
            lst.append(b_id)
        return lst

    def get_partition_size_dict(self):
        par_sizes = {}
        for par in self.partitions:
            par_sizes[par] = len(self.partitions[par].blocks)
        
        return par_sizes

    def get_partition_size(self, pid):
        if pid not in self.partitions:
            print(f'{pid} not in pmanager')
            return -1
        return len(self.partitions[pid].blocks)
    
    def update_loads(self, pids, a_matrix, k):
        for pid in pids:
            if pid not in self.partitions:
                continue
            self.loads[pid] = self.partitions[pid].calculate_load(a_matrix, k)

    def get_overload_partitions(self, max_partition_load):
        partition_load = {}
        for pid in self.partitions:
            if pid not in self.loads:
                continue

            if self.loads[pid] > max_partition_load:
                partition_load[pid] = self.loads[pid]
            
        sorted_items = sorted(partition_load.items(), key=lambda x: x[1])
        sorted_dict = dict(sorted_items)
        return sorted_dict
    
    def get_least_filled_partition(self):
        least_cap = float('inf')
        res_partition = ""
        for p in self.partitions.values():
            if p.get_size() < least_cap:
                least_cap = p.get_size()
                res_partition = p.partition_id
        return res_partition
    