from Backend.Util.Block import Block
from Backend.Util.PartitionManager import PartitionManager, Partition
from Backend.Util.AffinityMatrix import AffinityMatrix
from Configuration.config import Config
from Backend.Database.helpers import get_distinct_block_indexes
from Backend.Util.LogComp import LogLine, QueryInfo


from typing import Dict, List
from collections import deque
import math
import _pickle as cPickle
import pandas as pd
import csv


class BlockManager:
    def __init__(self):
        self.blocks : Dict[str, Block] = {}
    
    def add_block(self, block: Block):
        self.blocks[block.name] = block
    
    def get_block(self, bid):
        return self.blocks.get(bid)

    def set_blocks(self, blocks):
        self.blocks = blocks
    
    def set_block_pid(self, bid, pid):
        self.blocks.get(bid).set_partitionid(pid)
    
    def get_block_partition(self, bid):
        return self.blocks.get(bid).pid

    def find_partition_set(self, block_list):
        res = set()
        for block in block_list:
            b = self.blocks.get(block)
            if b is not None:
                res.add(b.pid)
        
        return list(res)
        
    
class QueryWindow:
    def __init__(self, window_size):
        self.window_size = window_size
        self.query_window = deque(maxlen=window_size)

    def insert_query(self, log_line):
        if len(self.query_window) >= self.window_size:
            self.query_window.popleft()  
        self.query_window.append(log_line)

    def clear_query_window(self):
        self.query_window.clear()

    def get_size(self):
        return len(self.query_window)


class Clump:
    def __init__(self):
        self.blocks = []
        self.candidate_partition = ""

    def is_empty(self):
        return len(self.blocks) == 0

    def set_candidate_partition(self, candidate_partition):
        self.candidate_partition = candidate_partition

    def set_tiles(self, tiles):
        self.blocks = tiles

    def add_tile(self, tid):
        self.blocks.append(tid)

    def clear(self):
        self.blocks.clear()
        self.candidate_partition = ""


class ClayPartitioner: 
    def __init__(self) -> None:
        self.max_queue_size = 500
        self.max_partition_size = 128
        self.partition_fill_portion = 0.9
        self.initial_empty_partitions = 0.1
        self.res_size_limit = 1000
        self.total_partition_access = 0
        self.total_received_queries = 0
        self.k = 1
        self.max_partition_load = 1
        self.weight_reset_threshold = 0.1

        self.p_manager = PartitionManager()
        self.b_manager = BlockManager()
        self.initialize_pmanager_bmanager()
        
        self.a_matrix = AffinityMatrix()
        self.q_window = QueryWindow(self.max_queue_size)
        
    def initialize_pmanager_bmanager(self):        
        print('initializing')
        initial_cap = math.ceil(self.max_partition_size * self.partition_fill_portion)
        tables = Config.table_list
        for tb in tables:
            block_indexes_df = get_distinct_block_indexes(tb)
            block_indexes = block_indexes_df['block_number'].tolist()

            new_par = Partition("p" + str(self.p_manager.get_increasing_index()))
            for bid in block_indexes:
                # create the block and insert it to partition
                block = Block(tb + '_' + str(bid))
                block.set_partitionid(new_par.partition_id)
                new_par.add_block(block.name)
                self.b_manager.add_block(block)
                if new_par.get_size() >= initial_cap:
                    self.p_manager.add_partition_with_load(new_par, 0)
                    new_par = Partition("p" + str(self.p_manager.get_increasing_index()))

            if new_par.get_size() > 0 and (not self.p_manager.has_partition(new_par.partition_id)):
                self.p_manager.add_partition_with_load(new_par, 0)

        additional_partitions = math.ceil(self.initial_empty_partitions * len(self.p_manager.partitions))
        for i in range(additional_partitions):
            # add some empty partitions in case:)
            self.p_manager.add_partition_with_load(Partition("p" + str(self.p_manager.get_increasing_index())), 0)

        print(f"initial number of partitions: {len(self.p_manager.partitions)}")

    def update_affinities(self, result_set):
        for bid in result_set:
            self.a_matrix.update_affinities(bid, result_set, self.b_manager, self.res_size_limit)

    def process_query(self, query:QueryInfo):
        self.q_window.insert_query(query)
        self.update_affinities(query.result_set)
        requested_partitions = self.b_manager.find_partition_set(query.result_set)
        self.total_partition_access += len(requested_partitions)
        self.total_received_queries += 1
        self.p_manager.update_loads(requested_partitions, self.a_matrix, self.k)
        if self.q_window.get_size() >= self.max_queue_size:
            partition_load = self.p_manager.get_overload_partitions(self.max_partition_load)
            if len(partition_load) == 0:
                print(f'No overload detected after {self.max_queue_size} queries.')
                self.a_matrix.multiply_weights(self.weight_reset_threshold)
                self.q_window.clear_query_window()
                return
            for pid, load in partition_load.items():
                most_overload_p:Partition = self.p_manager.partitions.get(pid)
                if most_overload_p.calculate_load(self.a_matrix, self.k):
                    continue
                
                updatePartitions(most_overload_p)
                

    #                 if (partitionManager.loads.get(pl.getKey()) > maxPartitionLoad) {
    #                     maxPartitionLoad = 1.05 *partitionManager.loads.get(pl.getKey());
    #                 }

    #                 overLoadCount++;
    #             }
    #             aMatrix.multiplyWeights(weightResetThreshold);
    #             qWindow.clearQueryWindow();
    #         }
    #         System.out.println("new query is processed");
        return

    def find_initial_c_partition(self, clump):
        return self.a_matrix.get_most_co_accessed_par_for_tile(clump[0], self.b_manager)

    def updatePartitions(self, most_overload_p:Partition):
        count = 0
        max_size_limit = self.get_max_clump_size()
        most_over_loaded_par_load = most_overload_p.calculate_load(self.a_matrix, self.k)
        print(f"MaxParLoad={self.max_partition_load} and found {most_overload_p.partition_id} 
              as most overloaded with load = {most_over_loaded_par_load}")
        m = []
        d = ""
        clump: Clump = Clump()
        look_ahead = 5
        affected_partitions = set()

        while most_overload_p.calculate_load(self.a_matrix, self.k) > self.max_partition_load:
            count += 1
            clumpNeighbor = self.find_clump_neighbor(m, d)
            if len(m) == 0:
                print("case 1-0")
                m.append(most_overload_p.get_hottest_block(self.a_matrix))
                affected_partitions.add(most_overload_p.partition_id)
                print(f"added {m[0]} to m[0]\n")
                d = self.find_initial_c_partition(m)
                print(f"d= {d}\n")
                if d == "":
                    print("could not find a candidate partition!!")

            elif clumpNeighbor != "":
                print("case 1-1")
                m.append(clumpNeighbor)
                affected_partitions.add(self.b_manager.get_block_partition(clumpNeighbor))
                d = updateCPartition(m, d, most_overload_p.partition_id);
                print("new d is " + d)
                if len(m) == self.max_partition_size:
                    print("case 1-1-2")
                    doneReParWithNewPartition(m, clump, affected_partitions, most_over_loaded_par_load);
                    break
            
            else:
                print("case 1-2")
                if not clump.is_empty():
                    print("case 1-2-1")
                    moveClumpToDestPartition(clump);
                    affected_partitions.add(clump.candidate_partition)
                    self.p_manager.updateLoads(list(affected_partitions), self.a_matrix, self.k);
                    break

                else:
                    if len(m) == max_size_limit:
                        print("case 1-2-2-2")
                        doneReParWithNewPartition(m, clump, affected_partitions, most_over_loaded_par_load);
                        break
                    if self.max_partition_load < most_over_loaded_par_load:
                        self.max_partition_load = 1.05 * most_over_loaded_par_load
                    break
                
            if feasible(m, d):
                print("case 2-1")
                clump.set_tiles(m)
                clump.set_candidate_partition(d)
            elif not clump.is_empty():
                print("case 2-2")
                print("reduced lookAhead")
                look_ahead -= 1
            if look_ahead == 0:
                print("lookAhead is zero");
                moveClumpToDestPartition(clump);
                affected_partitions.add(clump.candidate_partition);
                self.p_manager.updateLoads(list(affected_partitions), self.a_matrix, self.k);
                break

            print("end of update partition loop")
        print("Repartitioning is done")

    def get_max_clump_size(self):
        least_filled_partition = self.p_manager.get_least_filled_partition()
        return self.max_partition_size - self.p_manager.partitions[least_filled_partition].get_size()

    def find_clump_neighbor(self, m, d):
        max_freq = 0
        most_frequent_neighbor_id = ""
        for bid in m:
            temp_id = self.a_matrix.get_most_co_accessed_p(bid, self.b_manager, d, m)
            if temp_id == "":
                continue
            temp_freq = self.a_matrix.get_affinity(bid, temp_id) * self.a_matrix.get_tile_access_count(bid)
            if temp_freq > max_freq:
                max_freq = temp_freq
                most_frequent_neighbor_id = temp_id
        print("finding clump neighbor. found", most_frequent_neighbor_id)
        return most_frequent_neighbor_id


def read_workload_file(file_path):
    sep = '\|\|'
    loglines = []
    df = pd.read_csv(
        file_path,
        header=0,
        sep=sep,
        quoting=csv.QUOTE_ALL,
        quotechar='"',
        engine='python',
    )

    for index, row in df.iterrows():
        res_blocks = row['resultBlock'][1:-1].replace(' ', '').split(',')
        new_query = QueryInfo(row['statement'], result_set=res_blocks)
        if 'seqNum' in df.columns:
            sn = row['seqNum']
        else:
            sn = index
        new_logline = LogLine(sn, row['theTime'], row['clientIP'], row['row'], new_query)
        loglines.append(new_logline)
    
    return loglines

if __name__ == '__main__':
    train_workload_path = 'birds_test1B8.csv'

    partitioner = ClayPartitioner()

    loglines: List[LogLine] = read_workload_file(train_workload_path)

    for logline in loglines:
        partitioner.process_query(logline.query)

