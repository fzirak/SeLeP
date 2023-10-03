from typing import List

from Configuration.config import Config
from .BackendUtilFunctions import *


class QueryInfo:
    def __init__(self, statement, result_set: List[str] = None, encoded_result=None, result_partitions=None):
        self.statement = statement
        self.result_set = result_set if result_set is not None else []
        self.result_partitions = result_partitions if result_partitions is not None else []
        self.encoded_result = encoded_result

    def calculate_query_result_mat_encoding(self, table_manager, requested_partitions, partition_manager, b_level=True):
        if len(self.result_set) == 0:
            return
        num_of_tbs, encoding_length = len(Config.table_list), Config.encoding_length
        block_encodings = []
        if b_level:
            for block in self.result_set:
                erb = calculate_block_matrix_encoding(block, encoding_length, num_of_tbs, table_manager)
                if erb is None:
                    continue
                block_encodings.append(erb)
        else:
            par_enc = []
            for par in requested_partitions:
                par_enc.append(partition_manager.get_partition_encoding(par))

        self.encoded_result = get_encoded_block_aggregation(par_enc, b_level)


class LogLine:
    def __init__(self, seq_num, the_time, client_ip, res_size, query: QueryInfo = None):
        self.seq_num = seq_num
        self.the_time = the_time
        self.client_ip = client_ip
        self.res_size = res_size
        self.query = query
