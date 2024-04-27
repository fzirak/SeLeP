from Backend.Models.ModelHelper import convert_adr_to_bid, get_LBA
from Backend.Util import PartitionManager
from Configuration.config import Config


def add_tbextend_to_list(tbe, preds):
    splited = tbe.split('/')
    extend = int(splited[1])
    min_adr = min(Config.extend_size * extend, Config.tb_bid_range[splited[0]][0])
    max_adr = max(Config.extend_size * (extend + 1), Config.tb_bid_range[splited[0]][1])
    for i in range(min_adr, max_adr):
        preds.append(f'{splited[0]}_{i}')
    return preds


class TraditionalModelWindow:
    def __init__(self, size=128, trigger_threshold=13, extend_type='table', num_digits=5):
        self.trace = []  # in table type, values are bids, and in partition type, values are [bid, pid]
        self.trace_map = {}
        self.last_bid = None
        self.size = size
        self.trigger_threshold = trigger_threshold
        self.extend_type = extend_type  # table or partition or naive
        self.num_digits = num_digits

    def add_bid(self, bid, pid=None):
        splited_bid = bid.rsplit('_', 1)
        tbi = Config.table_lookup[splited_bid[0]]
        # block_num = str(splited_bid[1]).zfill(self.num_digits)
        # b_adr = f'{tbi}{block_num}'
        b_adr = get_LBA(bid)
        popped = None
        if len(self.trace) >= self.size:
            popped = self.trace.pop(0)

        if self.extend_type == 'partition':
            if popped is not None:
                self.trace_map_pop(popped[1], popped[0])
            if pid is None:
                print('None value for pid, abort!')
                return
            self.trace.append([bid, pid])
            if pid not in self.trace_map:
                self.trace_map[pid] = []
            self.trace_map[pid].append(bid)

        elif self.extend_type == 'table':
            if popped is not None:
                self.trace_map_pop(popped.rsplit('_', 1)[0], popped)
            extend_id = f'{splited_bid[0]}/{int(int(splited_bid[1])/Config.extend_size)}'
            full_adr = f'{extend_id}_{splited_bid[1]}'
            self.trace.append(full_adr)
            if extend_id not in self.trace_map:
                self.trace_map[extend_id] = []
            self.trace_map[extend_id].append(full_adr)
            self.last_bid = int(b_adr)


        elif self.extend_type == 'naive':
            if self.last_bid is None:
                self.last_bid = int(b_adr)
                return
            self.trace.append(int(b_adr) - self.last_bid)
            
        self.last_bid = int(b_adr)

    def trace_map_pop(self, key, bid):
        try:
            self.trace_map[key].remove(bid)
        except ValueError:
            print('the popped value does not exist in the map')

    def make_prefetch_decision(self, partition_manager: PartitionManager.PartitionManager, strategy='linear'):
        preds = []

        if len(self.trace) < 1:
            return preds
        
        if strategy == 'linear':
            temp_last_adr = self.last_bid
            adr_preds = []
            for i in range(prefetch_max_count):
                adr_preds.append(temp_last_adr + 1)
                temp_last_adr = temp_last_adr + 1
            preds = convert_adr_to_bid([adr_preds])[0]

        else:
            if self.extend_type == 'partition':
                if len(self.trace) < self.trigger_threshold:
                    return preds
                if strategy == 'random':
                    partition_preds = []
                    for pid, p_trace in self.trace_map.items():
                        if len(p_trace) > self.trigger_threshold:
                            partition_preds.append(pid)
                            partition_manager.put_partition_in_list(pid, preds)  
                # elif strategy == 'linear':
                    # for pid, p_trace in self.trace_map.items():
                    #     if len(p_trace) > self.trigger_threshold:
                    #         tb_trace_map = {}
                    #         for bid in p_trace:
                    #             bid_splited = bid.rsplit('_', 1)
                    #             if not bid_splited[0] in tb_trace_map:
                    #                 tb_trace_map[bid_splited[0]] = [int(bid_splited[1]), 1]
                    #                 continue
                    #             last_adr, count = tb_trace_map[bid_splited[0]]
                    #             if abs(int(bid_splited[1]) - last_adr) != 1:  # it is not sequential, reset!
                    #                 count = 0
                    #             tb_trace_map[bid_splited[0]] = [int(bid_splited[1]), count+1]
                    #         sum_count = 0
                    #         for tb, tb_trace in tb_trace_map.items():
                    #             sum_count += tb_trace[1]
                    #         if sum_count > self.trigger_threshold:
                    #             partition_manager.put_partition_in_list(pid, preds) 

            elif self.extend_type == 'table':
                if len(self.trace) < self.trigger_threshold:
                    return preds
                if strategy == 'random':
                    for tbe, tb_trace in self.trace_map.items():
                        if len(tb_trace) > self.trigger_threshold:
                            preds = add_tbextend_to_list(tbe, preds)
                elif strategy == 'linear':
                    for tbe, tb_trace in self.trace_map.items():
                        if len(tb_trace) > self.trigger_threshold:
                            tb_trace.sort()
                            count = 0
                            last_adr = None
                            for bid in tb_trace:
                                if last_adr is None:
                                    last_adr = int(bid.rsplit('_', 1)[1])
                                    continue
                                if int(bid.rsplit('_',1)[1]) - last_adr == 0:
                                    continue
                                if abs(int(bid.rsplit('_', 1)[1]) - last_adr) != 1:
                                    count = 0
                                count += 1
                            if count > self.trigger_threshold:
                                preds = add_tbextend_to_list(tbe, preds)

            elif self.extend_type == 'naive':
                most_frequent_lba = max(self.trace, key=self.trace.count)
                # most_frequent_lba = self.trace[-1]
                prefetch_max_count = Config.prefetching_k * Config.max_partition_size
                temp_last_adr = self.last_bid
                adr_preds = []
                for i in range(prefetch_max_count):
                    adr_preds.append(temp_last_adr + most_frequent_lba)
                    temp_last_adr = temp_last_adr + most_frequent_lba
                preds = convert_adr_to_bid([adr_preds])[0]

        return preds
