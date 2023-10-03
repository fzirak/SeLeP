import os
import time
from typing import List, Dict
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import model_from_json

from Backend.Database.LRUCache import LRUCache
from Backend.Database.helpers import insert_block_to_cache
from Backend.Models import ModelHelper as model_utils
from Backend.Util.LogComp import LogLine
from Backend.Util.TableManager import TableManager
from Backend.Util.TraditionalModelWindow import TraditionalModelWindow
from Utils import PCACalculator as pca_c
from Backend.Util import Table, TableManager, PartitionManager, AffinityMatrix
from Backend.Database import helpers as db_helper
from Configuration.config import Config
import _pickle as cPickle

import Utils.utilFunction as general_util
import numpy as np
import pprint
import subprocess
from tqdm import tqdm
import pandas as pd
import csv
csv.field_size_limit(100 * 1024 * 1024) 

import random

tests = []
cache_size = 66000

class ActionTimestamp:
    def __init__(self, name, begin=0.0, end=0.0):
        self.name = name
        self.begin = begin
        self.end = end

    def get_duration(self):
        return self.end - self.begin

    def __str__(self) -> str:
        return f'{self.name}: {self.get_duration()}'

    def __repr__(self) -> str:
        return f'{self.name}: {self.get_duration()}'


def get_tables_pca(accuracy=0.95):
    """
    the function first gets list of all tables and the non-numerical columns of each table
    in a dict format. Then, calculates the pca of all tables.
    :param accuracy:
    :return: a list containing tables' pca
    """
    tables, feature_exclude_dict = pca_c.get_pca_table_info()
    print(f'list of tables in config: {Config.table_list}\nand list of tables in get_table_pca: {tables}')
    if Config.db_name == 'tpcds':
        tables_pca = pca_c.pca_calculator2(tables, feature_exclude_dict, accuracy)    
    else:
        tables_pca = pca_c.pca_calculator(tables, feature_exclude_dict, accuracy)
    return tables_pca


def create_table_manager(read_from_file, latent_dim, epoch_no, file_path=''):
    if file_path == '':
        file_path = f'./SavedFiles/TableManagers/{Config.db_name}table_managerB{Config.logical_block_size}P{Config.max_partition_size}.p'

    if not read_from_file:
        table_manager = TableManager.TableManager()
        tables_pca = get_tables_pca()

        for tb_pca in tables_pca:
            print(f'in create table manager, working on {tb_pca.table_name}')
            new_table = Table.Table(tb_pca.table_name, tb_pca)
            # find the block boundaries
            block_indexes = db_helper.get_block_indexes(new_table.name)
            # get block pca values
            new_table.extract_block_pcas(block_indexes)
            table_manager.add_table(new_table)
            print(f'added {new_table.name} to table manager')
        # all tables have their blocks pca value, using these values and the autoencoder
        # calculate the encoding of the blocks
        table_manager.calculate_table_encodings(latent_dim, epoch_no)
        cPickle.dump(table_manager, open(file_path, 'wb'))
        return table_manager
    return cPickle.load(open(file_path, 'rb'))


def create_affinity_matrix(read_from_file, partition_manager, i='', is_navi=False): #NAVI: make is_navi=True
    file_suffix = f'B{Config.logical_block_size}P{Config.max_partition_size}'
    file_name = 'navi_affinityMatrix' if is_navi else 'affinityMatrix32'
    if i:
        file_suffix = file_suffix + f'_i'
    if not read_from_file:
        aff_matrix = AffinityMatrix.AffinityMatrix()
        aff_matrix.read_affinities_from_file(
            f'{Config.db_name}affinityMatrix{file_suffix}.txt'
        )
        aff_matrix.set_blocks_partition_id(partition_manager)
        cPickle.dump(aff_matrix, open(f'./SavedFiles/AffinityMatrices/{Config.db_name}{file_name}{file_suffix}.p', 'wb'))
        return aff_matrix
    return cPickle.load(open(f'./SavedFiles/AffinityMatrices/{Config.db_name}{file_name}{file_suffix}.p', 'rb'))


def create_par_manager_and_aff_matrix(read_par_manager, read_aff_matrix, table_manager, i='', is_navi=False): #NAVI
    file_suffix = f'B{Config.logical_block_size}P{Config.max_partition_size}'
    file_name = 'navi_partition_manager' if is_navi else 'partition_manager32'
    if i:
        file_suffix = file_suffix + f'_i'
    if read_par_manager:
        partition_manager = cPickle.load(open(f'./SavedFiles/PartitionManagers/{Config.db_name}{file_name}{file_suffix}.p', 'rb'))
        aff_matrix = create_affinity_matrix(read_aff_matrix, partition_manager, i)
        return partition_manager, aff_matrix
    else:
        partition_manager = PartitionManager.PartitionManager()
        partition_manager.read_partitions_from_file(
            f'{Config.db_name}partitions{file_suffix}.txt'
        )
        partition_manager.calculate_partition_encodings(table_manager)
        aff_matrix = create_affinity_matrix(read_aff_matrix, partition_manager, )
        partition_manager.create_partition_graph_from_aff_matrix(aff_matrix)
        cPickle.dump(partition_manager, open(f'./SavedFiles/PartitionManagers/{Config.db_name}{file_name}{file_suffix}.p', 'wb'))
        return partition_manager, aff_matrix
    

def create_pm_af_for_configs(read_par_manager, read_aff_matrix, table_manager, suffix):
    file_suffix = f'{suffix}B{Config.logical_block_size}P{Config.max_partition_size}'
    if 'adapt' in suffix:
        file_name = f'{Config.db_name}_adapt_partitions{suffix[6:]}B{Config.logical_block_size}P{Config.max_partition_size}.txt'
        print("'adapt' in suffix")
        print(file_name)
    else:
        f'{Config.db_name}pManager_{file_suffix}.txt'

    if read_par_manager:
        partition_manager = cPickle.load(open(f'./SavedFiles/PartitionManager/{Config.db_name}partition_manager32{file_suffix}.p', 'rb'))
        aff_matrix = AffinityMatrix.AffinityMatrix()
        return partition_manager, aff_matrix
    else:
        partition_manager = PartitionManager.PartitionManager()
        partition_manager.read_partitions_from_file(
            f'{file_name}'
        )
        partition_manager.calculate_partition_encodings(table_manager)
        aff_matrix = AffinityMatrix.AffinityMatrix()
        cPickle.dump(partition_manager, open(f'./SavedFiles/PartitionManager/{Config.db_name}partition_manager32{file_suffix}.p', 'wb'))
        return partition_manager, aff_matrix


def get_bid_offset_map(bid_ranges):
    offset_map = {}
    init_offset = 0
    for tb in Config.table_list:
        offset_map[tb] = init_offset
        init_offset += bid_ranges[tb][1] + (100 - (bid_ranges[tb][1] % 100))    
    return offset_map


def get_tables_bid_range(load_from_file=True):
    file_name = f'{Config.db_name}_tb_bid_rangeWT{len(Config.table_list)}WB{Config.logical_block_size}.p'
    try:
        if load_from_file:
            tables_bid_range = cPickle.load(open(file_name, 'rb'))
            Config.tb_lba_offset = get_bid_offset_map(tables_bid_range)
            return tables_bid_range
    except FileNotFoundError as e:
        print(f'file {file_name} not found, create the tb_bid_range file.')

    tables_bid_range = {}
    global_max_bid = 0
    for table in Config.table_list:
        min_bid, max_bid = db_helper.get_block_index_range(table)
        tables_bid_range[table] = (min_bid, max_bid)
        global_max_bid = max(global_max_bid, max_bid)
    Config.adr_digit_num = len(str(global_max_bid))
    print(f'Max BlockID has {Config.adr_digit_num} digits')
    Config.tb_lba_offset = get_bid_offset_map(tables_bid_range)
    cPickle.dump(tables_bid_range, open(file_name, 'wb'))
    return tables_bid_range


def get_tables_actual_bid_range(load_from_file=True):
    file_name = f'{Config.db_name}_tb_real_bid_rangeWT{len(Config.table_list)}WB{Config.logical_block_size}.p'
    try:
        if load_from_file:
            return cPickle.load(open(file_name, 'rb'))
    except FileNotFoundError as e:
        print(f'file {file_name} not found, create the tb_real_bid_range file.')

    tables_bid_range = {}
    for table in Config.table_list:
        min_bid, max_bid = db_helper.get_block_actual_index_range(table)
        tables_bid_range[table] = (min_bid, max_bid)
    cPickle.dump(tables_bid_range, open(file_name, 'wb'))
    return tables_bid_range


def adapt_test(config_suffix, method, test_repeat=1, measure_time=False, save_to_file=False):
    Config.tb_bid_range = get_tables_bid_range()
    Config.actual_tb_bid_range = get_tables_actual_bid_range()
    config_suffix_ = config_suffix
    config_suffix = f'{config_suffix}0'

    print('Getting the components')
    table_manager: TableManager = create_table_manager(1, Config.encoding_length, Config.encoding_epoch_no,
                    f'{Config.db_name}table_manager{config_suffix}B{Config.logical_block_size}P{Config.max_partition_size}.p')
    p_, a_ = create_pm_af_for_configs(
        read_par_manager=1, read_aff_matrix=1,
        table_manager=table_manager, suffix=config_suffix
    )
    partition_manager: PartitionManager.PartitionManager = p_
    aff_matrix: AffinityMatrix.AffinityMatrix = a_

    tests = ['adapt_test']
    file_name_ = tests[0]
    res_hit_miss = []
    ns = 0
    
    for batch_i in range(ns,16):
        config_suffix = f'{config_suffix_}{batch_i}'
        if batch_i > 0:
            print(f'Getting the components for batch {batch_i}')
            table_manager: TableManager = create_table_manager(1, Config.encoding_length, Config.encoding_epoch_no,
                            f'{Config.db_name}table_manager{config_suffix}B{Config.logical_block_size}P{Config.max_partition_size}.p')
            partition_manager, aff_matrix = create_pm_af_for_configs(
                read_par_manager=1, read_aff_matrix=1,
                table_manager=table_manager, suffix=config_suffix
            )

        tlim = None
        if Config.db_name == 'tpcds':
            tlim = 50  
        ti = -1
        method_res = {}
        total_maxes = []
        print(f'k = {Config.prefetching_k}')
        for test in tests:
            test = f'{test}{batch_i}'
            file_name = f'{Config.db_name}_{test}WB{Config.logical_block_size}WP{Config.max_partition_size}'
            test_log_lines: List[LogLine] = general_util.get_log_lines(file_name, False)
            if Config.db_name == 'tpcds':
                test_log_lines = test_log_lines[:tlim]

            if method == 'SGDP':
                preds = cPickle.load(open(
                    f'sgdp3_{Config.db_name}_{test}WB{Config.logical_block_size}_preds.p',
                    'rb'))
                preds = model_utils.convert_lba_to_bid(preds)

            if method == 'forecache':
                preds = cPickle.load(open(f"forecache_pred_test{file_name[8]}.p", "rb"))

            if batch_i == ns:
                general_cache = LRUCache(cache_size)
                all_cache = LRUCache(cache_size)
                first_block_retrival = 0
                extent_type = method if method == 'naive' else Config.extend_type
                trace_window = TraditionalModelWindow(
                    size=Config.trace_win_size, trigger_threshold=Config.read_ahead_trigger_threshold,
                    extend_type=extent_type, num_digits=Config.adr_digit_num)

            timestamps = []
            sum_q_exec_time = 0
            sum_pref_time = 0
            pred_time = 0
            test_res = {}
            requested_blocks = set()
            prefetched_blocks = set()

            for tr in range(test_repeat):
                print(f'{test}-{tr}')
                if measure_time: db_helper.clear_cache()
                for i in tqdm(range(len(test_log_lines))):
                    if measure_time: 
                        db_helper.clear_sys_cache()

                        if method == 'Rerun':
                            conn = db_helper.get_db_connection()
                            q_exec_time = db_helper.get_query_execution_time(test_log_lines[i].query.statement, conn)
                            db_helper.clear_sys_cache()

                        conn = db_helper.get_db_connection()
                        q_exec_time = db_helper.get_query_execution_time(test_log_lines[i].query.statement, conn)
                        sum_q_exec_time += q_exec_time

                    # Update affinities
                    h_hit = all_cache.hit_count
                    h_miss = all_cache.miss_count
                    for b_id in test_log_lines[i].query.result_set:
                        requested_blocks.add(b_id)
                        b_pid = table_manager.get_block_pid(b_id)
                        if b_pid is not None:
                            trace_window.add_bid(b_id, b_pid)
                        general_cache.put(b_id, increase_hit=True)
                        all_cache.put(b_id, increase_hit=True)
                        if not aff_matrix.check_block_existence(b_id):
                            first_block_retrival += 1
                        aff_matrix.update_affinities(b_id, test_log_lines[i].query.result_set)
                    
                    q_hit = all_cache.hit_count - h_hit
                    q_miss = all_cache.miss_count - h_miss
                    res_hit_miss.append([q_hit, q_miss])

                    # get this step prediction
                    # block based prefetching
                    if i == len(test_log_lines) - 1:
                        continue
                    if method == 'NP' or method == 'Rerun':
                        continue
                    if method == 'forecache':
                        pref_blocks = preds[i]
                    elif method == 'SGDP':
                        if i > len(preds) - 1:
                            continue
                        pref_blocks = preds[i]
                    else:
                        t1 = time.time()
                        pref_blocks = trace_window.make_prefetch_decision(partition_manager, strategy=method)
                        t2 = time.time()
                        pred_time += (t2 - t1)

                    pref_count = 0
                    pp = []
                    for pred_b in reversed(pref_blocks):
                        if pref_count > Config.prefetching_k * Config.max_partition_size:
                        # and 'linear' not in method:
                            break
                        all_cache.put(pred_b, increase_hit=False, insert_type='p')
                        prefetched_blocks.add(pred_b)
                        if measure_time:
                            pp.append(pred_b)
                        pref_count += 1

                    t1 = time.time()
                    if measure_time: db_helper.insert_blocks_to_cache(pp)
                    t2 = time.time()
                    sum_pref_time += (t2 - t1)

                print(f'\n{sum_q_exec_time}')

                timestamps.append(ActionTimestamp('exec_time_total', 0.0, sum_q_exec_time))
                timestamps.append(ActionTimestamp('pref_time_total', 0.0, sum_pref_time))

                res_dict = {
                    'general_block_cache': general_cache.report_dict(),
                    'combined_block_cache': all_cache.report_dict()
                }

                test = file_name_
                # test = file_name[:9]

                if tr == 0:
                    useless = 0
                    for b_id in prefetched_blocks:
                        if b_id not in requested_blocks:
                            useless += 1
                    res_dict['useless_prefs'] = useless
                    res_dict['total_misses'] = general_cache.get_total_access() - general_cache.hit_count
                    res_dict['eliminated_misses'] = res_dict['total_misses'] - (
                                all_cache.get_total_access() - all_cache.hit_count)
                    for key in res_dict.keys():
                        test_res[f'{test}_{key}'] = res_dict[key]
                        test_res[f'{test}_pred_time'] = pred_time

                if tr == test_repeat - 1:
                    test_res[f'{test}_exec_time'] = sum_q_exec_time / test_repeat
                    test_res[f'{test}_pref_time'] = sum_pref_time / test_repeat

                for key in test_res:
                    method_res[f'{test}_{key}'] = test_res[key]     
                if save_to_file:
                    print('saving')   
                    cPickle.dump(
                        method_res,
                        open(f'{result_base_path}/{Config.db_name}_{method}4GB_{Config.prefetching_k * Config.max_partition_size}_WB{Config.logical_block_size}WP{Config.max_partition_size}.p', 'wb')
                    )
                    # pprint.pprint(method_res)
                else:
                    print(f'{result_base_path}/{Config.db_name}_{method}4GB_{Config.prefetching_k * Config.max_partition_size}_WB{Config.logical_block_size}WP{Config.max_partition_size}:\t',)
                    pprint.pprint(method_res)

    cPickle.dump(res_hit_miss, open(f'{method}res_hit_miss_F{ns}_T{batch_i}K{Config.prefetching_k}{int(cache_size/16500)}GBfreeze.p', 'wb'))


def test_user_other_methods(file_name_, read_from_file, aff_matrix: AffinityMatrix.AffinityMatrix,
                           partition_manager: PartitionManager.PartitionManager,
                           table_manager: TableManager.TableManager, method='naive',
                            test_repeat=1, measure_time=False, save_to_file=True):
    
    file_name = f'{Config.db_name}_{file_name_}WB{Config.logical_block_size}WP{Config.max_partition_size}'

    test_log_lines: List[LogLine] = general_util.get_log_lines(file_name, read_from_file)
    if Config.db_name == 'tpcds':
        test_log_lines = test_log_lines[:50]

    if method == 'SGDP':
        preds = cPickle.load(open(
            f'sgdp3_{Config.db_name}_{file_name_}WB{Config.logical_block_size}_preds.p',
            'rb'))
        preds = model_utils.convert_lba_to_bid(preds)   

    if method == 'forecache':
        preds = cPickle.load(open(f"forecache_pred_test{file_name[8]}.p", "rb"))


    general_cache = LRUCache(cache_size)
    all_cache = LRUCache(cache_size)
    first_block_retrival = 0
    extent_type = method if method == 'naive' else Config.extend_type
    trace_window = TraditionalModelWindow(
        size=Config.trace_win_size, trigger_threshold=Config.read_ahead_trigger_threshold,
        extend_type=extent_type, num_digits=Config.adr_digit_num)

    timestamps = []
    sum_q_exec_time = 0
    sum_pref_time = 0
    pred_time = 0
    test_res = {}
    requested_blocks = set()
    prefetched_blocks = set()

    for tr in range(test_repeat):
        print(f'{file_name_}-{tr}')
        if measure_time: db_helper.clear_cache()
        for i in tqdm(range(len(test_log_lines))):
            if measure_time: 
                db_helper.clear_sys_cache()

                if measure_time and method == 'Rerun':
                    conn = db_helper.get_db_connection()
                    q_exec_time = db_helper.get_query_execution_time(test_log_lines[i].query.statement, conn)
                    db_helper.clear_sys_cache()

                conn = db_helper.get_db_connection()
                q_exec_time = db_helper.get_query_execution_time(test_log_lines[i].query.statement, conn)
                sum_q_exec_time += q_exec_time

            # Update affinities
            for b_id in test_log_lines[i].query.result_set:
                requested_blocks.add(b_id)
                b_pid = table_manager.get_block_pid(b_id)
                if b_pid is not None:
                    trace_window.add_bid(b_id, b_pid)
                general_cache.put(b_id, increase_hit=True)
                all_cache.put(b_id, increase_hit=True)
                if not aff_matrix.check_block_existence(b_id):
                    first_block_retrival += 1
                aff_matrix.update_affinities(b_id, test_log_lines[i].query.result_set)

            # get this step prediction
            # block based prefetching
            if i == len(test_log_lines) - 1:
                continue
            if method == 'NP' or method == 'Rerun':
                continue
            if method == 'forecache':
                pref_blocks = preds[i]
            elif method == 'SGDP':
                if i > len(preds) - 1:
                    continue
                pref_blocks = preds[i]
            else:
                t1 = time.time()
                pref_blocks = trace_window.make_prefetch_decision(partition_manager, strategy=method)
                t2 = time.time()
                pred_time += (t2 - t1)

            pref_count = 0
            pp = []
            for pred_b in reversed(pref_blocks):
                if pref_count > Config.prefetching_k * Config.max_partition_size:
                # and 'linear' not in method:
                    break
                all_cache.put(pred_b, increase_hit=False, insert_type='p')
                prefetched_blocks.add(pred_b)
                if measure_time:
                    pp.append(pred_b)
                pref_count += 1

            t1 = time.time()
            if measure_time: db_helper.insert_blocks_to_cache(pp)
            t2 = time.time()
            sum_pref_time += (t2 - t1)

        print(f'\n{sum_q_exec_time}')

        timestamps.append(ActionTimestamp('exec_time_total', 0.0, sum_q_exec_time))
        timestamps.append(ActionTimestamp('pref_time_total', 0.0, sum_pref_time))
        # print(f'{method} - {file_name}')
        # # for timestamp in timestamps:
        # #     print(timestamp)
        # print(pref_block_cache.report('Block Cache'))
        # print(general_cache.report('General Cache'))
        # print(all_cache.report('All Cache Block'))
        # print('------------------------------------')
        # print(f'first time block retrival: {first_block_retrival}')

        res_dict = {
            'general_block_cache': general_cache.report_dict(),
            'combined_block_cache': all_cache.report_dict()
        }

        test = file_name_

        if tr == 0:
            useless = 0
            for b_id in prefetched_blocks:
                if b_id not in requested_blocks:
                    useless += 1
            res_dict['useless_prefs'] = useless
            res_dict['total_misses'] = general_cache.get_total_access() - general_cache.hit_count
            res_dict['eliminated_misses'] = res_dict['total_misses'] - (
                        all_cache.get_total_access() - all_cache.hit_count)
            for key in res_dict.keys():
                test_res[f'{test}_{key}'] = res_dict[key]
                test_res[f'{test}_pred_time'] = pred_time

        if tr == test_repeat - 1:
            test_res[f'{test}_exec_time'] = sum_q_exec_time / test_repeat
            test_res[f'{test}_pref_time'] = sum_pref_time / test_repeat

    return test_res


def other_methods_run_main(methods: List[str], save_to_file=True, result_base_path='./', test_repeat=1,
                measure_time=False):
    Config.tb_bid_range = get_tables_bid_range()
    Config.actual_tb_bid_range = get_tables_actual_bid_range()
    read_tb_manager = 1
    read_par_manager = 1
    read_aff_matrix = 1
    b_level = 0

    table_manager: TableManager = create_table_manager(
        read_tb_manager, Config.encoding_length, Config.encoding_epoch_no,
        # file_path = 'sdss_1_navi_table_manager.p' #NAVI
    )
    partition_manager, aff_matrix = create_par_manager_and_aff_matrix(
        read_par_manager=read_par_manager, read_aff_matrix=read_aff_matrix, table_manager=table_manager)

    for method in methods:
        print(f'method {method}')
        method_res = {}
        splited_method = method.split('_')
        if len(splited_method) > 1:
            Config.extend_type = splited_method[1]

        for test in tests:
            print(f'\t{test}')
            res = test_user_other_methods(
                # f'{test}WB{Config.logical_block_size}WP{Config.max_partition_size}', False, aff_matrix, #NAVI
                test, False, aff_matrix,
                partition_manager, table_manager, method=splited_method[0], test_repeat=test_repeat,
                measure_time=measure_time, save_to_file=save_to_file)
            for key in res:
                method_res[f'{test}_{key}'] = res[key]

        ending = '_timed' if measure_time else ''
        if save_to_file:
            print('saving')                
            cPickle.dump(
                method_res,
                open(f'{result_base_path}/{Config.db_name}_{method}_{Config.prefetching_k * Config.max_partition_size}_WB{Config.logical_block_size}WP{Config.max_partition_size}{ending}.p', 'wb')
            )
            pprint.pprint(method_res)
            mres = method_res
        else:
            print(f'{result_base_path}/{Config.db_name}_{method}_{Config.prefetching_k * Config.max_partition_size}_WB{Config.logical_block_size}WP{Config.max_partition_size}{ending}:\t',)
            pprint.pprint(method_res)


def main1(result_base_path='./', save_res_to_file=True):
    Config.tb_bid_range = get_tables_bid_range()
    Config.actual_tb_bid_range = get_tables_actual_bid_range()
    read_tb_manager = 1
    read_par_manager = 1
    read_aff_matrix = 1

    table_manager: TableManager = create_table_manager(
        read_tb_manager, Config.encoding_length, Config.encoding_epoch_no, file_path='')
    print(table_manager.tables.keys())
    
    partition_manager, aff_matrix = create_par_manager_and_aff_matrix(
        read_par_manager=read_par_manager, read_aff_matrix=read_aff_matrix, table_manager=table_manager)

    set_block_pid(partition_manager, table_manager) #sdss_1_navi_table_manager.p


def set_block_pid(partition_manager, table_manager, file_name=''):
    if file_name == '':
        file_name = f'{Config.db_name}table_managerB{Config.logical_block_size}P{Config.max_partition_size}.p'
    for partition in partition_manager.partitions.values():
        for block in partition.blocks:
            tb = block.rsplit('_', 1)
            if tb[0] == 'dataconstants':
                continue
            table_manager.tables[tb[0]].blocks[block].set_partitionid(partition.partition_id)
    cPickle.dump(table_manager, open(file_name, 'wb'))


if __name__ == '__main__':
    result_base_path = os.path.join(Config.base_dir, 'Results')
    # main1(result_base_path=result_base_path, save_res_to_file=True)
    methods = [
            # 'NP',
            # 'Rerun',
            'random_table', 
            'SGDP',
            'naive', 
            'linear_table',
        ]

    # tests = ['naviTest3'] #NAVI
    tests = ['test1_1gen', 'test1_2', 'test2_1', 'test2_2', 'test3_1', 'test3_2', 'testMixed2', 'testMixed9'] 

    cache_size = 66000
    other_methods_run_main(
        methods, save_to_file=True, result_base_path=result_base_path, test_repeat=1, measure_time=True)
    
    # adapt_test('_adapt', methods[0])
    print('done')


