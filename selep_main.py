import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
from tensorflow import keras

import pandas as pd
import numpy as np
import csv
csv.field_size_limit(100 * 1024 * 1024) 
import _pickle as cPickle
import time
from tqdm import tqdm
import pprint as pp
from typing import List
from sklearn.model_selection import train_test_split

from Backend.Database import helpers as db_helper
from Backend.Database.LRUCache import LRUCache
from Backend.Util import PartitionManager, AffinityMatrix
from Backend.Util.BackendUtilFunctions import get_encoded_block_aggregation, get_binary_result_partitions
from Backend.Util.LogComp import LogLine
from Backend.Util.TableManager import TableManager
from Configuration.config import Config
from Utils.utilFunction import get_result_partition_based_encoding, get_log_lines, aggregate_result_dicts
from main import create_table_manager, create_par_manager_and_aff_matrix, get_tables_bid_range, \
    get_tables_actual_bid_range, create_pm_af_for_configs
from Backend.Models.LSTM import *


cache_size = 0
read_tb_manager = 1
read_par_manager = 1
read_aff_matrix = 1
b_level = False
base_model_file_dir = "./SavedFiles/Models/"
result_base_path = './Results'


def calculate_aggregated_result_seqs(df, enc_seq, binary_output_seq, num_partitions):
    encoded_seqs = []
    binary_output_seqs = []
    for i in range(1, len(df)):
        if df.loc[i, 'encodedResult'] == '-':
            continue
        if df.loc[i, 'clientIP'] == df.loc[i - 1, 'clientIP']:
            enc_seq.append(get_encoded_block_aggregation(df.loc[i, 'encodedResult'], b_level))
            binary_output_seq.append(
                get_binary_result_partitions(df.loc[i, 'resultPartitions'], num_partitions))
        elif len(enc_seq) > 0:
            encoded_seqs.append(enc_seq)
            binary_output_seqs.append(binary_output_seq)
            enc_seq = [get_encoded_block_aggregation(df.loc[i, 'encodedResult'], b_level)]
            binary_output_seq = [
                get_binary_result_partitions(df.loc[i, 'resultPartitions'], num_partitions)]
    if len(enc_seq) > 0:
        encoded_seqs.append(enc_seq)
        binary_output_seqs.append(binary_output_seq)
    
    return encoded_seqs, binary_output_seqs


def convert_to_supervised(encoded_seqs, binary_output_seqs, look_back):
    data_x, data_y = [], []
    for j in range(len(encoded_seqs)):
        for i in range(len(encoded_seqs[j]) - look_back):
            data_x.append(np.array(encoded_seqs[j][i:(i + look_back)]))
            data_y.append(np.array(binary_output_seqs[j][i + look_back]))
    data_x = np.array(data_x)
    data_x = data_x.reshape(data_x.shape[0], look_back, data_x.shape[2] * data_x.shape[3])
    return data_x, data_y



def main(model_name, result_file_name, do_train=0, config_suffix='', test_repeat=1, total_repeat=1,
         measure_time=False, do_optimize=0, save_to_file=True):
    
    look_back = Config.look_back
    rows = len(Config.table_list)
    cols = Config.encoding_length
    file_name = f'{Config.db_name}_all_train{config_suffix}WB{Config.logical_block_size}WP{Config.max_partition_size}'
    # file_name = "sdss_1_navi_all_workloadWB8WP68" #NAVI

    if config_suffix == '':
        table_manager: TableManager = create_table_manager(
            read_tb_manager, Config.encoding_length, Config.encoding_epoch_no,
            # file_path='sdss_1_navi_table_manager.p' #NAVI
        )  
        print(table_manager.tables.keys())

        p_, a_ = create_par_manager_and_aff_matrix(
            read_par_manager=read_par_manager, read_aff_matrix=read_aff_matrix,
            table_manager=table_manager
            # , i=str(batch_number)
        )
    else:
        print(f'Getting the components with suffix "{config_suffix}"')
        table_manager: TableManager = create_table_manager(1, Config.encoding_length, Config.encoding_epoch_no,
                        f'{Config.db_name}table_manager{config_suffix}B{Config.logical_block_size}P{Config.max_partition_size}.p')
        p_, a_ = create_pm_af_for_configs(
            read_par_manager=1, read_aff_matrix=1,
            table_manager=table_manager, suffix=config_suffix
        )

    partition_manager: PartitionManager.PartitionManager = p_
    aff_matrix: AffinityMatrix.AffinityMatrix = a_
    num_partitions = partition_manager.get_max_pid()
    model = None
    initial_iter_no = 0

    for iteration in range(initial_iter_no, total_repeat):   
        print(f'iter-{iteration}')
        sep = '\|\|' if Config.db_name != 'tpcds' else '$'
        if do_train:
            if iteration == initial_iter_no:  # Prepare the data only for the first iteration
                print('Preparing the data')
                train_df = pd.read_csv(
                    f'{file_name}.txt',
                    header=0,
                    sep=sep,
                    quoting=csv.QUOTE_ALL,
                    quotechar='"',
                    engine='python'
                )

                # Get result set's list of block/partition encodings 
                train_df['encodedResult'] = '-'
                train_df = get_result_partition_based_encoding(train_df, partition_manager)

                # Aggregate the result encoding and calculate train_df sequence of encoded result set (input)
                # and the binary result set (output)
                enc_seq = [get_encoded_block_aggregation(train_df.loc[0, 'encodedResult'], b_level)]
                binary_output_seq = [
                    get_binary_result_partitions(train_df.loc[0, 'resultPartitions'], num_partitions)]
                encoded_seqs, binary_output_seqs = calculate_aggregated_result_seqs(
                    train_df, enc_seq, binary_output_seq, num_partitions)
                
                # Convert the sequences to actual input/output for the model and make the model input and output
                data_x, data_y = convert_to_supervised(encoded_seqs, binary_output_seqs, look_back)
                print(data_x.shape)
                

            model = create_binary_lstm_model(num_partitions, look_back, rows, cols)
            model.compile(
                # loss=keras.losses.MeanSquaredError(),
                # loss=keras.losses.CategoricalCrossentropy(),
                loss=keras.losses.BinaryCrossentropy(from_logits=False),
                optimizer=keras.optimizers.Adam(),
                metrics=[keras.metrics.MeanAbsoluteError(), 'accuracy'])
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, mode='min', verbose=1)
            model.summary()
            
            # train the model
            data_x_train, data_x_val, data_y_train, data_y_val = train_test_split(data_x, np.array(data_y), test_size=0.10, random_state=42)
            print('Training the model')
            model.fit(data_x_train, data_y_train, epochs=25, validation_data=(data_x_val, data_y_val), verbose=1, callbacks=[early_stopping]
                    #   , batch_size=16
                )

            store_model(model, model_name, base_model_file_dir) 
        else:
            model = load_model(model_name, base_model_file_dir)

        tests = ['test1_1gen', 'test1_2', 'test2_1', 'test2_2', 'test3_1', 'test3_2', 'testMixed2', 'testMixed9']

        tlim = None if Config.db_name != 'tpcds' else 50
        ti = -1
        method_res = {}
        for test in tests:
            timestamps = []
            ti += 1
            file_name = f'{Config.db_name}_{test}{config_suffix}WB{Config.logical_block_size}WP{Config.max_partition_size}'
            # file_name = f'{test}WB{Config.logical_block_size}WP{Config.max_partition_size}' #NAVI
            test_df = pd.read_csv(
                f'{file_name}.txt',
                header=0,
                sep=sep,
                quoting=csv.QUOTE_ALL,
                quotechar='"',
                engine='python',
            )
            if tlim: test_df = test_df.iloc[:tlim]
            # test_df = get_proper_data(test_df, table_manager)
 
            test_df['clientIP'] = '127.0.0.1'
            test_df['encodedResult'] = '-'
            t1 = time.time()
            test_df = get_result_partition_based_encoding(test_df, partition_manager)
            t2 = time.time()
            timestamps.append(ActionTimestamp('test_data_preparation_time1', t1, t2))

            # Calculate test_df sequence of encoded result set and the binary result set
            enc_seq = []
            binary_output_seq = []
            t1 = time.time()
            for j in range(0, Config.look_back - 1):
                enc_seq.append(np.zeros((len(Config.table_list), Config.encoding_length)))
                binary_output_seq.append([])
            enc_seq.append(get_encoded_block_aggregation(test_df.loc[0, 'encodedResult'], b_level))
            binary_output_seq.append(get_binary_result_partitions(test_df.loc[0, 'resultPartitions'], num_partitions))
            encoded_seqs, binary_output_seqs = calculate_aggregated_result_seqs(test_df, enc_seq, binary_output_seq, num_partitions)

            test_data_x, test_data_y = convert_to_supervised(encoded_seqs, binary_output_seqs, look_back)
            t2 = time.time()
            output = model.predict(test_data_x, verbose=1)
            t3 = time.time()
            timestamps.append(ActionTimestamp('test_data_preparation_time2', t1, t2))
            timestamps.append(ActionTimestamp('prediction_time', t2, t3))
            pred_time = t3 - t2
            print(output.shape)

            # Evaluate the test result
            test_log_lines: List[LogLine] = get_log_lines(file_name, read_from_file=True)
            if tlim: test_log_lines = test_log_lines[:tlim]
            par_general_cache = LRUCache(cache_size / Config.max_partition_size)
            general_cache = LRUCache(cache_size)
            all_cache = LRUCache(cache_size)
            sum_q_exec_time = 0
            sum_pref_time = 0
            requested_blocks = set()
            prefetched_blocks = set()

            requested_pars = set()
            prefetched_pars = set()

            for tr in range(test_repeat):
                print(f'{test}-{tr}')
                if measure_time: db_helper.clear_cache()
                for i in tqdm(range(len(test_log_lines))):
                    if measure_time: 
                        db_helper.clear_sys_cache()
                        conn = db_helper.get_db_connection()
                        q_exec_time = db_helper.get_query_execution_time(test_log_lines[i].query.statement, conn)
                        sum_q_exec_time += q_exec_time

                    if tr == 0:
                        for b_id in test_log_lines[i].query.result_set:
                            requested_blocks.add(b_id)
                            general_cache.put(b_id, increase_hit=True)
                            all_cache.put(b_id, increase_hit=True)
                            aff_matrix.update_affinities(b_id, test_log_lines[i].query.result_set)
                        requested_partitions = test_log_lines[i].query.result_partitions
                        for p_id in requested_partitions:
                            requested_pars.add(p_id)
                            par_general_cache.put(p_id, increase_hit=True)

                        partition_manager.update_partition_graph(requested_partitions)

                    if i == len(test_log_lines) - 1:
                        continue

                    prediction = output[i]
                    top_k_indices = np.argsort(prediction)[-Config.prefetching_k:]
                    if do_optimize:
                        partitions_to_prefetch = top_k_indices[prediction[top_k_indices] > 0.1]
                        if len(partitions_to_prefetch) < 3 and Config.prefetching_k >= 3:
                            partitions_to_prefetch = top_k_indices[-3:]
                    else:
                        partitions_to_prefetch = top_k_indices

                    prefetched_partitions = ['p' + str(partition + 1) for partition in partitions_to_prefetch]

                    blocks_to_insert = []
                    for pred_par in prefetched_partitions:
                        p_block_list = partition_manager.partitions.get(pred_par).blocks
                        if tr == 0:
                            partition_manager.put_partition_in_cache(pred_par, all_cache)
                            for b_id in p_block_list:
                                prefetched_blocks.add(b_id)
                            prefetched_pars.add(pred_par)
                        if measure_time:
                            for b_id in p_block_list:
                                blocks_to_insert.append(b_id)

                    if measure_time:
                        # print(f'partition list to be prefetched: {prefetched_partitions}')
                        pref_time = db_helper.insert_blocks_to_cache(blocks_to_insert)
                        sum_pref_time += pref_time

                timestamps.append(ActionTimestamp('exec_time_total', 0.0, sum_q_exec_time))
                timestamps.append(ActionTimestamp('pref_time_total', 0.0, sum_pref_time))
                if total_repeat == 1:
                    # # for timestamp in timestamps:
                    # #     print(timestamp)
                    print('------------------------------------')
                    print(sum_pref_time)
                    print(sum_q_exec_time)
                    print(general_cache.report('General Cache'))
                    print(all_cache.report('All Cache'))
                    print('------------------------------------')

                res_dict = {
                    'general_block_cache': general_cache.report_dict(),
                    'combined_block_cache': all_cache.report_dict()
                }

                if tr == 0:
                    useless = 0
                    for b_id in prefetched_blocks:
                        if b_id not in requested_blocks:
                            useless += 1

                    par_useless = 0
                    for pid in prefetched_pars:
                        if pid not in requested_pars:
                            par_useless += 1

                    res_dict['useless_prefs'] = int(useless)
                    res_dict['useless_par_prefs'] = int(par_useless)
                    res_dict['total_misses'] = general_cache.get_total_access() - general_cache.hit_count
                    res_dict['eliminated_misses'] = res_dict['total_misses'] - (
                                all_cache.get_total_access() - all_cache.hit_count)

                    for key in res_dict.keys():
                        method_res[f'{test}_{key}'] = res_dict[key]
                        method_res[f'{test}_pred_time'] = pred_time

                if tr == test_repeat - 1:
                    method_res[f'{test}_exec_time'] = sum_q_exec_time / test_repeat
                    method_res[f'{test}_pref_time'] = sum_pref_time / test_repeat

        # print(method_res)
        if save_to_file:
            print('saving')        
            ending = '_timed' if measure_time else ''
            full_res_file_name = f'{result_base_path}/{Config.db_name}_{result_file_name}_{Config.prefetching_k * Config.max_partition_size}_WB{Config.logical_block_size}WP{Config.max_partition_size}{ending}.p'
            if iteration > 0:
                avg_k = 1 if iteration < total_repeat - 1 else total_repeat
                try:
                    res_history = cPickle.load(open(full_res_file_name, 'rb'))
                    method_res = aggregate_result_dicts(res_history, method_res, avg_k)
                except FileNotFoundError:
                    print(f'!!! File {full_res_file_name} does not exist !!!')

            cPickle.dump(method_res, open(full_res_file_name, 'wb'))

    print('done')


def adaptivity_test_main(model_name, result_file_name, do_train=0, config_suffix='', test_repeat=1, total_repeat=1,
         measure_time=False, do_optimize=0, save_to_file=True):
        
    look_back = Config.look_back
    rows = len(Config.table_list)
    cols = Config.encoding_length
    file_name = f'{Config.db_name}_all_train{config_suffix}WB{Config.logical_block_size}WP{Config.max_partition_size}'
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

    num_partitions = partition_manager.get_max_pid()
    model = None
    initial_iter_no = 0

    for iteration in range(initial_iter_no, total_repeat):   
        print(f'iter-{iteration}')
        sep = '\|\|' if Config.db_name != 'tpcds' else '$'
        if do_train:
            if iteration == initial_iter_no:  #Prepare the data only for the first iteration
                print('Preparing the data')
                train_df = pd.read_csv(
                    f'{file_name}.txt',
                    header=0,
                    sep=sep,
                    quoting=csv.QUOTE_ALL,
                    quotechar='"',
                    engine='python'
                )
                if 'ClientIP' in train_df.columns:
                    train_df.rename(columns={'ClientIP': 'clientIP'}, inplace=True)            

                train_df['encodedResult'] = '-'
                train_df = get_result_partition_based_encoding(train_df, partition_manager)

                enc_seq = [get_encoded_block_aggregation(train_df.loc[0, 'encodedResult'], b_level)]
                binary_output_seq = [
                    get_binary_result_partitions(train_df.loc[0, 'resultPartitions'], num_partitions)]
                encoded_seqs, binary_output_seqs = calculate_aggregated_result_seqs(
                    train_df, enc_seq, binary_output_seq, num_partitions)
                
                data_x, data_y = convert_to_supervised(encoded_seqs, binary_output_seqs, look_back)
                print(data_x.shape)
                
            model = create_binary_lstm_model(num_partitions, look_back, rows, cols)
            model.compile(
                # loss=keras.losses.MeanSquaredError(),
                # loss=keras.losses.CategoricalCrossentropy(),
                loss=keras.losses.BinaryCrossentropy(from_logits=False),
                optimizer=keras.optimizers.Adam(),
                metrics=[keras.metrics.MeanAbsoluteError(), 'accuracy'])
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, mode='min', verbose=1)
            model.summary()
            
            # train the model
            data_x_train, data_x_val, data_y_train, data_y_val = train_test_split(data_x, np.array(data_y), test_size=0.10, random_state=42)
            print('Training the model')
            model.fit(data_x_train, data_y_train, epochs=25, validation_data=(data_x_val, data_y_val), verbose=1, callbacks=[early_stopping]
                    #   , batch_size=16
                )

            store_model(model, model_name, base_model_file_dir)  # binary_lstm_premodel, embed_binary, cross_embed_binary
        else:
            model = load_model(model_name, base_model_file_dir)
            model.compile(
                loss=keras.losses.BinaryCrossentropy(from_logits=False),
                optimizer=keras.optimizers.Adam(),
                metrics=[keras.metrics.MeanAbsoluteError(), 'accuracy']
            )
            print(model.layers)
            for layer in model.layers:
                if layer.name != 'dense_2':
                    layer.trainable = False
            model.summary()

        tests = ['adapt_test']
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
  
            ti = -1
            method_res = {}
            print(f'k = {Config.prefetching_k}')
            for test in tests:
                test = f'{test}{batch_i}'
                timestamps = []
                ti += 1
                file_name = f'{Config.db_name}_{test}WB{Config.logical_block_size}WP{Config.max_partition_size}'
                test_df = pd.read_csv(
                    f'{file_name}.txt',
                    header=0,
                    sep=sep,
                    quoting=csv.QUOTE_ALL,
                    quotechar='"',
                    engine='python',
                )
                if 'ClientIP' in test_df.columns:
                    test_df.rename(columns={'ClientIP': 'clientIP'}, inplace=True)  
                test_df['clientIP'] = '127.0.0.1'
                test_df['encodedResult'] = '-'
                t1 = time.time()
                test_df = get_result_partition_based_encoding(test_df, partition_manager)
                t2 = time.time()
                timestamps.append(ActionTimestamp('test_data_preparation_time1', t1, t2))

                enc_seq = []
                binary_output_seq = []
                t1 = time.time()
                for j in range(0, Config.look_back - 1):
                    enc_seq.append(np.zeros((len(Config.table_list), Config.encoding_length)))
                    binary_output_seq.append([])
                enc_seq.append(get_encoded_block_aggregation(test_df.loc[0, 'encodedResult'], b_level))
                binary_output_seq.append(get_binary_result_partitions(test_df.loc[0, 'resultPartitions'], num_partitions))
                encoded_seqs, binary_output_seqs = calculate_aggregated_result_seqs(test_df, enc_seq, binary_output_seq, num_partitions)

                test_data_x, test_data_y = convert_to_supervised(encoded_seqs, binary_output_seqs, look_back)
                t2 = time.time()
                output = model.predict(test_data_x, verbose=1)
                t3 = time.time()
                timestamps.append(ActionTimestamp('test_data_preparation_time2', t1, t2))
                timestamps.append(ActionTimestamp('prediction_time', t2, t3))
                pred_time = t3 - t2
                print(output.shape)

                # Evaluate the test result
                test_log_lines: List[LogLine] = get_log_lines(file_name, False)
                if batch_i == ns:
                    general_cache = LRUCache(cache_size)
                    par_general_cache = LRUCache(cache_size / Config.max_partition_size)
                    all_cache = LRUCache(cache_size)
                sum_q_exec_time = 0
                sum_pref_time = 0
                requested_blocks = set()
                prefetched_blocks = set()

                requested_pars = set()
                prefetched_pars = set()

                for tr in range(test_repeat):
                    print(f'{test}-{tr}')
                    if measure_time: db_helper.clear_cache()
                    for i in tqdm(range(len(test_log_lines))):
                        if measure_time: 
                            db_helper.clear_sys_cache()
                            conn = db_helper.get_db_connection()
                            q_exec_time = db_helper.get_query_execution_time(test_log_lines[i].query.statement, conn)
                            sum_q_exec_time += q_exec_time

                        if tr == 0:
                            h_hit = all_cache.hit_count
                            h_miss = all_cache.miss_count
                            for b_id in test_log_lines[i].query.result_set:
                                requested_blocks.add(b_id)
                                general_cache.put(b_id, increase_hit=True)
                                all_cache.put(b_id, increase_hit=True)
                                aff_matrix.update_affinities(b_id, test_log_lines[i].query.result_set)
                            requested_partitions = test_log_lines[i].query.result_partitions
                            for p_id in requested_partitions:
                                requested_pars.add(p_id)
                                par_general_cache.put(p_id, increase_hit=True)

                            q_hit = all_cache.hit_count - h_hit
                            q_miss = all_cache.miss_count - h_miss
                            res_hit_miss.append([q_hit, q_miss])
                            partition_manager.update_partition_graph(requested_partitions)

                        if i == len(test_log_lines) - 1:
                            continue

                        prediction = output[i]

                        top_k_indices = np.argsort(prediction)[-Config.prefetching_k:]
                        if do_optimize:
                            partitions_to_prefetch = top_k_indices[prediction[top_k_indices] > 0.1]
                            if len(partitions_to_prefetch) < 3 and Config.prefetching_k >= 3:
                                partitions_to_prefetch = top_k_indices[-3:]
                        else:
                            partitions_to_prefetch = top_k_indices

                        prefetched_partitions = ['p' + str(partition + 1) for partition in partitions_to_prefetch]

                        blocks_to_insert = []
                        for pred_par in prefetched_partitions:
                            p_block_list = partition_manager.partitions.get(pred_par).blocks
                            if tr == 0:
                                partition_manager.put_partition_in_cache(pred_par, all_cache)
                                for b_id in p_block_list:
                                    prefetched_blocks.add(b_id)
                                prefetched_pars.add(pred_par)
                            if measure_time:
                                for b_id in p_block_list:
                                    blocks_to_insert.append(b_id)

                        t1 = time.time()
                        if measure_time:
                            pref_time = db_helper.insert_blocks_to_cache(blocks_to_insert)
                            sum_pref_time += pref_time
                        t2 = time.time()

                    print('Training the model')
                    
                    model.fit(test_data_x, np.array(test_data_y), epochs=20, verbose=1
                            #   , batch_size=16
                        )
                    timestamps.append(ActionTimestamp('exec_time_total', 0.0, sum_q_exec_time))
                    timestamps.append(ActionTimestamp('pref_time_total', 0.0, sum_pref_time))
                    if total_repeat == 1:
                        print('------------------------------------')
                        print(sum_pref_time)
                        print(sum_q_exec_time)
                        print(general_cache.report('General Cache'))
                        print(all_cache.report('All Cache'))
                        print('------------------------------------')

                    res_dict = {
                        'general_block_cache': general_cache.report_dict(),
                        'combined_block_cache': all_cache.report_dict()
                    }

                    if tr == 0:
                        useless = 0
                        for b_id in prefetched_blocks:
                            if b_id not in requested_blocks:
                                useless += 1

                        par_useless = 0
                        for pid in prefetched_pars:
                            if pid not in requested_pars:
                                par_useless += 1

                        res_dict['useless_prefs'] = int(useless / 3)
                        res_dict['useless_par_prefs'] = int(par_useless)
                        res_dict['total_misses'] = general_cache.get_total_access() - general_cache.hit_count
                        res_dict['eliminated_misses'] = res_dict['total_misses'] - (
                                    all_cache.get_total_access() - all_cache.hit_count)

                        for key in res_dict.keys():
                            method_res[f'{test}_{key}'] = res_dict[key]
                            method_res[f'{test}_pred_time'] = pred_time

                    if tr == test_repeat - 1:
                        method_res[f'{test}_exec_time'] = sum_q_exec_time / test_repeat
                        method_res[f'{test}_pref_time'] = sum_pref_time / test_repeat

            # print(method_res)
            if save_to_file:
                print('saving')        
                ending = '_timed' if measure_time else ''
                full_res_file_name = f'{result_base_path}/{Config.db_name}_{result_file_name}_{Config.prefetching_k * Config.max_partition_size}_WB{Config.logical_block_size}WP{Config.max_partition_size}{ending}.p'
                if iteration > 0:
                    avg_k = 1 if iteration < total_repeat - 1 else total_repeat
                    try:
                        res_history = cPickle.load(open(full_res_file_name, 'rb'))
                        method_res = aggregate_result_dicts(res_history, method_res, avg_k)
                    except FileNotFoundError:
                        print(f'!!! File {full_res_file_name} did not exist !!!')      
                cPickle.dump(method_res, open(full_res_file_name, 'wb'))

        cPickle.dump(res_hit_miss, open(f'res_hit_miss_F{ns}_T{batch_i}K{Config.prefetching_k}{int(cache_size/16500)}GBfreeze.p', 'wb'))
            

    print('done')


def test_different_k(k_options, model_name, result_file_name):
    for fk in k_options:
        do_train = 0
        Config.prefetching_k = fk
        print(f'Working with k = {Config.prefetching_k}')
        main(model_name, result_file_name, do_train=do_train, test_repeat=1, measure_time=False,
             do_optimize=0, save_to_file=True)


def test_different_lookback(l_options, model_name_, result_file_name, prefetching_k=0):
    if prefetching_k == 0:
        prefetching_k = Config.prefetching_k
    else:
        Config.prefetching_k = prefetching_k
    for fk in l_options:
        do_train = 0
        Config.look_back = fk
        model_name = f'{model_name_}_lookback{fk}'
        if fk == 4:
            do_train = 0
            model_name = model_name_
        print(f'Working with l = {Config.look_back}')
        main(model_name, f'{result_file_name}_lookback{fk}', do_train=do_train, test_repeat=1, measure_time=False,
             do_optimize=0, save_to_file=True)


def test_different_cachesize(c_options, model_name_, result_file_name, prefetching_k=0):
    if prefetching_k == 0:
        prefetching_k = Config.prefetching_k
    else:
        Config.prefetching_k = prefetching_k
    for fk in c_options:
        do_train = 0
        cache_size = fk
        model_name = f'{model_name_}cache{fk}'
        print(f'Working with cache size = {cache_size} ~ {cache_size/16500} GB')
        main(model_name, f'{result_file_name}_lookback{fk}', do_train=do_train, test_repeat=1, measure_time=False,
             do_optimize=0, save_to_file=True)
        

def test_partitioning_configs(model_name, result_file_name):
    options_dict = {
        'maxWinSize': [500, 1000, 2500, 4000, 5000, 6000, 7500, 8000, 10000, 12500, 15000, 17500, 20000],
        'weightResetVal': [0.1, 0.25, 0.5, 0.75, 0.9, 1],
        'k': [1, 5, 10, 25, 50],
        'maxLoad': [0.05, 0.25, 0.5, 1, 2.5, 5, 7.5, 10, 12.5, 15],
        'empPar': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
        'fillPortion': [0.5, 0.75, 0.85, 0.9, 0.95]
    }

    parameter_values_ = {
        'maxWinSize': '1000',
        'weightResetVal': '0.25',
        'k': '1',
        'empPar': '0.05',
        'fillPortion': '0.85',
        'maxLoad': '1',
        'maxPartitionSize': '128'
    }

    done_configs = []

    for param, options in options_dict.items():
        print(f'working on {param}')
        for option in options: 
            print(f'\t{param}-{option}')
            parameter_values = parameter_values_.copy()
            parameter_values[param] = str(option)

            if list(parameter_values.items()) in done_configs:
                print("\t!!duplicate config")
                continue

            suffix = '_'.join(parameter_values.values())
            suffix = suffix.replace('0.', '')
            main(f'{model_name}_config{suffix}', f'{result_file_name}_config{suffix}', config_suffix=suffix, do_train=0, test_repeat=1, measure_time=False,
                do_optimize=0, save_to_file=True, total_repeat=1)


if __name__ == '__main__':
    Config.tb_bid_range = get_tables_bid_range()
    Config.actual_tb_bid_range = get_tables_actual_bid_range()
    cache_size = 66000 # 33000 is 2 GB

    # NAVI
    model_names = ['binary_cross_entropy2']
    result_file_names = ['binary_lstm2', 'binary_navi']

    # test_partitioning_configs(model_name=model_names[5], result_file_name=result_file_names[4])
    # test_different_k(list(range(1, 45, 3)), model_name=model_names[1], result_file_name=f'{result_file_names[1]}4GB')
    # test_different_lookback(list(range(2, 11)), model_name_=model_names[1], result_file_name=f'{result_file_names[0]}6GB', prefetching_k=20)

    main(model_names[0], f'{result_file_names[0]}', do_train=1, test_repeat=1, measure_time=False,
            do_optimize=0, save_to_file=True)


    # adaptivity_test_main(f'{model_names[1]}_adapt', f'{result_file_names[1]}_adapt', config_suffix='_adapt', do_train=0, test_repeat=1, measure_time=False,
    #     do_optimize=0, save_to_file=False, total_repeat=1)
