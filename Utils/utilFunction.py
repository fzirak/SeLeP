from typing import List, Dict

import numpy as np
import pandas as pd
import csv
from numpy import array
from tensorflow import keras
import math
import _pickle as cPickle

from Backend.Util.AffinityMatrix import AffinityMatrix
from Backend.Util.BackendUtilFunctions import calculate_block_matrix_encoding, get_encoded_block_aggregation
from Backend.Util.LogComp import LogLine, QueryInfo
from Backend.Util.PartitionManager import PartitionManager, Partition
from Configuration.config import Config


def my_to_supervised(seqs, look_back, n_out=1):
    data_x, data_y = [], []
    for seq in seqs:
        # print('new seq')/
        for i in range(len(seq) - look_back):
            a = np.array(seq[i:(i + look_back)])
            data_x.append(array(a))
            data_y.append(array([seq[i + look_back]]))

    return array(data_x), array(data_y)


def my_one_hot_encode(sequence, max_length=100):
    encoding = list()
    for l in sequence:
        print('working on l')
        e = list()
        for value in l:
            vector = [0 for _ in range(max_length)]
            for i in range(len(value)):
                if value[i] == '1':
                    vector[i] = 1
            e.append(vector)
        encoding.append(e)
    return encoding


def decimal_padding(n_type, sequence, max_length=100):
    encoding = list()
    for l in sequence:
        # print('working on l')
        e = list()
        for value in l:
            vector = [0 for _ in range(max_length)]
            for i in range(len(value)):
                vector[i] = int(value[i])
            if n_type == 'digit':
                e.append(vector)
            elif n_type == 'word':
                e.append([int(''.join(str(i) for i in vector))])
        encoding.append(e)
    return encoding
    # return sequence


def get_max_length(seqs):
    max_seq_len = max(max([len(s) for s in seq]) for seq in seqs)
    return max_seq_len


def read_seqs(filepath):
    with open(filepath) as file:
        lines = [line.rstrip() for line in file]
    seqs = []
    for line in lines:
        temp = line.replace(']', '').replace('[', '').replace("'", '')
        seqs.append(temp.replace(' ', '').split(','))
    return seqs


def get_mean_and_std(encoded_seqs):
    flat_list = np.array([item for sublist in encoded_seqs for item in sublist])
    mean_seq, std_seq = [], []
    mean_seq = np.mean(flat_list, axis=0)
    std_seq = np.std(flat_list, axis=0)
    return mean_seq, std_seq


def normaliser(n_type, encoded_seqs, mean_seq, std_seq):
    if n_type == 'digit':
        return digit_based_normaliser(encoded_seqs, mean_seq, std_seq)
    elif n_type == 'word':
        return word_based_normaliser(encoded_seqs, mean_seq, std_seq)


def word_based_normaliser(encoded_seqs, mean_seq, std_seq):
    normalised_seqs = []
    for seq in encoded_seqs:
        print('the size of a seq in seqs is ', len(seq))
        normalised_s = []
        for s in seq:
            norm_s = (s - mean_seq) / std_seq
            norm_s[np.isnan(norm_s)] = 0
            normalised_s.append(norm_s)
        normalised_seqs.append(normalised_s)
    return normalised_seqs


def digit_based_normaliser(encoded_seqs, mean_seq, std_seq):
    normalised_seqs = []
    for seq in encoded_seqs:
        print('the size of a seq in seqs is ', len(seq))
        normalised_s = []
        for s in seq:
            norm_s = (s - mean_seq) / std_seq
            norm_s[np.isnan(norm_s)] = 0
            normalised_s.append(norm_s)
        normalised_seqs.append(normalised_s)
    return normalised_seqs


def nearest_value(x, range_start, range_end):
    x_rounded = round(x)
    return min(max(x_rounded, range_start), range_end)


def decoder(n_type, seqs, mean_seq, std_seq):
    if n_type == 'digit':
        return digit_based_decoder(seqs, mean_seq, std_seq)
    elif n_type == 'word':
        return word_based_decoder(seqs, mean_seq, std_seq)


def word_based_decoder(seqs, mean_seq, std_seq):
    return


def digit_based_decoder(seqs, mean_seq, std_seq):
    denormalised = []
    for seq in seqs:
        d_s = []
        for s in seq:
            # s = s.numpy()
            ds = (s * std_seq) + mean_seq
            d_str = []
            for d in ds:
                if math.isnan(d):
                    print(seq)
                d_str.append(nearest_value(d, 0, 9))
            d_s.append(d_str)
            break
        denormalised.append(d_s)
    return np.array(denormalised)


def compare_arrays(arr1, arr2, block_size):
    flat1 = arr1.flatten()
    flat2 = arr2.flatten()
    num_matching_digits = np.sum((flat1 == flat2) & (flat1 != 0))

    five_digits1 = flat1.reshape(-1, block_size)
    five_digits2 = flat2.reshape(-1, block_size)
    num_matching_blocks = 0
    for i in range(five_digits1.shape[0]):
        if np.all(five_digits1[i] == 0) or np.all(five_digits2[i] == 0):
            continue
        if np.all(five_digits1[i] == five_digits2[i]):
            num_matching_blocks += 1

    return num_matching_digits, num_matching_blocks


def get_result_block_based_encoding(df, table_manager):
    num_tbs = len(Config.table_list)
    for index, row in df.iterrows():
        res_blocks = row['resultBlock'].replace(']', '').replace('[', '').split(',')
        erbs = []
        for rb in res_blocks:
            rb = rb.strip()
            erb = np.zeros((num_tbs, Config.encoding_length))
            temp = rb.rsplit('_', 1)
            tb_row_enc = table_manager.get_block_encoding(temp[0], rb)
            if tb_row_enc == '':
                continue
            erb[Config.table_lookup.get(temp[0]) - 1] = tb_row_enc
            # np.round(erb, decimals=8)
            erbs.append(erb)
        df.at[index, 'encResultBlock'] = erbs
    return df


def get_res_partitions(res_blocks, aff_matrix: AffinityMatrix, partition_manager: PartitionManager, table_manager) -> \
Dict[str, int]:
    partitions_count = {}
    unallocated_blocks = []
    for block_id in res_blocks:
        p_id = aff_matrix.get_block_partition_id(block_id)
        if p_id == '':
            unallocated_blocks.append(block_id)
            continue
        count = partitions_count.get(p_id)
        partitions_count[p_id] = 1 if count is None else count + 1

    # first try to add the unallocated blocks to the partitions accessed by the query.
    partitions_count = dict(sorted(partitions_count.items(), key=lambda item: item[1]))
    it = iter(partitions_count.items())
    while len(unallocated_blocks) > 0:
        try:
            p_id, _ = next(it)
        except StopIteration:
            break

        partition: Partition = partition_manager.get_partition(p_id)
        while partition.get_size() < Config.max_partition_size:
            if not unallocated_blocks:
                break
            block_mat_enc = calculate_block_matrix_encoding(unallocated_blocks[-1], Config.encoding_length,
                                                            len(Config.table_list), table_manager)
            if block_mat_enc is None:
                unallocated_blocks.pop()
                continue
            partition.add_block_and_update_enc(unallocated_blocks[-1], block_mat_enc)
            aff_matrix.update_block_partition(unallocated_blocks[-1], partition.partition_id)
            unallocated_blocks.pop()
            partitions_count[partition.partition_id] += 1

    # Allocate the rest (if any) to new partitions.
    if unallocated_blocks:
        new_partition = Partition(f'p{partition_manager.get_increasing_index()}')
        partitions_count[new_partition.partition_id] = 0
        for b_id in unallocated_blocks:
            new_partition.add_block(b_id)
            if b_id == '':
                print('empty b_id', res_blocks, '-', unallocated_blocks, '-')
            aff_matrix.update_block_partition(b_id, new_partition.partition_id)
            partitions_count[new_partition.partition_id] += 1
            if new_partition.get_size() >= Config.max_partition_size:
                new_partition.calculate_encoding(table_manager, len(Config.table_list), Config.encoding_length)
                partition_manager.add_partition(new_partition)
                new_partition = Partition(f'p{partition_manager.get_increasing_index()}')
                partitions_count[new_partition.partition_id] = 0
        if new_partition.get_size() > 0 and not partition_manager.has_partition(new_partition.partition_id):
            new_partition.calculate_encoding(table_manager, len(Config.table_list), Config.encoding_length)
            partition_manager.add_partition(new_partition)

        if partitions_count[new_partition.partition_id] == 0:
            partitions_count.pop(new_partition.partition_id)

    return partitions_count


def get_result_partition_based_encoding(df, partition_manager: PartitionManager):
    for index, row in df.iterrows():
        # res_partitions = get_res_partitions(res_blocks, aff_matrix)
        res_partitions = row['resultPartitions'].replace(']', '').replace('[', '').split(',')
        if res_partitions[0] == '':
            continue
        erps = []
        # for partition, count in dict(row['ResultPartitionsC']).items():
        for rp in res_partitions:
            if rp == '':
                print(index)
                print(f'-{res_partitions}-')
                print(row['resultPartitions'])
            rp = rp.strip()
            if rp == 'None':
                continue
            erps.append(partition_manager.get_partition_encoding(rp))
            # erp = partition_manager.get_partition_encoding()
            # for i in range(count):
            #     erps.append(erp)
        df.at[index, 'encodedResult'] = erps
    return df

def generate_batch_encoded_seq(file_name, partition_manager:PartitionManager, batch_num=1):
    df = pd.read_csv(
        "./Data/" + file_name + ".txt",
        header=0,
        sep='\|\|',
        quoting=csv.QUOTE_ALL,
        quotechar='"',
    )
    batch_dfs = []

    batch_size = math.ceil(len(df)/batch_num)
    for batch_number, group in df.groupby(df.index // batch_size):
        partition_manager.get_partitions
        batch_df = group.copy()
        batch_df['encResultBlock'] = '-'
        batch_dfs.append(batch_df)

    result_df = pd.concat(batch_dfs, ignore_index=True)


def generate_mat_encoded_and_par_seq(file_name, table_manager, b_level=True, partition_manager=None):
    df = pd.read_csv(
        "./Data/" + file_name + ".txt",
        header=0,
        sep='\|\|',
        quoting=csv.QUOTE_ALL,
        quotechar='"',
    )
    print(len(df))

    df['encResultBlock'] = '-'
    if b_level:
        df = get_result_block_based_encoding(df, table_manager)
    else:
        df = get_result_partition_based_encoding(df, partition_manager)

    encoded_seqs = []
    enc_seq = [get_encoded_block_aggregation(df.loc[0, 'encResultBlock'], b_level)]

    for i in range(1, len(df)):
        if df.loc[i, 'encResultBlock'] == '-':
            continue
        if df.loc[i, 'clientIP'] == df.loc[i - 1, 'clientIP']:
            enc_seq.append(get_encoded_block_aggregation(df.loc[i, 'encResultBlock'], b_level))
        elif len(enc_seq) > 0:
            encoded_seqs.append(enc_seq)
            enc_seq = [get_encoded_block_aggregation(df.loc[i, 'encResultBlock'], b_level)]

    if len(enc_seq) > 0:
        encoded_seqs.append(enc_seq)

    print(encoded_seqs[0][0])
    print('writing in file')
    cPickle.dump(encoded_seqs, open(file_name + '_seqs_mat.p', 'wb'))
    return encoded_seqs


def get_log_lines(file_name, read_from_file) -> List[LogLine]:
    if read_from_file:
        loglines = cPickle.load(open(f'./SavedFiles/Loglines/{file_name}_loglines.p', 'rb'))
        return loglines

    sep = '\|\|'
    if Config.db_name == 'tpcds':
        sep = '$'

    loglines = []
    df = pd.read_csv(
        "./Data/" + file_name + ".txt",
        header=0,
        sep=sep,
        quoting=csv.QUOTE_ALL,
        quotechar='"',
        engine='python',
    )
    if 'ClientIP' in df.columns:
        df.rename(columns={'ClientIP': 'clientIP'}, inplace=True)
    if 'rows' in df.columns:
        df.rename(columns={'rows': 'row'}, inplace=True)
    for index, row in df.iterrows():
        res_blocks = row['resultBlock'].replace(']', '').replace('[', '').replace(' ', '').split(',')
        res_partitions = row['resultPartitions'].replace(']', '').replace('[', '').replace(' ', '').split(',')
        new_query = QueryInfo(row['statement'], result_set=res_blocks, result_partitions=res_partitions)
        if 'seqNum' in df.columns:
            sn = row['seqNum']
        else:
            sn = index
        new_logline = LogLine(sn, row['theTime'], row['clientIP'], row['row'], new_query)
        loglines.append(new_logline)

    cPickle.dump(loglines, open(f'./SavedFiles/Loglines/{file_name}_loglines.p', 'wb'))
    return loglines


def aggregate_result_dicts(hist_dict, res_dict, k):
    result = {}   
    for key in res_dict:
        try:
            if isinstance(res_dict[key], dict) and isinstance(hist_dict[key], dict):
                result[key] = aggregate_result_dicts(res_dict[key], hist_dict[key], k)
            else:
                result[key] = (float(res_dict[key]) + float(hist_dict[key]))/k
        except Exception as e:
            print(f'!!Skipped key {key}')
            print(e.__repr__())
    return result
