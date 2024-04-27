from Configuration.config import Config
import numpy as np


def get_encoded_block_aggregation(block_encs, b_level=True):
    if b_level:
        block_encs_arr = np.array(block_encs)
        # print(falatten.shape)
        # falatten = np.squeeze(arr, axis=1)
        means = np.mean(block_encs_arr, axis=0)
        # print('means', means)
        stds = np.std(block_encs_arr, axis=0)
        # print('stds', stds)
        return means
        # return np.hstack((means, stds))
    else:
        block_encs_arr = np.array(block_encs)
        return np.mean(block_encs_arr, axis=0)


def get_binary_result_partitions(partitions_str, num_partitions):
    partitions = partitions_str.replace(']', '').replace('[', '').split(',')
    result_array = np.zeros(num_partitions, dtype=int)
    indexes = [int(partition.strip()[1:])-1 for partition in partitions]
    result_array[indexes] = 1
    return result_array


def get_matrix_variables_range(mean_std_matrix):
    """ input shape is (1, #tables, 2*encoding_length) or (#tables, 2*encoding_length)"""
    mean, std = 0, 0
    if len(mean_std_matrix.shape) == 3:
        m = mean_std_matrix.shape[2] // 2
        mean = mean_std_matrix[:, :, :m]
        std = mean_std_matrix[:, :, m:]
    elif len(mean_std_matrix.shape) == 2:
        m = mean_std_matrix.shape[1] // 2
        mean = mean_std_matrix[:, :m]
        std = mean_std_matrix[:, m:]
    else:
        raise Exception(f'{mean_std_matrix.shape} is an invalid shape for get_matrix_variables_range function ')
    range_low = mean - std
    range_high = mean + std
    return range_low, range_high


def calculate_block_matrix_encoding(block, encoding_length, num_of_tbs, table_manager):
    erb = np.zeros((num_of_tbs, encoding_length))
    temp = block.rsplit('_', 1)
    if temp[0] in ['dataconstants', 'null', 'dbviewcols', 'runqa', 'segment']:
        return None
    tb_row_enc = table_manager.get_block_encoding(temp[0], block)
    if tb_row_enc is None:
        return None
    if not Config.table_lookup.get(temp[0]):
        print(temp[0])
    erb[Config.table_lookup.get(temp[0]) - 1] = tb_row_enc
    return erb


def update_encoding_aggregation(enc_aggregation, block_encoding, n):
    means = enc_aggregation[:, :enc_aggregation.shape[1] // 2]
    stds = enc_aggregation[:, enc_aggregation.shape[1] // 2:]

    means = (means * n + block_encoding) / (n + 1)
    stds = np.sqrt(((stds ** 2) * n + (block_encoding - means) ** 2) / (n + 1))
    return np.hstack((means, stds))


def get_zero_rows_idxs(mat, do_round):
    if do_round:
        mat = np.round(mat, decimals=1)
    zero_rows_idxs = []
    for i in range(len(mat[0])):
        if not mat[0][i].any():
            zero_rows_idxs.append(i)
    return zero_rows_idxs
