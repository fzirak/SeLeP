import random
import numpy as np
from keras.metrics import Accuracy, Precision, Recall, MeanAbsoluteError
from keras.models import model_from_json
from Configuration.config import Config
from collections import OrderedDict
import pandas as pd
from tqdm import tqdm

from Backend.Util.TableManager import TableManager


def get_train_and_test_sequences(data, train_seqs_num=5, test_seqs_num=2):
    for s in data:
        print(len(s))
    print('------------------')
    test_seq = []
    for i in range(test_seqs_num):
        # for i in [6, 10, 12]:
        s = data.pop(random.randrange(len(data)))
        # s = seqs.pop(i)
        test_seq.append(s)

    for i in range(len(data) - train_seqs_num):
        data.pop(random.randrange(len(data)))
    train_seq = data

    return train_seq, test_seq


def get_mean_and_std(encoded_seqs):
    flat_list = np.array([item for sublist in encoded_seqs for item in sublist])
    mean_seq, std_seq = [], []
    mean_seq = np.mean(flat_list, axis=0)
    std_seq = np.std(flat_list, axis=0)
    return mean_seq, std_seq


def remove_duplicates_preserve_order(lst):
    return list(OrderedDict.fromkeys(lst))


def validate_bidx(b_idx, tb):
    min_bid, max_bid = Config.tb_bid_range[tb]
    # print(f'{tb} ({min_bid}, {max_bid}) and {b_idx}')
    return min_bid <= int(b_idx) <= max_bid


def convert_adr_to_bid(preds):
    invalid_count = 0
    bid_preds = []
    max_pred = Config.max_partition_size * Config.prefetching_k
    n = 1
    for pred in preds:
        pred = pred[:min(len(pred), max_pred)]
        pred = remove_duplicates_preserve_order(pred)
        bid_pred = []
        for adr in pred:
            if adr < 100000:
                invalid_count += 1
                # bid_pred.append(f'cacheFillTb1_{n}')
                n += 1
                continue
            b_idx = adr % 100000
            tb_idx = adr // 100000
            if tb_idx > len(Config.table_list):
                continue
            tb = Config.table_lookup[tb_idx]
            if validate_bidx(b_idx, tb):
                bid_pred.append(f'{tb}_{b_idx}')
            else:
                invalid_count += 1
        bid_preds.append(bid_pred)

    # zero_count = 0
    # for pred in bid_preds:
    #     if len(pred) == 0:
    #         zero_count += 1
    # print(zero_count)
    # print(invalid_count)
    return bid_preds


def get_LBA(bid):
    splited_bid = bid.rsplit('_', 1)
    tb = splited_bid[0]
    lba = Config.tb_lba_offset[tb] + int(splited_bid[1])
    return lba


def get_bid(sorted_offset_map, lba):
    tb_name = ''
    for tb in sorted_offset_map:
        if lba < sorted_offset_map[tb]:
            continue
        tb_name = tb
        break
    if tb_name == '':
        return 
    bid = f'{tb_name}_{lba - sorted_offset_map[tb]}'
    return bid


def convert_lba_to_bid(preds):
    invalid_count = 0
    none_count = 0
    bid_preds = []
    max_pred = Config.max_partition_size * Config.prefetching_k
    sorted_offset_map = {k: v for k, v in sorted(Config.tb_lba_offset.items(), key=lambda item: -item[1])}
    for pred in preds:
        pred = pred[:min(len(pred), max_pred)]
        pred = remove_duplicates_preserve_order(pred)
        bid_pred = []
        for lba in pred:
            bid = get_bid(sorted_offset_map, lba)

            if bid == None:
                none_countn += 1
                continue

            splited_bid = bid.rsplit('_', 1)
            if validate_bidx(splited_bid[1], splited_bid[0]):
                bid_pred.append(bid)
            else:
                invalid_count += 1

        bid_preds.append(bid_pred)
    #     print(f'nones {none_count}')
    #     print(f'invalids {invalid_count}')
    # print(f'nones {none_count}')
    # print(f'invalids {invalid_count}')
    return bid_preds


def evaluate_output(output, test_y):
    accuracy_metric = Accuracy()
    precision_metric = Precision()
    recall_metric = Recall()
    mae_metric = MeanAbsoluteError()

    accuracy_metric.update_state(test_y, output)
    precision_metric.update_state(test_y, output)
    recall_metric.update_state(test_y, output)
    mae_metric.update_state(test_y, output)

    accuracy = accuracy_metric.result().numpy()
    precision = precision_metric.result().numpy()
    recall = recall_metric.result().numpy()
    mae = mae_metric.result().numpy()
    print(' accuracy:', accuracy)
    print(' precision:', precision)
    print(' recall:', recall)
    print(' MAE:', mae)


def get_proper_data(df: pd.DataFrame, table_manager: TableManager) -> pd.DataFrame:
    res_df = []
    df['resultPartitions'] = ''

    for _, row in tqdm(df.iterrows(), total=len(df)):
        new_row = row.copy()

        bids = row['resultBlock'].replace('[', '').replace(']', '').replace(' ', '').split(',')
        logical_bids = {f"{bid.rsplit('_', 1)[0]}_{int(bid.rsplit('_', 1)[1]) // Config.logical_block_size}"
                        for bid in bids if is_bid_valid(bid)}

        new_row['resultBlock'] = list(logical_bids)

        partitions = {table_manager.get_block_pid(lbid) for lbid in logical_bids}
        new_row['resultPartitions'] = list(partitions)

        res_df.append(new_row)

    res = pd.DataFrame(res_df)
    return res
    

def is_bid_valid(bid):
    keywords = ['null']
    for keyword in keywords:
        if keyword in bid:
            return False
    return True
