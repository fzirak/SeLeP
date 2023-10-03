
import os


def read_table_list_from_file(file_path):
    with open(file_path) as file:
        lines = [line.rstrip() for line in file]
    return lines


def read_table_lookups(table_lookup_file_path):
    lines = read_table_list_from_file(table_lookup_file_path)
    lookup = {}
    for i in range(1, len(lines)+1):
        lookup[lines[i-1]] = i
        lookup[i] = lines[i-1]
    return lines, lookup


class Config:
    is_on_server = True
    """ Database Config"""
    
    # db_name = 'tpcds'
    db_name = 'sdss_1'
    db_user = 'user'
    db_password = 'pass'

    db_host = '127.0.0.1'
    db_port = '5432'

    """ model config """
    encoding_length = 32
    encoding_epoch_no = 100
    do_normalisation = False
    look_back = 4

    """ system config """
    block_level_query_encoding = False
    prefetching_k = 42
    prefetching_augment_k = 0
    max_partition_size = 128
    logical_block_size = 8
    trace_win_size = 256 #10000 or 256
    read_ahead_trigger_threshold = 13
    extend_size = max_partition_size
    extend_type = 'table'
    adr_digit_num = 5


    """ Config files """    
    base_dir = "./"
    
    table_lookup_file_path = os.path.join(base_dir, f"Data/{db_name}_tableLookUp.txt")
    pca_exclude_file_path = os.path.join(base_dir, 'Data/pcaExclude.txt')

    table_list, table_lookup = read_table_lookups(table_lookup_file_path)
    tb_bid_range = {}
    actual_tb_bid_range = {}
    tb_lba_offset = {}
    pca_exclude_tables = read_table_list_from_file(pca_exclude_file_path)

    def __init__(self):
        pass

def alter_config(dbname='sdss_1', max_par_size=64, tb_lookup_fp='/navi_tableLookUp.txt'):
    Config.db_name = dbname
    Config.max_partition_size = max_par_size
    Config.table_lookup_file_path = tb_lookup_fp
