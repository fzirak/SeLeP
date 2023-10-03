import time

from Configuration.config import Config
import psycopg2
import pandas as pd
import subprocess
from psycopg2.extras import LoggingConnection, LoggingCursor
import logging

from .LRUCache import LRUCache



class MyLoggingCursor(LoggingCursor):
    def execute(self, query, vars=None):
        self.timestamp = time.time()
        return super(MyLoggingCursor, self).execute(query, vars)

    def callproc(self, procname, vars=None):
        self.timestamp = time.time()
        return super(MyLoggingCursor, self).callproc(procname, vars)

# MyLogging Connection:
#   a) calls MyLoggingCursor rather than the default
#   b) adds resulting execution (+ transport) time via filter()
class MyLoggingConnection(LoggingConnection):
    def filter(self, msg, curs):
        return "   %d ms" % int((time.time() - curs.timestamp) * 1000)

    def cursor(self, *args, **kwargs):
        kwargs.setdefault('cursor_factory', MyLoggingCursor)
        return LoggingConnection.cursor(self, *args, **kwargs)

def get_db_connection(autocommit=True):
    # logging.basicConfig(level=logging.DEBUG)
    # logger = logging.getLogger(__name__)

    conn = psycopg2.connect(database=Config.db_name, user=Config.db_user, password=Config.db_password,
                            host=Config.db_host, port=Config.db_port
                            # , connection_factory=MyLoggingConnection
                            )

    # conn.initialize(logger)
    conn.autocommit = autocommit
    return conn


def find_non_numeric_columns(table_name):
    conn = get_db_connection()
    cursor = conn.cursor()
    sql = (f'SELECT column_name,data_type\n'
           f'            FROM information_schema.columns\n'
           f'            WHERE table_name = \'{table_name}\';')
    cursor.execute(sql)
    df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
    # has_img = 'img' in df['column_name'].values
    df = df[df['data_type'].isin(['character varying', 'timestamp without time zone', 'character', 'text', 'bytea'])]
    res = list(df['column_name'].values)
    # if has_img:
    #     res.append('img')
    return res


def get_block_indexes(table_name):
    conn = get_db_connection()
    cursor = conn.cursor()
    sql = ('select (ctid::text::point)[0]::bigint/{} as block_number\n'
           '             from {};').format(Config.logical_block_size, table_name)
    cursor.execute(sql)
    df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
    return df


def insert_block_to_cache(insertion_dict):
    conn = get_db_connection()
    cursor = conn.cursor()
    n = Config.logical_block_size
    # n = 1
    t1 = time.time()
    for tb, range_list in insertion_dict.items():
        if tb == 'dataconstants':
            continue
        for range_ in range_list:
            min_block = max(range_[0] * n, Config.actual_tb_bid_range[tb][0])
            max_block = min(((range_[1]+1) * n)-1, Config.actual_tb_bid_range[tb][1])
            if min_block > max_block:
                # print(f'min gt max {tb}-{range_}')
                continue
            sql = f"SELECT pg_prewarm(" \
                  f"'{tb}'::regclass, " \
                  f"first_block=>{min_block}, " \
                  f"last_block=>{max_block});"
            cursor.execute(sql)
            res = cursor.fetchall()
    t2 = time.time()
    return t2 - t1


def clear_cache():
    subprocess.call(
        'echo {} | sudo -S bash clear_all_caches.sh'.format('pass'),
        shell=True)


def clear_sys_cache():
    subprocess.call(
        'echo {} | sudo -S bash clear_sys_caches.sh'.format('pass'),
        shell=True)


def get_block_index_range(table_name):
    conn = get_db_connection()
    cursor = conn.cursor()
    sql = f'SELECT MIN((ctid::text::point)[0]::bigint)/{Config.logical_block_size} as min_bid, ' \
          f'MAX((ctid::text::point)[0]::bigint)/{Config.logical_block_size} as max_bid\n'\
          f'from {table_name};'
    cursor.execute(sql)
    row = cursor.fetchone()
    if row:
        min_bid = row[0]
        max_bid = row[1]
    else:
        print(f'error while getting block_id range for table{table_name}')
    return min_bid, max_bid

def get_block_actual_index_range(table_name):
    conn = get_db_connection()
    cursor = conn.cursor()
    sql = f'SELECT MIN((ctid::text::point)[0]::bigint) as min_bid, ' \
          f'MAX((ctid::text::point)[0]::bigint) as max_bid\n'\
          f'from {table_name};'
    cursor.execute(sql)
    row = cursor.fetchone()
    if row:
        min_bid = row[0]
        max_bid = row[1]
    else:
        print(f'error while getting block_id range for table{table_name}')
    return min_bid, max_bid


def execute_query(statement):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(statement)
    df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
    return df

def get_query_execution_time(statement, conn):
    # conn = get_db_connection()
    cursor = conn.cursor()
    t1 = time.time()
    cursor.execute(statement)
    df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
    t2 = time.time()
    return t2 - t1


def insert_blocks_to_cache(blocks_to_insert):
    tb_block = {}
    for bid in blocks_to_insert:
        table, number = bid.rsplit('_', 1)
        number = int(number)
        if table not in tb_block:
            tb_block[table] = []
        tb_block[table].append(number)

    tb_block_range = {}
    for tb, b_list in tb_block.items():
        tb_block_range[tb] = []
        b_list.sort()
        min_b = b_list[0]
        prev_b = b_list[0]
        if len(b_list):
            tb_block_range[tb].append((min_b, prev_b))
        for b in b_list[1:]:
            if b - prev_b == 1:
                prev_b = b
                if b == b_list[-1]:
                    tb_block_range[tb].append((min_b, prev_b))
                continue
            tb_block_range[tb].append((min_b, prev_b))
            min_b = b
            prev_b = b
            if b == b_list[-1]:
                tb_block_range[tb].append((b, b))
    return insert_block_to_cache(tb_block_range)


def sync_caches(cache: LRUCache):
    insert_blocks_to_cache(cache.get_all_content())


def get_non_numeric_cols(table_name):
    query = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}'"
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()

    column_dict = {}
    for row in rows:
        column_name, data_type = row
        if data_type not in ('integer', 'bigint', 'numeric', 'float', 'double precision'):
            column_dict[column_name] = {"type": data_type, 'action': ''}

    return column_dict


# data = ['photoobjall_1', 'photoobjall_2', 'photoobjall_13', 'photoobjall_3', 'photoz_1', 'photoz2_1', 'photoz2_2', 'photoz_3']
# print(insert_blocks_to_cache(data))