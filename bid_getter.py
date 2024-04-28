import psycopg2
import pandas as pd
from random import shuffle
from datetime import datetime, timedelta
from tqdm import tqdm
import subprocess

from Configuration.config import Config


cache_sql = '''
        select distinct c.relname as tb, b.relblocknumber as b_num
        from pg_class c inner join pg_buffercache b
            on b.relfilenode=c.relfilenode inner join pg_database d
            on (b.reldatabase=d.oid and d.datname='tpcds')
        where c.relname not like 'pg_%' and c.reltype != 0
        limit 100000
    '''
system_password = Config.db_password


def clear_cache():
    subprocess.call(
        'echo {} | sudo -S service postgresql restart'.format(system_password),
        shell=True)
    

def get_df_bid(df, file_name, sep='$'):
    df['resultBlock'] = '-'
    db_params = {
        'dbname': Config.db_name,
        'user': Config.db_user,
        'password': Config.db_password,
        'host': Config.db_host,
        'port': Config.db_port,
    }

    with open(file_name, 'a') as f:
        header = sep.join(df.columns)
        f.write(header + '\n')
        for idx, row in tqdm(df.iterrows()):
            try:
                clear_cache()
                conn = psycopg2.connect(**db_params)
                cursor = conn.cursor()
                conn.autocommit = True
                cursor.execute("SET statement_timeout TO 1000000")
                cursor.execute(row['statement'])
                res = cursor.fetchall()

                cursor.execute(cache_sql)
                res_df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
                bids = []
                for _, cache_row in res_df.iterrows():
                    bids.append(f"{cache_row['tb']}_{cache_row['b_num']}")
                row['resultBlock'] = bids
                row_str = sep.join(str(val) for val in row)
                f.write(row_str + '\n')
                # if idx == 3:
                #     break

            except Exception as e:
                print(f'error at {idx}')
                print(e.__repr__())


def sort_df(df):
    df['theTime'] = pd.to_datetime(df['theTime'], format='%m/%d/%Y %I:%M:%S %p')
    df['minTime'] = df.groupby('clientIP')['theTime'].transform('min')
    df = df.sort_values(by=['minTime', 'theTime'])
    df = df.reset_index(drop=True)
    df = df.drop('minTime', axis=1)

    df['theTime'] = df['theTime'].dt.strftime('%m/%d/%Y %I:%M:%S %p')


if __name__ == '__main__':
    base_dir = './'
    workload_files = ['train_all', 'test1']
    for wfile in workload_files:
        df = pd.read_csv(f'{base_dir}{wfile}.csv', header=0, sep='$', engine='python')
        sort_df(df)
        get_df_bid(df, f'{base_dir}{wfile}B1.csv', sep='$')