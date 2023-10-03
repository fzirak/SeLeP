import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from Backend.Database.helpers import get_db_connection, get_non_numeric_cols, get_block_indexes
from Backend.Util import TableManager, Table
from Configuration.config import Config

from tqdm import tqdm
import json
from datetime import datetime


class TablePCA:
    def __init__(self, name, pca_df, eig_values_='', eig_vectors_=''):
        self.table_name = name
        self.pca_df = pca_df
        self.eig_values = eig_values_
        self.eig_vectors = eig_vectors_


def get_table_data(table_name, removing_features, cursor):
    sql = 'select * from ' + table_name
    cursor.execute(sql)
    df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

    if removing_features:
        for feature in removing_features:
            df = df.drop(feature, axis=1)
    # else:
    #     df = df.drop('f_id', axis=1)

    return df


def pca_calculator(table_list, removing_features, precision, null_threshold=0.6, var_threshold=0.001):
    tables_pca = []
    conn = get_db_connection()
    cursor = conn.cursor()
    # success = []
    max_length = 0
    print('Calculating PC values for all tables:')
    for table in tqdm(table_list):
        try:
            df = get_table_data(table, removing_features.get(table), cursor)
            null_percentages = df.isnull().mean()
            columns_to_drop = null_percentages[null_percentages > null_threshold].index.tolist()
            print(f'{len(columns_to_drop)} NaN columns to drop from {len(df.columns)}')
            df = df.drop(columns=columns_to_drop)

            # if df.shape[1] > 100:
            variances = df.var()  # or df.std()
            low_variance_cols = variances[variances < var_threshold].index
            print(f'{len(low_variance_cols)} low variance columns to drop from {len(df.columns)}')
            df.drop(columns=low_variance_cols, inplace=True)

            # nan_count = df.isnull().sum().sum()
            # print(f'nan count is {nan_count}')
            df.fillna(df.mean(), inplace=True)  # it was 0 instead of mean before
            scalar = MinMaxScaler()
            # # scalar = StandardScaler()
            x = scalar.fit_transform(df)
            pca = PCA(n_components=precision)
            principal_components = pca.fit_transform(x)
            principal_df = pd.DataFrame(data=MinMaxScaler().fit_transform(principal_components))
            tables_pca.append(TablePCA(table, principal_df, pca.explained_variance_, pca.components_))
            max_length = max(max_length, len(pca.explained_variance_ratio_))
            print(f'{table}: { len(pca.explained_variance_ratio_)}')

        except Exception as e:
            print('exception for table {} with message: {}'.format(table, e))
            # break
    print('max num of PCs is: ' + str(max_length))
    return tables_pca


def pca_calculator2(table_list, feature_exclude_dict, precision, null_threshold=0.6, var_threshold=0.001):
    tables_pca = []
    conn = get_db_connection()
    cursor = conn.cursor()
    max_length = 0
    for table in table_list:
        print(f'>>> {table}')
        try:
            col_actions = feature_exclude_dict.get(table)
            if col_actions is None:
                print(f'No col-action entry for table {table}')
                removing_features = None
            else:
                removing_features = []
                for key, val in col_actions.items():
                    if val['action'] == 'del':
                        removing_features.append(key)
            df = get_table_data(table, removing_features, cursor)
            null_percentages = df.isnull().mean()
            columns_to_drop = null_percentages[null_percentages > null_threshold].index.tolist()
            print(f'{len(columns_to_drop)} NaN columns to drop from {len(df.columns)}')
            df = df.drop(columns=columns_to_drop)
            if col_actions is not None:
                for key, val in col_actions.items():
                    if val['action'] == 'cast':
                        df[key] = df[key].str.rstrip()
                        df[key] = df[key].fillna("0")
                        df[key] = df[key].astype(int)
                    elif val['action'] == 'enum':
                        df[key] = df[key].str.rstrip()
                        unique_values = df[key].unique()
                        value_to_integer = {value: idx for idx, value in enumerate(unique_values)}
                        df[key] = df[key].map(value_to_integer)
                    elif val['action'] == 'unix':
                        if 'char' in val['type']:
                            df[key] = df[key].str.rstrip()
                            df[key] = pd.to_datetime(df[key], format='%Y-%m-%d')
                        # df['temp_col'] = df[key].apply(lambda x_: int(x_.timestamp()))
                        df['temp_col'] = df[key].apply(
                            lambda x_: int(datetime(x_.year, x_.month, x_.day).timestamp()) if x_ is not None else None)
                        df[key] = df['temp_col']
                        df = df.drop(columns=['temp_col'])

            variances = df.var(numeric_only=True)  # or df.std()
            low_variance_cols = variances[variances < var_threshold].index
            print(f'{len(low_variance_cols)} low variance columns to drop from {len(df.columns)}')
            df.drop(columns=low_variance_cols, inplace=True)

            df.fillna(df.mean(), inplace=True) 
            scalar = MinMaxScaler()
            x = scalar.fit_transform(df)

            # # Do not apply PCA
            # principal_df = pd.DataFrame(data=x)
            # tables_pca.append(TablePCA(table, principal_df))

            pca = PCA(n_components=precision)
            principal_components = pca.fit_transform(x)
            principal_df = pd.DataFrame(data=MinMaxScaler().fit_transform(principal_components))
            tables_pca.append(TablePCA(table, principal_df, pca.explained_variance_, pca.components_))
            max_length = max(max_length, len(pca.explained_variance_ratio_))
            print(f'{table}: {len(pca.explained_variance_ratio_)}')

        except NotADirectoryError as e:
            print('exception for table {} with message: {}'.format(table, e.__repr__()))
            break
    print('max num of PCs is: ' + str(max_length))
    return tables_pca


def get_pca_table_info():
    if Config.db_name == 'tpcds':
        tables = [item for item in Config.table_list]
        feature_exclude_dict = {}
        json_file_path = Config.base_dir + 'Data/tb_col_summary.json'
        # for tb in tables:
        #     col_summary = get_non_numeric_cols(tb)
        #     if len(col_summary) > 0:
        #         feature_exclude_dict[tb] = col_summary
        #
        # with open(json_file_path, 'w') as json_file:
        #     json.dump(feature_exclude_dict, json_file, indent=4)

        with open(json_file_path, 'r') as json_file:
            feature_exclude_dict = json.load(json_file)
    else:
        tables = [item for item in Config.table_list if item not in Config.pca_exclude_tables]
        feature_exclude_dict = {'chunk': ['targetversion', 'exportversion'], 'dataconstants': ['field', 'name', 'description'], 'dbobjects': ['name', 'type', 'access', 'description', 'text'], 'frame': ['img'], 'history': ['filename', 'date', 'name', 'description'], 'indexmap': ['code', 'type', 'tablename', 'fieldlist', 'foreignkey', 'indexgroup'], 'mask': ['area'], 'match': ['miss'], 'objmask': ['span'], 'partitionmap': ['filegroupname', 'comment'], 'platex': ['expid', 'taihms', 'dateobs', 'timesys', 'quality', 'name', 'program', 'version', 'observer', 'camver', 'spec2dver', 'utilsver', 'spec1dver', 'readver', 'combver', 'fscanmode', 'programname', 'plateversion', 'fscanversion', 'fmapversion'], 'profiledefs': [], 'propermotions': [], 'pubhistory': ['tend', 'name'], 'qsobest': [], 'qsobunch': ['headobjtype'], 'qsocatalogall': ['headobjtype'], 'rc3': ['aliases', 'pgc_name', 'rc2_type', 'rc2_typesource', 'name', 'b_tsource'], 'region2box': ['regiontype', 'boxtype'], 'regionpatch': ['type', 'convexstring'], 'rmatrix': ['mode'], 'sdssconstants': ['name', 'unit', 'description'], 'sector': ['tiles', 'targetversion'], 'segment': ['photoversion', 'targetastroid', 'targetastroversion', 'exportastroid', 'exportastroversion', 'targetfcalibid', 'targetfcalibversion', 'exportfcalibid', 'exportfcalibversion', 'loaderversion', 'objectsource', 'targetsource', 'targetversion', 'photoid'], 'sitediagnostics': ['name', 'type'], 'speclineall': [], 'speclineindex': ['name'], 'specobjall': ['img', 'objtypename'], 'specphotoall': [], 'stetson': ['name'], 'stripedefs': ['hemisphere', 'htmarea'], 'tileall': ['programname', 'completetileversion'], 'tilinggeometry': ['nsbx', 'targetversion'], 'tilingrun': ['tileversion', 'tilepid', 'programname']}
    return tables, feature_exclude_dict


def main():
    tables, feature_exclude_dict = get_pca_table_info()
    pca_calculator(tables, feature_exclude_dict, 0.9)


if __name__ == '__main__':
    main()
    print('done')
