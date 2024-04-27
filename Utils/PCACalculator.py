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

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense


from gensim.models import Word2Vec

tokenizer = None
model = None
scaler = None
max_len = None


class TablePCA:
    def __init__(self, name, pca_df, eig_values_='', eig_vectors_=''):
        self.table_name = name
        self.pca_df = pca_df
        self.eig_values = eig_values_
        self.eig_vectors = eig_vectors_

def get_column_dtype(table_name, cursor):
    sql ='''
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = '{tb}';
    '''
    cursor.execute(sql.format(tb=table_name))
    df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

    res = {}
    for idx, row in df.iterrows():
        res[row['column_name']] = row['data_type']
    return res


def generate_string_encoder(vals, embedding_dim):
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(vals)

    sequences = tokenizer.texts_to_sequences(vals)

    max_len = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_len)

    scaler = MinMaxScaler()
    padded_sequences = scaler.fit_transform(padded_sequences)

    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim, input_length=max_len))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(embedding_dim, activation='relu'))  # Encoder
    model.add(Dense(max_len, activation='sigmoid')) 

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    model.fit(padded_sequences, padded_sequences, epochs=25)
    encoder_model = Model(inputs=model.input, outputs=model.layers[-2].output)

    return tokenizer, encoder_model, scaler, max_len


def smart_get_table_data(table_name, removing_features, cursor, enc_model='w'):
    if enc_model == 'w':
        embedding_dim = 8 #32
    if enc_model == 'c':
        embedding_dim = 32
    sql = 'select * from ' + table_name
    if Config.is_test:
        sql = sql + ' limit 1000;'
    cursor.execute(sql)
    df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
    cols = [desc[0] for desc in cursor.description]
    col_dtype_dict = get_column_dtype(table_name, cursor)
    all_str_vals = []
    str_cols = []
    
    for feature in cols:
        if Config.do_string_encoding:
            if col_dtype_dict[feature] == 'character varying':
                str_cols.append(feature)
            elif 'date' in col_dtype_dict[feature] or 'timestamp' in col_dtype_dict[feature]:
                df[feature] = df[feature].apply(lambda x: datetime.timestamp(pd.to_datetime(x)) if pd.notnull(x) else 0)
            elif  'time' in col_dtype_dict[feature]:
                default_date = datetime(1970, 1, 1) 
                df[feature] = df[feature].apply(lambda x: datetime.combine(default_date, x))
                df[feature] = df[feature].astype(int) // 10**9  # Convert nanoseconds to seconds
        else:
            df = df.drop(feature, axis=1)
    
    if len(str_cols) == 0:
        return df
    
    for col in str_cols:
        df[col] = df[col].fillna('-far-')
        col_vals = df[col].str.lower().to_list()
        if enc_model == 'c':
            all_str_vals = all_str_vals + list(set(col_vals))
        if enc_model == 'w':
            all_str_vals.append(list(set(col_vals)))
    
    global tokenizer, model, scaler, max_len
    if enc_model == 'w':
        model = Word2Vec(all_str_vals, vector_size=embedding_dim, min_count=1)
        model.build_vocab(all_str_vals)
        model.train(all_str_vals, total_examples=len(all_str_vals), epochs=10, report_delay=1)
    
    if enc_model == 'c':
        tokenizer, model, scaler, max_len = generate_string_encoder(list(set(all_str_vals)), embedding_dim)

    print(len(df.columns.to_list()))
    for col in str_cols:
        if enc_model == 'w':
            df[col] = df[col].apply(get_word_embedding)
        if enc_model == 'c':
            df[col] = df[col].apply(get_character_embedding)
        df2 = pd.DataFrame(df[col].to_list(), columns=[f'{col}_{i}' for i in range(embedding_dim)])
        df = pd.concat([df, df2], axis=1)
        df = df.drop(col, axis=1)
    print(len(df.columns.to_list()))

    return df


def get_word_embedding(new_string):
    global model
    return model.wv[new_string.lower()]


def get_character_embedding(new_string):
    global tokenizer, model, scaler, max_len
    new_string = new_string.lower()
    new_sequence = tokenizer.texts_to_sequences([new_string])
    new_padded_sequence = pad_sequences(new_sequence, maxlen=max_len)
    new_padded_sequence = scaler.transform(new_padded_sequence)
    embedding = model.predict(new_padded_sequence, verbose=3)
    return embedding[0]


def pca_calculator(table_list, removing_features, precision, null_threshold=0.6, var_threshold=0.001):
    tables_pca = []
    conn = get_db_connection()
    cursor = conn.cursor()
    max_length = 0
    default_values = [-9999, -999]
    print('Calculating PC values for all tables:')
    for table in tqdm(table_list):
        try:
            df = smart_get_table_data(table, removing_features.get(table), cursor)
            if 'Normalize_NoPCA' in Config.tb_encoding_method:
                df.fillna(0, inplace=True)
                principal_df = pd.DataFrame(data=MinMaxScaler().fit_transform(df))
                max_length = max(max_length, principal_df.shape[1])
                tables_pca.append(TablePCA(table, principal_df))
                continue

            if 'NoPCA' in Config.tb_encoding_method:
                df.fillna(0, inplace=True)
                tables_pca.append(TablePCA(table, df))
                continue

            if 'sdss' in Config.db_name:
                for col in df.columns:
                    if df[col].isin(default_values).any():
                        min_val = df[col][(df[col] != -9999) & (df[col] != -999)].min()
                        df[col] = np.where((df[col] == -9999) | (df[col] == -999), min_val - 5, df[col])

            if 'PCAOnly' not in Config.tb_encoding_method:
                null_percentages = df.isnull().mean()
                columns_to_drop = null_percentages[null_percentages > null_threshold].index.tolist()
                print(f'{len(columns_to_drop)} NaN columns to drop from {len(df.columns)}')
                df = df.drop(columns=columns_to_drop)

                variances = df.var()  # or df.std()
                low_variance_cols = variances[variances < var_threshold].index
                print(f'{len(low_variance_cols)} low variance columns to drop from {len(df.columns)}')
                df.drop(columns=low_variance_cols, inplace=True)

            df.fillna(df.mean(), inplace=True)  

            if 'PCAOnly' in Config.tb_encoding_method:
                precision = Config.encoding_length
                num_cols_to_add = Config.encoding_length - df.shape[1]
                if num_cols_to_add > 0:
                    row_mean = df.mean(axis=1)
                    print(f'Injecting {num_cols_to_add} columns to {table}')
                    additional_cols = pd.DataFrame({f'col_{i}': row_mean for i in range(num_cols_to_add+1)})
                    df = pd.concat([df, additional_cols], axis=1)
                df.fillna(0, inplace=True) 
                

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
            print(e.__repr__())
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
        with open(json_file_path, 'r') as json_file:
            feature_exclude_dict = json.load(json_file)
    elif Config.db_name in ['genomic', 'birds']:
        tables = [item for item in Config.table_list]
        feature_exclude_dict = {}
    else:
        tables = [item for item in Config.table_list if item not in Config.pca_exclude_tables]
        feature_exclude_dict = {}
    return tables, feature_exclude_dict


def main():
    tables, feature_exclude_dict = get_pca_table_info()
    pca_calculator(tables, feature_exclude_dict, 0.9)


if __name__ == '__main__':
    main()
    print('done')
