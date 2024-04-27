import random
from typing import Dict

import pandas as pd
import numpy as np
from  lshashpy3 import *
from sklearn.decomposition import PCA, DictionaryLearning
from keras import backend as K

from Utils.AutoEncoder import encode_table
from . import Block
from Configuration.config import Config


class Table:
    def __init__(self, name, pca_val=None):
        if pca_val is None:
            pca_val = []
        self.blocks: Dict[str, Block.Block] = {}
        self.name = name
        self.pca = pca_val

    def add_block(self, b):
        self.blocks[b.name] = b

    def get_block_list(self):
        return [block.pca_df for block in self.blocks.values()]

    def set_block_encoding(self, block_name, enc):
        self.blocks.get(block_name).set_encoding(enc)

    def set_blocks_encoding(self, autoencoder):
        for block in self.blocks.values():
            encoded_block = autoencoder.encoder(np.array([block.pca_df])).numpy()
            block.set_encoding(encoded_block)
            self.blocks[block.name] = block
            # print(encoded_block)

    def set_blocks_encoding(self, encoding_method, encoder_model):
        for block in self.blocks.values():
            if 'AutoEncoder' in encoding_method:
                encoded_block = encoder_model.encoder(np.array([block.pca_df])).numpy()
            elif 'LSH' in encoding_method:
                encoded_block = self.get_block_encoding_from_lsh(block.pca_df, lshash_=encoder_model, 
                                    l_hash=int(encoding_method.split('_')[-1]), n_hash_tb=int(encoding_method.split('_')[-2]))
            elif 'PCAOnly' in encoding_method:
                aggregated_rows = block.pca_df.mean(axis=0)
                encoded_block = np.array(aggregated_rows)
            elif 'SparceEncoding' in encoding_method:
                encoded_block = self.get_sparse_representation(encoder_model, block.pca_df)
            block.set_encoding(encoded_block)
            self.blocks[block.name] = block
        
        K.clear_session()

    def get_sparse_representation(self, encoder_model, block):
            flattened_data = block.values.flatten()
            sparse_representation = encoder_model.transform(flattened_data)
            return sparse_representation


    def get_block_encoding_from_lsh(self, block, lshash_, l_hash, n_hash_tb):
        if block.values.flatten().tolist() is None:
            print(f'potential issue in table {self.name}. None type for block')
            return None
        hash_index = lshash_.index(block.values.flatten().tolist())
        hash_index_concat = ''.join(hash_index)
        chunk_size = int(l_hash*n_hash_tb/Config.encoding_length)
        encoding = []
        for i in range(0, len(hash_index_concat), chunk_size):
            chunk = hash_index_concat[i:i+chunk_size]
            val = int(chunk, 2)/(2 ** chunk_size)
            if val > 1:
                print('----- val gt 1!')
            encoding.append(val)
        return np.array(encoding)
 

    def extract_block_pcas(self, block_info_df):
        pca_with_block_df = pd.concat([self.pca.pca_df, block_info_df], axis=1)
        block_pcas = pca_with_block_df.groupby('block_number')
        max_rows = max(block_pca.shape[0] for _, block_pca in block_pcas)
        # print(f'{self.name}\'s max rows in extract pcas is {max_rows} and there are {len(block_pcas)} blocks '
        #       f'({len(block_pcas) * max_rows}rows)')
        for block_number, block_pca in block_pcas:
            block_pca = block_pca.drop('block_number', axis=1)
            # make sure all blocks has the same number of rows to have same size matrices
            if block_pca.shape[0] < max_rows:
                means = block_pca.mean(axis=0)
                new_rows = pd.DataFrame([[None] * len(block_pca.columns)] * (max_rows - block_pca.shape[0]),
                                        columns=block_pca.columns)
                new_rows = new_rows.fillna(means)
                block_pca = pd.concat([block_pca, new_rows])
            self.add_block(Block.Block(self.name + '_' + str(block_number), block_pca))


    def calculate_table_block_encoding(self, latent_dim, epoch_no, encoding_method='AutoEncoder_0'):
        print(f'>> Encoding table {self.name}')
        eval_percent = 10
        block_matrices = self.get_block_list()
        encoder_model = None
        encoding_opts = encoding_method.split('_')
        if 'AutoEncoder' in encoding_method:
            eval_matrices = random_matrix_selection(eval_percent, block_matrices)
            if self.name == 'specphotoall' and 'NoPCA' in Config.tb_encoding_method:
                block_matrices = block_matrices[:int(len(block_matrices)/1.5)]
                eval_matrices = random_matrix_selection(5, block_matrices)

            encoder_model = encode_table(block_matrices, eval_matrices, latent_dim, epoch_no, encoder_option=int(encoding_opts[-1]))
        elif 'LSH' in encoding_method:
            n_rows, n_cols = block_matrices[0].shape[0], block_matrices[0].shape[1]
            encoder_model = LSHash(int(encoding_opts[-1]), n_rows*n_cols, num_hashtables=int(encoding_opts[-2]))
        elif 'SparceEncoding' in encoding_method:
            encoder_model = DictionaryLearning(n_components=Config.encoding_length, alpha=1.0)
            block_vectors = [matrix.values.flatten() for matrix in block_matrices]
            X = np.array(block_vectors)
            encoder_model.fit(X)
        elif 'PCAOnly' in encoding_method:
            # n_rows, n_cols = block_matrices[0].shape[0], block_matrices[0].shape[1]
            # encoder_model = PCA(n_components=32)
            encoder_model = 'PCAOnly'
        self.set_blocks_encoding(encoding_method, encoder_model)

    
    def get_block_encoding(self, block_name):
        # print(self.blocks.keys())
        if self.blocks.get(block_name) is None:
            # print(self.blocks.keys())
            print(block_name + ' is missing. bypassing the error')
            return None

        return self.blocks.get(block_name).encoding

    def get_block_pid(self, bid):
        b = self.blocks.get(bid)
        if b is None:
            return None
        return b.pid



def random_matrix_selection(percentage, block_matrices):
    n = int(len(block_matrices) * percentage / 100)
    random_matrices = random.sample(block_matrices, n)
    return random_matrices
