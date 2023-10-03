import random
from typing import Dict

from Utils.AutoEncoder import encode_table
from . import Block
import pandas as pd
import numpy as np


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

    def calculate_table_block_encoding(self, latent_dim, epoch_no):
        block_matrices = self.get_block_list()
        eval_matrices = random_matrix_selection(10, block_matrices)
        autoencoder = encode_table(block_matrices, eval_matrices, latent_dim, epoch_no)
        self.set_blocks_encoding(autoencoder)

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
