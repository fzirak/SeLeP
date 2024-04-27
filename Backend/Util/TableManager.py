from typing import Dict

from Backend.Util.Table import Table


class TableManager:
    def __init__(self):
        self.tables: Dict[str, Table] = {}

    def add_table(self, tb):
        self.tables[tb.name] = tb

    def calculate_table_encodings(self, latent_dim, epoch_no, encoding_method):
        for table in self.tables.values():
            table.calculate_table_block_encoding(latent_dim, epoch_no, encoding_method)

    def get_block_encoding(self, table_name, block_name):
        # TODO: solve this issue
        if table_name not in self.tables:
            print(self.tables.keys())
            print(table_name + ' and ' + block_name + ' is missing. bypassing the error')
            return ''
        return self.tables.get(table_name).get_block_encoding(block_name)

    def get_block_pid(self, bid):
        tb = bid.rsplit('_', 1)[0]
        return self.tables[tb].get_block_pid(bid)
