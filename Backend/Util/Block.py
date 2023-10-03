class Block:
    def __init__(self, name, pca_val=[]):
        self.name = name
        self.pca_df = pca_val
        self.encoding = None
        self.pid = None

    def set_encoding(self, enc):
        self.encoding = enc

    def set_partitionid(self, pid):
        self.pid = pid

