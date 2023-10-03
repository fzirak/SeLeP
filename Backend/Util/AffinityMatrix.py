from typing import Dict


class AffinityMatrix:
    def __init__(self):
        self.affinities: Dict[str, AffinityEntry] = {}

    def set_affinities(self, affinities):
        self.affinities = affinities

    def read_affinities_from_file(self, file_path):
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line == '':
                    continue
                block_id = line[0:line.index('(')]
                if block_id == '':
                    continue
                access_count = int(line[line.index('(')+1:line.index(')')])
                freqs_str = line[line.index('{')+1:line.index('}')]
                freqs = {}
                if freqs_str != '':
                    for bid_freq in freqs_str.split(', '):
                        temp = bid_freq.split('=')
                        freqs[temp[0]] = float(temp[1])
                self.affinities[block_id] = AffinityEntry(block_id, '', freqs, access_count)

    def update_affinities(self, b_id, result_set):
        aff_entry = self.affinities.get(b_id)
        if aff_entry is None:
            # print(f'aff mat does not have {b_id}. adding it')
            aff_entry = AffinityEntry(b_id)

        total_blocks = len(result_set)
        aff_entry.increment_access_count()
        for cb_id in result_set:
            if cb_id == b_id:
                continue
            aff_entry.update_frequency(cb_id, total_blocks)
        self.affinities[b_id] = aff_entry
        # print(f'{b_id} was added to aff mat')
        
    def __str__(self):
        return str(self.affinities)

    def set_blocks_partition_id(self, partition_manager):
        for partition in partition_manager.partitions.values():
            for block in partition.blocks:
                if block in self.affinities:
                    self.affinities[block].set_partition_id(partition.partition_id)

    def get_block_aff_entry(self, b_id):
        return self.affinities.get(b_id)

    def get_block_affinities(self, b_id) -> Dict[str, float]:
        try:
            return self.affinities.get(b_id).freqs
        except AttributeError:
            return {}

    def get_block_partition_id(self, b_id):
        if self.affinities.get(b_id) is None:
            print(f'---no affinity entry exists for {b_id}!')
            self.affinities[b_id] = AffinityEntry(b_id)
            return ''
        return self.affinities.get(b_id).p_id

    def update_block_partition(self, b_id, p_id):
        if self.affinities.get(b_id) is None:
            print(f'bp aff mat does not have {b_id}')
        self.affinities.get(b_id).p_id = p_id

    def check_block_existence(self, b_id):
        aff = self.affinities.get(b_id)
        if aff is None:
            return False
        # if aff.p_id == '':
        #     return False
        return True


class AffinityEntry:
    decimal_digit = 6

    def __init__(self, b_id, p_id='', freqs=None, access_count=0):
        self.b_id = b_id
        self.p_id = p_id
        self.freqs: Dict[str, float] = {} if freqs is None else freqs
        self.access_count = access_count

    def set_partition_id(self, p_id):
        self.p_id = p_id

    def __str__(self):
        return '{}: pid={} accessCount={} freqs={} '\
            .format(self.b_id, self.p_id, self.access_count, self.freqs)

    def __repr__(self):
        return f'({self.b_id}: pid={self.p_id}, accessCount={self.access_count}, freqs={self.freqs})'

    def increment_access_count(self):
        self.access_count += 1

    def update_frequency(self, cb_id, total_blocks):
        added_freq = 1.0/total_blocks
        prev_freq = self.freqs.get(cb_id)
        new_freq = prev_freq + added_freq if prev_freq is not None else added_freq
        # print(f'block {self.b_id}, set new freq for {cb_id} = {new_freq}')
        self.freqs[cb_id] = new_freq
