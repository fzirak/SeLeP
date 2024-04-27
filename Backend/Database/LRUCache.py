from typing import Dict


class Node:
    def __init__(self, key):
        self.key = key
        self.prev = None
        self.next = None


class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache: Dict[str, Node] = {}
        self.head = Node(None)
        self.tail = Node(None)
        self.head.next = self.tail
        self.tail.prev = self.head
        self.hit_count = 0
        self.miss_count = 0
        self.total_pres = 0

    def get_all_content(self):
        return list(self.cache.keys())

    def get(self, key, increase_hit: bool):
        if key in self.cache:
            node = self.cache[key]
            self._remove(node)
            self._add(node)
            if increase_hit:
                self.hit_count += 1
            return node.key
        if increase_hit:
            self.miss_count += 1
        return None

    def put(self, key, increase_hit: bool, insert_type='n'):
        if key not in self.cache:
            if increase_hit:
                self.miss_count += 1

            if insert_type == 'p':
                self.total_pres += 1
        else:
            if increase_hit:
                self.hit_count += 1
            self._remove(self.cache[key])
        node = Node(key)
        self._add(node)
        self.cache[key] = node
        if len(self.cache) > self.capacity:
            node_to_remove = self.head.next
            self._remove(node_to_remove)
            del self.cache[node_to_remove.key]

    def _add(self, node):
        prev_node = self.tail.prev
        prev_node.next = node
        node.prev = prev_node
        node.next = self.tail
        self.tail.prev = node

    def _remove(self, node):
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def get_hit_ratio(self):
        if self.get_total_access() == 0:
            return 0
        return self.hit_count/self.get_total_access()

    def get_total_access(self):
        return self.hit_count + self.miss_count

    def __repr__(self):
        return f"LRUCache({self.capacity})"

    def __str__(self):
        return f"LRUCache({self.capacity}): {list(self.cache.keys())}"

    def report(self, name):
        return f'{name}: total access = {self.get_total_access()}, total prefetches = {self.total_pres} ' \
               f'hits = {self.hit_count}, hit ratio = {self.get_hit_ratio()}'

    def report_dict(self):
        res = {
            'total_access': self.get_total_access(),
            'total_prefetches': self.total_pres,
            'hits': self.hit_count,
            'hit_ratio': self.get_hit_ratio(),
            'final_cache_usage': len(self.cache)
        }
        return res
