from collections import OrderedDict


class LruCache:
    def __init__(self, max_size, create_fn):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.create_fn = create_fn

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]

        value = self.create_fn(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
        return value
