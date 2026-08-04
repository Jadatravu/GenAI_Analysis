"""A small LRU cache implementation and a decorator to apply it to functions."""
from collections import OrderedDict
from functools import wraps


class LRUCache:
    """Fixed-capacity least-recently-used cache backed by an OrderedDict."""

    def __init__(self, capacity: int = 128):
        self.capacity = capacity
        self._store: OrderedDict = OrderedDict()

    def get(self, key):
        """Return the cached value for `key`, or None if missing. Marks as recently used."""
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key, value):
        """Insert or update `key` with `value`, evicting the LRU item if over capacity."""
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)

    def clear(self):
        """Remove all entries from the cache."""
        self._store.clear()


def cached(capacity: int = 128):
    """Decorator factory that memoizes a function's return value using LRUCache."""

    def decorator(func):
        cache = LRUCache(capacity=capacity)

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            hit = cache.get(key)
            if hit is not None:
                return hit
            result = func(*args, **kwargs)
            cache.put(key, result)
            return result

        return wrapper

    return decorator
