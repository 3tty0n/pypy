"""KV caches: the dashboard's eval forwards run with use_cache=False, so a
port that builds one has left the measured path."""


class Cache(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("%s: KV cache" % type(self).__name__)


class DynamicCache(Cache):
    pass


class EncoderDecoderCache(Cache):
    pass


class StaticCache(Cache):
    pass
