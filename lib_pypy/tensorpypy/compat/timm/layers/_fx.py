"""timm.layers._fx: upstream records these functions and modules as leaves
for torch.fx feature extraction, and returns them unchanged; no fx tracing
happens here."""


def register_notrace_function(func):
    return func


def register_notrace_module(module):
    return module


def create_feature_extractor(*args, **kwargs):
    raise NotImplementedError("fx feature extraction")


def get_graph_node_names(*args, **kwargs):
    raise NotImplementedError("fx feature extraction")
