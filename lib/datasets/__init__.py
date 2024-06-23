
from .WebCeph2k import WebCeph2k

__all__ = ['WebCeph2k', 'get_dataset']


def get_dataset(config):

    if config.DATASET.DATASET == 'WebCeph2k':
        return WebCeph2k
    else:
        raise NotImplemented()

