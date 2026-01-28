from .factory import DataFactory


def register_dataset(manipdata_type):
    def decorator(cls):
        DataFactory.register(manipdata_type, cls)
        return cls

    return decorator