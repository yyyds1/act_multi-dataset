import os
import importlib
from .base import BaseDataset


class DataFactory:
    _registry = {}

    @classmethod
    def register(cls, data_type: str, data_class):
        """Register a new data type."""
        cls._registry[data_type] = data_class

    @classmethod
    def create_data(cls, data_type: str, *args, **kwargs) -> BaseDataset:
        if data_type not in cls._registry:
            raise ValueError(f"Data type '{data_type}' not registered.")
        return cls._registry[data_type](*args, **kwargs)

    @classmethod
    def auto_register_data(cls, directory: str, base_package: str):
        """Automatically import all data modules in the directory."""
        for filename in os.listdir(directory):
            if filename.endswith(".py") and filename != "__init__.py":
                module_name = f"{base_package}.{filename[:-3]}"
                importlib.import_module(module_name)