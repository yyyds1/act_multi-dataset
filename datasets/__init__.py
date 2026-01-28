import os
from .factory import DataFactory
from .base import BaseDataset

current_dir = os.path.dirname(os.path.abspath(__file__))
base_package = __name__

DataFactory.auto_register_data(current_dir, base_package)
