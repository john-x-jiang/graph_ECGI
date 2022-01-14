import os
import numpy as np

from data_loader.heart_data import HeartGraphDataset, HeartGraphDomainDataset
from torch_geometric.loader import DataLoader


class HeartDataLoader(DataLoader):
    def __init__(self, batch_size, data_dir='data/', split='train', shuffle=True, collate_fn=None,
                 num_workers=1, data_name=None, signal_type=None, num_mesh=None, seq_len=None, k_shot=None):
        # assert split in ['train', 'valid', 'test', 'test00', 'test01','test10', 'test11']

        self.dataset = HeartGraphDataset(data_dir, data_name, signal_type, num_mesh, seq_len, split)

        super().__init__(self.dataset, batch_size, shuffle, drop_last=True, num_workers=num_workers)
