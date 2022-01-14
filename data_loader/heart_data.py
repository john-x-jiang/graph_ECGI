import os.path as osp
import numpy as np

import scipy.io
import torch
from torch.utils.data import Dataset

from torch_geometric.data import Data


class HeartGraphDataset(Dataset):
    """
    A dataset of Data objects (in pytorch geometric) with graph attributes
    from a pre-defined graph hierarchy. 
    """

    def __init__(self,
                 root,
                 data_name,
                 signal_type='egm',
                 num_mesh=None,
                 seq_len=None,
                 split='train',
                 subset=1):
        self.root = osp.expanduser(osp.normpath(root))
        self.raw_dir = osp.join(self.root, 'signal/{}/'.format(data_name))

        filename = '{}_{}_{}.mat'.format(split, signal_type, num_mesh)
        self.data_path = osp.join(self.raw_dir, filename)
        matFiles = scipy.io.loadmat(self.data_path, squeeze_me=True, struct_as_record=False)
        dataset = matFiles['params']
        label = matFiles['label']

        dataset = dataset.transpose(2, 0, 1)

        N = dataset.shape[0]
        if subset == 1:
            index = np.arange(N)
        elif subset == 0:
            raise RuntimeError('No data')
        else:
            indices = list(range(N))
            np.random.shuffle(indices)
            split = int(np.floor(subset * N))
            sub_index = indices[:split]
            dataset = dataset[sub_index, :, :]
            index = np.arange(dataset.shape[1])
        
        label = label.astype(int)
        self.label = torch.from_numpy(label[index])
        self.data = torch.from_numpy(dataset[index, :, :]).float()
        # self.corMfree = corMfree
        self.heart_name = data_name
        print('final data size: {}'.format(self.data.shape[0]))

    def __len__(self):
        return (self.data.shape[0])

    def __getitem__(self, idx):
        x = self.data[[idx], :, :]
        y = self.label[[idx]]
        sample = Data(
            x=x,
            y=y,
            pos=self.heart_name
        )
        return sample


class HeartEmptyGraphDataset(Dataset):
    """
    A dataset of Data objects (in pytorch geometric) with graph attributes
    from a pre-defined graph hierarchy. The features and target values are 
    set to zeros in given graph.
    Not suitable for training.
    """

    def __init__(self,
                 mesh_graph,
                 label_type=None):
        self.graph = mesh_graph
        dim = self.graph.pos.shape[0]
        self.datax = np.zeros((dim, 101))
        self.label = np.zeros((101))

    def __len__(self):
        return (self.datax.shape[1])

    def __getitem__(self, idx):
        x = torch.from_numpy(self.datax[:, [idx]]).float()  # torch.tensor(dataset[:,[i]],dtype=torch.float)
        y = torch.from_numpy(self.label[[idx]]).float()  # torch.tensor(label_aha[[i]],dtype=torch.float)

        sample = Data(x=x,
                      y=y,
                      edge_index=self.graph.edge_index,
                      edge_attr=self.graph.edge_attr,
                      pos=self.graph.pos)
        return sample
