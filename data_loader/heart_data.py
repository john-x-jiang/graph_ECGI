import os.path as osp
import numpy as np
import h5py
import scipy.io
import torch
from torch.utils.data import Dataset

from torch_geometric.data import Data


class DataWithDomain(Data):
    def __init__(self, x, y, label, pos, mask=None, D_x=None, D_y=None, D_label=None):
        super().__init__()
        self.x = x
        self.y = y
        self.label = label
        self.pos = pos
        self.mask = mask
        self.D_x = D_x
        self.D_y = D_y
        self.D_label = D_label


class HeartGraphDataset(Dataset):
    """
    A dataset of Data objects (in pytorch geometric) with graph attributes
    from a pre-defined graph hierarchy. 
    """

    def __init__(self,
                 root,
                 data_name,
                 signal_type='egm',
                 seq_len=None,
                 split='train'):
        self.root = osp.expanduser(osp.normpath(root))
        self.raw_dir = osp.join(self.root, 'signal/{}/'.format(data_name))

        filename = '{}_{}.mat'.format(split, signal_type)
        self.data_path = osp.join(self.raw_dir, filename)

        self.heart_name = data_name
        try:
            matFiles = scipy.io.loadmat(self.data_path, squeeze_me=True, struct_as_record=False)
            xs = matFiles['sources']
            ys = matFiles['measurements']
            label = matFiles['label']

            xs = xs.transpose(2, 0, 1)
            ys = ys.transpose(2, 0, 1)
            label = label.astype(int)

            self.label = torch.from_numpy(label)
            self.xs = torch.from_numpy(xs).float()
            self.ys = torch.from_numpy(ys).float()

            self.file_versions = 1
        except NotImplementedError:
            matFiles = h5py.File(self.data_path, 'r')
            xs = matFiles['sources']
            ys = matFiles['measurements']
            label = matFiles['label']

            self.xs = xs
            self.ys = ys
            self.label = label

            self.file_versions = 2
        except:
            ValueError('could not read at all...')
        
        print('final data size: {}'.format(self.xs.shape[0]))

    def __len__(self):
        return (self.xs.shape[0])

    def __getitem__(self, idx):
        if self.file_versions == 1:
            x = self.xs[[idx], :, :]
            y = self.ys[[idx], :, :]
            label = self.label[[idx]]
        elif self.file_versions == 2:
            x = self.xs[[idx], :, :]
            y = self.ys[[idx], :, :]
            label = self.label[:, [idx]]

            x = torch.from_numpy(x).float()
            y = torch.from_numpy(y).float()
            label = torch.from_numpy(label).int()
            x = x.permute(0, 2, 1).contiguous()
            y = y.permute(0, 2, 1).contiguous()
            label = label.permute(1, 0).contiguous()
        else:
            raise NotImplemented
        
        sample = Data(
            x=x,
            y=y,
            label=label,
            pos=self.heart_name
        )
        return sample


class HeartEpisodicDataset(Dataset):
    def __init__(self,
                 root,
                 data_name,
                 signal_type='egm',
                 seq_len=None,
                 split='train',
                 shuffle=True,
                 k_shot=2):
        self.root = osp.expanduser(osp.normpath(root))
        self.raw_dir = osp.join(self.root, 'signal/{}/'.format(data_name))
        self.k_shot = k_shot
        self.split_name = split
        self.shuffle = shuffle

        filename = '{}_{}_{}.mat'.format(split, signal_type)
        self.data_path = osp.join(self.raw_dir, filename)
        
        self.heart_name = data_name
        try:
            matFiles = scipy.io.loadmat(self.data_path, squeeze_me=True, struct_as_record=False)
            xs = matFiles['sources']
            ys = matFiles['measurements']
            label = matFiles['label']

            xs = xs.transpose(2, 0, 1)
            ys = ys.transpose(2, 0, 1)
            label = label.astype(int)
            scar = label[:, 1]
            
            self.label = torch.from_numpy(label)
            self.xs = torch.from_numpy(xs).float()
            self.ys = torch.from_numpy(ys).float()

            self.file_versions = 1
        except NotImplementedError:
            matFiles = h5py.File(self.data_path, 'r')
            xs = matFiles['sources']
            ys = matFiles['measurements']
            label = matFiles['label']

            self.xs = xs
            self.ys = ys
            self.label = label
            scar = label[1, :].astype(int)

            self.file_versions = 2
        except:
            ValueError('could not read at all...')

        unique_scar = np.unique(scar)
        self.scar_idx = {}
        for s in unique_scar:
            idx = np.where(scar == s)[0]
            self.scar_idx[s] = idx

        print('final data size: {}'.format(self.xs.shape[0]))
        self.split()

    def __len__(self):
        return (self.qry_idx.shape[0])

    def __getitem__(self, idx):
        if self.file_versions == 1:
            label = self.label[[self.qry_idx[idx]]]
            x = self.xs[[self.qry_idx[idx]], :, :]
            y = self.ys[[self.qry_idx[idx]], :, :]

            scar = label[:, 1].numpy()[0]
            D_x = self.xs[self.spt_idx[scar], :, :]
            D_y = self.ys[self.spt_idx[scar], :, :]
            D_label = self.label[self.spt_idx[scar]]
            
        elif self.file_versions == 2:
            label = self.label[:, [self.qry_idx[idx]]]
            x = self.xs[[self.qry_idx[idx]], :, :]
            y = self.ys[[self.qry_idx[idx]], :, :]
            
            x = torch.from_numpy(x).float()
            y = torch.from_numpy(y).float()
            label = torch.from_numpy(label).int()
            x = x.permute(0, 2, 1).contiguous()
            y = y.permute(0, 2, 1).contiguous()
            label = label.permute(1, 0).contiguous()

            scar = label[:, 1].numpy()[0]
            D_x = self.xs[self.spt_idx[scar], :, :]
            D_y = self.ys[self.spt_idx[scar], :, :]
            D_label = self.label[:, self.spt_idx[scar]]

            D_x = torch.from_numpy(D_x).float()
            D_y = torch.from_numpy(D_y).float()
            D_label = torch.from_numpy(D_label).int()
            D_x = D_x.permute(0, 2, 1).contiguous()
            D_y = D_y.permute(0, 2, 1).contiguous()
            D_label = D_label.permute(1, 0).contiguous()
        else:
            raise NotImplemented

        num_sample = min(len(self.scar_idx[scar]), self.k_shot)
        D_label = D_label.view(1, num_sample, -1)

        sample = DataWithDomain(
            x=x,
            y=y,
            label=label,
            pos=self.heart_name,
            # mask=self.mask,
            D_x=D_x,
            D_y=D_y,
            D_label=D_label
        )
        return sample
    
    def split(self):
        self.spt_idx = {}
        self.qry_idx = []
        for scar_id, samples in self.scar_idx.items():
            sample_idx = np.arange(0, len(samples))
            if len(samples) < self.k_shot:
                self.spt_idx = samples
            else:
                if self.shuffle:
                    np.random.shuffle(sample_idx)
                    spt_idx = np.sort(sample_idx[0:self.k_shot])
                else:
                    spt_idx = sample_idx[0:self.k_shot]
                self.spt_idx[scar_id] = samples[spt_idx]
            
            self.qry_idx.extend(samples.tolist())
        
        self.qry_idx = np.array(self.qry_idx)
        self.qry_idx = np.sort(self.qry_idx)


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
