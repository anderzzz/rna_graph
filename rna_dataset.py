'''Bla bla for now

'''
from os import listdir

import torch
from torch_geometric.data import Dataset

from data_creator import GRAPH_NAME

class RNADataset(Dataset):
    '''Bla bla for now

    '''
    def __init__(self, root_dir):

        self.root_dir = root_dir
        super(RNADataset, self).__init__(root_dir)

    def len(self):
        return len([file_name for file_name in listdir(self.root_dir) if '.pt' in file_name])

    def _load(self, idx):
        return torch.load('{}/{}_{}.pt'.format(self.root_dir, GRAPH_NAME, idx))

    def get(self, idx):
        return self._load(idx)

    @property
    def n_seq_score(self, idx):
        return self._load(idx).n_seq_score

    @property
    def n_seq_total(self, idx):
        return self._load(idx).n_seq_total
