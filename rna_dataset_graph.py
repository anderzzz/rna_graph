'''Creation of Graph Dataset from RNA rawdata

'''
import os
import pandas as pd

import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import is_undirected

from rawdata import graph_props_x_, create_node_nbond

ALL_DATA_NAME = 'rna_save_file'

class RNADatasetGraph(Dataset):
    '''Bla bla for now

    '''
    def __init__(self, file_in, tmp_file_dir='./tmp_out', create_data=True,
                 filter_noise=True, nonbond_as_node_feature=True, consider_loop_type=True):

        self.file_in = file_in
        self.save_dir = tmp_file_dir
        self.filter_noise = filter_noise
        self.nonbond_nodefeature = nonbond_as_node_feature
        self.consider_loop_type = consider_loop_type

        if create_data:
            self._create_data()
        else:
            print ('Data no created, only loaded. Noise filter and node feature configurations implied.')

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        super(RNADatasetGraph, self).__init__()

    def get_raw_data(self, idx):

        df = pd.read_json(self.file_in, lines=True)
        if self.filter_noise:
            df = df.loc[df['SN_filter'] == 1]
        return df.iloc[idx]

    def _create_data(self):

        # Include rows that are explicitly provided
        df = pd.read_json(self.file_in, lines=True)

        counter = 0
        for node_props_x, node_props_y, e_index_1, e_index_2, edge_props, n_scored, sn_val, sn_filter in graph_props_x_(df, self.consider_loop_type):
            edge_index = torch.tensor([e_index_1, e_index_2], dtype=torch.long)
            if not is_undirected(edge_index):
                raise RuntimeError('Directed graph created, should be impossible')

            # Ignore all data that does not meet signal-noise threshold
            if self.filter_noise:
                if sn_filter != 1:
                    continue

            # Add feature of incoming edges to node as a node feature
            if self.nonbond_nodefeature:
                node_nbond_feature = create_node_nbond(node_props_x, e_index_1, e_index_2, edge_props)
                x = torch.tensor(node_nbond_feature, dtype=torch.float)
            else:
                x = torch.tensor(node_props_x, dtype=torch.float)
            edge_x = torch.tensor(edge_props, dtype=torch.float)

            # Add ground-truth node features
            if not node_props_y is None:
                y = torch.tensor(node_props_y, dtype=torch.float)
            else:
                y = None

            # Create mask for which nodes are to be predicted
            train_mask = torch.zeros(x.shape[0], dtype=torch.bool)
            train_mask[:n_scored] = True

            data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_x,
                        train_mask=train_mask, sn_val=sn_val)
            torch.save(data, '{}/{}_{}.pt'.format(self.save_dir, ALL_DATA_NAME, counter))
            counter += 1

        self.total_processed = counter

    def len(self):
        return self.total_processed

    def get(self, idx):
        return torch.load('{}/{}_{}.pt'.format(self.save_dir, ALL_DATA_NAME, idx))

    @property
    def n_pred_dim(self):
        return self.get(0).y.shape[-1]

    @property
    def n_node_dim(self):
        return self.get(0).x.shape[-1]

    @property
    def n_edge_dim(self):
        return self.get(0).edge_attr.shape[-1]


def test1():
    rna_data = RNADatasetGraph('./data/train.json', './tmp_out', True, True, True)
    print (rna_data)
    print (rna_data[0])

    rna_data = RNADatasetGraph('./data/train.json', './tmp_out', True, True, False)
    print (rna_data)
    print (rna_data[5])
    print (rna_data[5].x)
    print (rna_data.get_raw_data(5).structure)
    print (rna_data.get_raw_data(5).sequence)

