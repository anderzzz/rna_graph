'''Creation of Datasets from RNA rawdata

'''
import os
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from rawdata import graph_props_x_, create_node_nbond, create_seqdist_nodefeature, vanilla_props_
ALL_DATA_NAME = 'rna_save_file'

class RNADataset(Dataset):
    '''RNA Dataset for PyTorch Deep Learning

    Args:
        file_in (str): Path to JSON file with raw data as obtained from Kaggle
        tmp_file_dir (str, optional): Path to directory in which processed dataset files are saved
        create_data (bool, optional): If True, the processed dataset files are created from the raw data. If False, the
            processed files are only linked to. The latter is a dangerous option since no validation of creation
            parameters is done
        filter_noise (bool, optional): Remove data from consideration with a flagged signal-noise value
        nonbond_as_node_feature (bool, optional): Add to the node features the non-bonded interaction the corresponding
            nucleotide engages in, which is either no interaction, A-U interaction, C-G interaction, or other pairing.
        consider_loop_type (bool, optional): Add to the node features the predicted loop type the corresponding
            nucleotide engages in

    '''
    def __init__(self, file_in, tmp_file_dir='./tmp_out', create_data=True, filter_noise=True,
                 bpp_root='./data/bpps',
                 nonbond_as_node_feature=True, consider_loop_type=True, consider_seqdist=True):

        self.file_in = file_in
        self.save_dir = tmp_file_dir
        self.filter_noise = filter_noise
        self.bpps = bpp_root
        self.nonbond_nodefeature = nonbond_as_node_feature
        self.consider_loop_type = consider_loop_type
        self.seqdist_nodefeature = consider_seqdist

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if create_data:
            self._create_data_2()
            with open('{}/toc.csv'.format(self.save_dir), 'w') as fout:
                print ('total_processed, {}'.format(self.total_processed), file=fout)
        else:
            with open('{}/toc.csv'.format(self.save_dir), 'r') as fin:
                self.total_processed = int(fin.readline().split(',')[1])
                print (self.total_processed)
            print ('Data not created, only loaded. Noise filter and node feature configurations implied.')

        super(RNADataset, self).__init__()

    def get_raw_data(self, idx):
        '''Convenience function for debugging. Provides raw data for the given IDX, corresponding to the IDX
        from the __getitem__ function of the dataset

        '''
        df = pd.read_json(self.file_in, lines=True)
        if self.filter_noise:
            df = df.loc[df['SN_filter'] == 1]
        return df.iloc[idx]

    def _create_data_2(self):

        df = pd.read_json(self.file_in, lines=True)
        counter = 0
        for id_label, node_vec_x, node_vec_y, n_scored, sn_val, sn_filter in vanilla_props_(df):

            # Ignore all data that does not meet signal-noise threshold, provided such flag is given (not the case for test data)
            if not sn_filter is None:
                if self.filter_noise:
                    if sn_filter != 1:
                        continue

            x = torch.tensor(node_vec_x, dtype=torch.float)

            # Add ground-truth node features
            if not node_vec_y is None:
                y = torch.tensor(node_vec_y, dtype=torch.float)
            else:
                y = None

            # Create mask for which nodes are to be predicted
            train_mask = torch.zeros(x.shape[0], dtype=torch.bool)
            train_mask[:n_scored] = True

            torch.save({'id_label': id_label,
                        'x': x, 'y': y,
                        'train_mask': train_mask,
                        'sn_filter': sn_filter},
                       '{}/{}_{}.pt'.format(self.save_dir, ALL_DATA_NAME, counter))
            counter += 1

        self.total_processed = counter

    def _create_data(self):
        '''Creates the data from the raw data

        '''
        df = pd.read_json(self.file_in, lines=True)

        counter = 0
        for id_label, node_props_x, node_props_y, e_index_1, e_index_2, edge_props, n_scored, sn_val, sn_filter in graph_props_x_(df, self.consider_loop_type):

            # Ignore all data that does not meet signal-noise threshold, provided such flag is given (not the case for test data)
            if not sn_filter is None:
                if self.filter_noise:
                    if sn_filter != 1:
                        continue

            # Add feature of incoming edges to node as a node feature
            if self.nonbond_nodefeature:
                node_nbond_feature = create_node_nbond(node_props_x, e_index_1, e_index_2, edge_props)
                if self.seqdist_nodefeature:
                    node_nbond_feature = create_seqdist_nodefeature(node_nbond_feature, e_index_1, e_index_2)
                x = torch.tensor(node_nbond_feature, dtype=torch.float)
            else:
                x = torch.tensor(node_props_x, dtype=torch.float)

            # Add ground-truth node features
            if not node_props_y is None:
                y = torch.tensor(node_props_y, dtype=torch.float)
            else:
                y = None

            # Create mask for which nodes are to be predicted
            train_mask = torch.zeros(x.shape[0], dtype=torch.bool)
            train_mask[:n_scored] = True

            torch.save({'id_label' : id_label,
                        'x' : x, 'y' : y,
                        'train_mask' : train_mask,
                        'sn_filter' : sn_filter},
                       '{}/{}_{}.pt'.format(self.save_dir, ALL_DATA_NAME, counter))
            counter += 1

        self.total_processed = counter

    def __len__(self):
        return self.total_processed

    def __getitem__(self, idx):
        return torch.load('{}/{}_{}.pt'.format(self.save_dir, ALL_DATA_NAME, idx))

    @property
    def n_pred_dim(self):
        try:
            return self.__getitem__(0)['y'].shape[-1]
        except KeyError:
            raise RuntimeError('Predicted dimension not available from data. Are you using test data?')

    @property
    def n_node_dim(self):
        return self.__getitem__(0)['x'].shape[-1]


