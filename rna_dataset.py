'''Creation of Graph Dataset from RNA rawdata

'''
import os
import pandas as pd

import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import is_undirected

'''Encode the nucleotide as one-hot vector'''
RNA_ONEHOT = {'G' : [1,0,0,0], 'C' : [0,1,0,0], 'A' : [0,0,1,0], 'U' : [0,0,0,1]}

'''Encode pair contact as one-hot vector. Put the atypical contact types in single category'''
BOND_ONEHOT = {('A','U') : [0,1,0,0], ('C','G') : [0,0,1,0], 'atypical' : [0,0,0,1], 'covalent' : [1,0,0,0]}

'''Encode non-bonded residue or not, high resolution'''
NBOND_ONEHOT = {'.' : [1,0,0,0], 'low' : [0,1,0,0], 'high':[0,0,1,0], 'atypical' : [0,0,0,1]}

'''Encode loop type as one-hot vector.'''
LOOP_ONEHOT = {'E' : [1,0,0,0,0,0,0], 'S' : [0,1,0,0,0,0,0], 'H' : [0,0,1,0,0,0,0], 'B' : [0,0,0,1,0,0,0],
               'X' : [0,0,0,0,1,0,0], 'I' : [0,0,0,0,0,1,0], 'M' : [0,0,0,0,0,0,1]}

'''Names for the properties to predict'''
Y_NAMES = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

'''Root of file name to save graph data in'''
ALL_DATA_NAME = 'graphs_pytorch'

class RNADataset(Dataset):
    '''Bla bla for now

    '''
    def __init__(self, file_in, tmp_file_dir='./tmp_out', filter_noise=True,
                 nonbond_as_node_feature=True, consider_loop_type=True):

        self.file_in = file_in
        self.save_dir = tmp_file_dir
        self.filter_noise = filter_noise
        self.nonbond_nodefeature = nonbond_as_node_feature
        self.consider_loop_type = consider_loop_type

        self._create_data()

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        super(RNADataset, self).__init__()

    def get_raw_data(self, idx):

        df = pd.read_json(self.file_in, lines=True)
        if self.filter_noise:
            df = df.loc[df['SN_filter'] == 1]
        return df.iloc[idx]

    def _create_data(self):

        # Include rows that are explicitly provided
        df = pd.read_json(self.file_in, lines=True)

        counter = 0
        for node_props_x, node_props_y, e_index_1, e_index_2, edge_props, n_scored, sn_val, sn_filter in self._graph_props_x_(df):
            edge_index = torch.tensor([e_index_1, e_index_2], dtype=torch.long)
            if not is_undirected(edge_index):
                raise RuntimeError('Directed graph created, should be impossible')

            # Ignore all data that does not meet signal-noise threshold
            if self.filter_noise:
                if sn_filter != 1:
                    continue

            # Add feature of incoming edges to node as a node feature
            if self.nonbond_nodefeature:
                node_nbond_feature = self._create_node_nbond(node_props_x, e_index_1, e_index_2, edge_props)
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

    def _create_node_features(self, seq, loop):
        if self.consider_loop_type:
            return [RNA_ONEHOT[letter_nuc] + LOOP_ONEHOT[letter_loop] for letter_nuc, letter_loop in zip(seq, loop)]
        else:
            return [RNA_ONEHOT[letter_nuc] for letter_nuc in seq]

    def _create_node_y_features(self, df):
        ret = []
        df_new = pd.DataFrame([df.loc[label] for label in Y_NAMES], index=Y_NAMES)
        for column in df_new:
            ret.append(df_new[column].to_list())

        return ret

    def _create_graph_matrix(self, structure, seq):

        bond_id = 0
        bond_ids = []
        for structure_letter in structure:

            if structure_letter == '.':
                bond_ids.append(None)
            elif structure_letter == '(':
                bond_id += 1
                bond_ids.append(bond_id)
            elif structure_letter == ')':
                bond_ids.append(-1 * bond_id)
                bond_id -= 1

            if bond_id < 0:
                raise RuntimeError('bond_id < 0, should not be possible in RNA structure definition')

        node_nonb1 = []
        node_nonb2 = []
        edge_property = []
        for node_id, bond_id in enumerate(bond_ids):
            if (bond_id is None) or (bond_id < 0):
                pass
            else:
                close_bracket = [n for n, b in enumerate(bond_ids[node_id:]) if b == -1 * bond_id][0]
                node2_id = close_bracket + node_id

                try:
                    b_type = BOND_ONEHOT[tuple(sorted([seq[node_id], seq[node2_id]]))]
                except:
                    b_type = BOND_ONEHOT['atypical']

                node_nonb1.append(node_id)
                node_nonb2.append(node2_id)
                edge_property.append(b_type)

            node_nonb1.append(node_id)
            node_nonb2.append(node_id + 1)
            edge_property.append(BOND_ONEHOT['covalent'])

        node_nonb1.pop(-1)
        node_nonb2.pop(-1)
        edge_property.pop(-1)

        # Make the graph undirected. Since no self-loops exists, this is done by simple transpose
        node_nonb1_ret = node_nonb1 + node_nonb2
        node_nonb2_ret = node_nonb2 + node_nonb1
        edge_property_ret = edge_property + edge_property

        return node_nonb1_ret, node_nonb2_ret, edge_property_ret

    def _create_node_nbond(self, props_x, e1, e2, edge_props):

        ret = []
        for node_id, node_prop in enumerate(props_x):
            k_edges = [ind_e for ind_e , ke in enumerate(e1) if ke == node_id]
            val = NBOND_ONEHOT['.']
            for k_edge in k_edges:
                if edge_props[k_edge] == BOND_ONEHOT[('A', 'U')]:
                    val = NBOND_ONEHOT['low']
                elif edge_props[k_edge] == BOND_ONEHOT[('C', 'G')]:
                    val = NBOND_ONEHOT['high']
                elif edge_props[k_edge] == BOND_ONEHOT['atypical']:
                    val = NBOND_ONEHOT['atypical']
            ret.append(node_prop + val)

        return ret

    def _graph_props_x_(self, df):

        for row_id, row in df.iterrows():
            node_features_x = self._create_node_features(row.sequence, row.predicted_loop_type)
            e_index_1, e_index_2, edge_prop = self._create_graph_matrix(row.structure, row.sequence)

            if Y_NAMES[0] in row.index:
                node_features_y = self._create_node_y_features(row.loc[Y_NAMES])

            else:
                node_features_y = None

            if 'signal_to_noise' in row.index:
                sn_val = row['signal_to_noise']
                sn_filter = row['SN_filter']

            else:
                sn_val = None
                sn_filter = None

            yield node_features_x, node_features_y, e_index_1, e_index_2, edge_prop, row['seq_scored'], sn_val, sn_filter

    def len(self):
        return self.total_processed

    def get(self, idx):
        return torch.load('{}/{}_{}.pt'.format(self.save_dir, ALL_DATA_NAME, idx))

    @property
    def n_pred_dim(self):
        return self.get(0).y.shape[-1]

def test1():
    rna_data = RNADataset('./data/train.json', './tmp_out', True, True, True)
    print (rna_data)
    print (rna_data[0])

    rna_data = RNADataset('./data/train.json', './tmp_out', True, True, False)
    print (rna_data)
    print (rna_data[5])
    print (rna_data[5].x)
    print (rna_data.get_raw_data(5).structure)
    print (rna_data.get_raw_data(5).sequence)

#test1()