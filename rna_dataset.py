'''Bla bla for now

'''
import os
from os import listdir
from itertools import compress
import numpy as np
import shutil
import pandas as pd

import torch
from torch_geometric.data import Dataset
from torch_geometric.utils import is_undirected

'''Encode the nucleotide as one-hot vector'''
RNA_ONEHOT = {'G' : [1,0,0,0], 'C' : [0,1,0,0], 'A' : [0,0,1,0], 'U' : [0,0,0,1]}

'''Encode pair contact as one-hot vector. Put the atypical contact types in single category'''
BOND_ONEHOT = {('A','U') : [0,1,0,0], ('C','G') : [0,0,1,0], 'atypical' : [0,0,0,1], 'covalent' : [1,0,0,0]}

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
    def __init__(self, file_in, tmp_file_dir='./tmp_out', filter_noise=True):

        self.file_in = file_in
        self.save_dir = tmp_file_dir
        self.filter_noise = filter_noise

        self._create_data()

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        super(RNADataset, self).__init__()

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

            x = torch.tensor(node_props_x, dtype=torch.float)
            edge_x = torch.tensor(edge_props, dtype=torch.float)

            if not node_props_y is None:
                y = torch.tensor(node_props_y, dtype=torch.float)
            else:
                y = None

            train_mask = torch.zeros(x.shape[0], dtype=torch.bool)
            train_mask[:n_scored] = True
            data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_x,
                        train_mask=train_mask, sn_val=sn_val)
            torch.save(data, '{}/{}_{}.pt'.format(self.save_dir, ALL_DATA_NAME, counter))
            counter += 1

        self.total_processed = counter

    def _create_node_features(self, seq, loop):
        return [RNA_ONEHOT[letter_nuc] + LOOP_ONEHOT[letter_loop] for letter_nuc, letter_loop in zip(seq, loop)]

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

            yield node_features_x, node_features_y, e_index_1, e_index_2, edge_prop, row['seq_scored'], sn_val, sn_filter

    def len(self):
        return self.total_processed

    def get(self, idx):
        return torch.load('{}/{}_{}.pt'.format(self.save_dir, ALL_DATA_NAME, idx))

    @property
    def n_pred_dim(self):
        return self.get(0).y.shape[-1]


from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data


def build_matrix(couples, size):
    mat = np.zeros((size, size))

    for i in range(size):  # neigbouring bases are linked as well
        if i < size - 1:
            mat[i, i + 1] = 1
        if i > 0:
            mat[i, i - 1] = 1

    for i, j in couples:
        mat[i, j] = 2
        mat[j, i] = 2

    return mat

def seq2nodes(sequence,loops,structures):
    type_dict={'A':0,'G':1,'U':2,'C':3}
    loop_dict={'S':0,'M':1,'I':2,'B':3,'H':4,'E':5,'X':6}
    struct_dict={'.':0,'(':1,')':2}
    # 4 types, 7 structural types
    nodes=np.zeros((len(sequence),4+7+3))
    for i,s in enumerate(sequence):
        nodes[i,type_dict[s]]=1
    for i,s in enumerate(loops):
        nodes[i,4+loop_dict[s]]=1
    for i,s in enumerate(structures):
        nodes[i,11+struct_dict[s]]=1
    return nodes

def get_couples(structure):
    """
    For each closing parenthesis, I find the matching opening one and store their index in the couples list.
    The assigned list is used to keep track of the assigned opening parenthesis
    """
    opened = [idx for idx, i in enumerate(structure) if i == '(']
    closed = [idx for idx, i in enumerate(structure) if i == ')']

    assert len(opened) == len(closed)

    assigned = []
    couples = []

    for close_idx in closed:
        for open_idx in opened:
            if open_idx < close_idx:
                if open_idx not in assigned:
                    candidate = open_idx
            else:
                break
        assigned.append(candidate)
        couples.append([candidate, close_idx])
        assigned.append(close_idx)
        couples.append([close_idx, candidate])

    assert len(couples) == 2 * len(opened)

    return couples

class MyOwnDataset(InMemoryDataset):
    def __init__(self, root='', train=True, public=True, ids=None, filter_noise=False, transform=None,
                 pre_transform=None):
        try:
            shutil.rmtree('./' + root)
        except:
            print("doesn't exist")
        self.train = train
        if self.train:
            self.data_dir = './data/train.json'
        else:
            self.data_dir = './data/test.json'
        #self.bpps_dir = '../input/stanford-covid-vaccine/bpps/'
        self.df = pd.read_json(self.data_dir, lines=True)
        if filter_noise:
            print (self.df.shape)
            self.df = self.df[self.df.SN_filter == 1]
            print (self.df.shape)
        if ids is not None:
            self.df = self.df[self.df['index'].isin(ids)]
        if public:
            self.df = self.df.query("seq_length == 107")
        else:
            self.df = self.df.query("seq_length == 130")
        self.target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C', 'deg_pH10', 'deg_50C']

        super(MyOwnDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        for idx in range(len(self.df)):
            structure = self.df['structure'].iloc[idx]
            sequence = self.df['sequence'].iloc[idx]
            loops = self.df['predicted_loop_type'].iloc[idx]
            # 2 x edges
            matrix = build_matrix(get_couples(structure), len(sequence))
            # nodes x features
            id_ = self.df['id'].iloc[idx]
            #bpps = np.load(self.bpps_dir + id_ + '.npy')
            edge_index = np.stack(np.where((matrix) != 0))
            node_attr = seq2nodes(sequence, loops, structure)
            #node_attr = np.append(node_attr, bpps.sum(axis=1, keepdims=True), axis=1)
            edge_attr = np.zeros((edge_index.shape[1], 4))
            edge_attr[:, 0] = (matrix == 2)[edge_index[0, :], edge_index[1, :]]
            edge_attr[:, 1] = (matrix == 1)[edge_index[0, :], edge_index[1, :]]
            edge_attr[:, 2] = (matrix == -1)[edge_index[0, :], edge_index[1, :]]
            #edge_attr[:, 3] = bpps[edge_index[0, :], edge_index[1, :]]
            # targets
            # padded_targets=np.zeros((130,5))
            if self.train:
                print (self.df[self.target_cols].iloc[idx])
                raise RuntimeError
                targets = np.stack(self.df[self.target_cols].iloc[idx]).T
            else:
                targets = np.zeros((130, 5))
            x = torch.from_numpy(node_attr)
            y = torch.from_numpy(targets)
            edge_attr = torch.from_numpy(edge_attr)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data.train_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
            data.train_mask[:68] = 1
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def seed_all(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def test1():
    dd = RNADataset('./data_processed_train', [0,1,2,5])
    for q in dd:
        print (q)

def test2():
    seed_all()
    all_data = pd.read_json('./data/train.json', lines=True)
    all_data.head(5)
    all_ids = np.arange(len(all_data))
    np.random.shuffle(all_ids)
    train_ids, val_ids = np.split(all_ids, [int(round(0.9 * len(all_ids), 0))])

    train_dataset = MyOwnDataset(ids=train_ids, root='train', filter_noise=True)
    print (train_dataset)
    print (train_dataset[0])
    print (len(train_dataset))
    val_dataset = MyOwnDataset(ids=val_ids, root='val', filter_noise=True)

#test2()