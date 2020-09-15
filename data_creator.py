'''Bla bla for now

'''
import pandas as pd
import numpy as np
from torch_geometric.data import Data

TRAIN_DATA_PATH = './data/train.json'
TEST_DATA_PATH = './data/test.json'

RNA_ONEHOT = {'G' : [1,0,0,0], 'C' : [0,1,0,0], 'A' : [0,0,1,0], 'U' : [0,0,0,1]}
BOND_ONEHOT = {('A','U') : [0,1,0,0], ('C','G') : [0,0,1,0], 'atypical' : [0,0,0,1], 'covalent' : [1,0,0,0]}

def create_node_features(seq):
    return [RNA_ONEHOT[letter] for letter in seq]

def create_graph_matrix(structure, seq):

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

    node_nonb2.pop(-1)
    edge_property.pop(-1)

    return node_nonb1, node_nonb2, edge_property

def create_inp_graphs(f_train, f_test):

    train = pd.read_json(f_train, lines=True)
    test = pd.read_json(f_test, lines=True)

    for row_id, row in train.iterrows():
        print (row)
        print (row.sequence)
        node_features = create_node_features(row.sequence)
        print (node_features)
        e_index_1, e_index_2, edge_prop = create_graph_matrix(row.structure, row.sequence)
        print (e_index_1)
        print (e_index_2)
        print (edge_prop)
        raise RuntimeError
    print (train)
    print (train.columns)

if __name__ == '__main__':
    create_inp_graphs(TRAIN_DATA_PATH, TEST_DATA_PATH)