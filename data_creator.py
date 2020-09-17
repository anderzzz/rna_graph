'''Bla bla for now

'''
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected

'''Encode the nucleotide as one-hot vector'''
RNA_ONEHOT = {'G' : [1,0,0,0], 'C' : [0,1,0,0], 'A' : [0,0,1,0], 'U' : [0,0,0,1]}

'''Encode pair contact as one-hot vector. Put the atypical contact types in single category'''
BOND_ONEHOT = {('A','U') : [0,1,0,0], ('C','G') : [0,0,1,0], 'atypical' : [0,0,0,1], 'covalent' : [1,0,0,0]}

'''Names for the properties to predict'''
Y_NAMES = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

'''Root of file name to save graph data in'''
GRAPH_NAME = 'graph_pytorch'

def create_node_features(seq):
    return [RNA_ONEHOT[letter] for letter in seq]

def create_node_y_features(df):

    ret = []
    df_new = pd.DataFrame([df.loc[label] for label in Y_NAMES], index=Y_NAMES)
    for column in df_new:
        ret.append(df_new[column].to_list())

    return ret

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

    node_nonb1.pop(-1)
    node_nonb2.pop(-1)
    edge_property.pop(-1)

    # Make the graph undirected. Since no self-loops exists, this is done by simple transpose
    node_nonb1_ret = node_nonb1 + node_nonb2
    node_nonb2_ret = node_nonb2 + node_nonb1
    edge_property_ret = edge_property + edge_property

    return node_nonb1_ret, node_nonb2_ret, edge_property_ret

def graph_props_x_(df):

    for row_id, row in df.iterrows():
        node_features_x = create_node_features(row.sequence)
        e_index_1, e_index_2, edge_prop = create_graph_matrix(row.structure, row.sequence)

        if Y_NAMES[0] in row.index:
            node_features_y = create_node_y_features(row.loc[Y_NAMES])

        else:
            node_features_y = None

        #print (row['deg_Mg_pH10'])
        #print (','.join(row['reactivity']))

        yield node_features_x, node_features_y, e_index_1, e_index_2, edge_prop, row['seq_scored']

def create_inp_graphs(f_in, f_out_dir):

    df = pd.read_json(f_in, lines=True)

    counter = 0
    for node_props_x, node_props_y, e_index_1, e_index_2, edge_props, n_scored in graph_props_x_(df):
        edge_index = torch.tensor([e_index_1, e_index_2], dtype=torch.long)
        if not is_undirected(edge_index):
            raise RuntimeError('Directed graph created, should be impossible')

        x = torch.tensor(node_props_x, dtype=torch.float)
        edge_x = torch.tensor(edge_props, dtype=torch.float)

        if not node_props_y is None:
            y = torch.tensor(node_props_y, dtype=torch.float)
        else:
            y = None

        data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_x,
                    n_seq_score=n_scored, n_seq_total=x.shape[0])
        torch.save(data, '{}/{}_{}.pt'.format(f_out_dir, GRAPH_NAME, counter))

        counter += 1

if __name__ == '__main__':
    create_inp_graphs('./data/train.json', './data_processed_train')
    create_inp_graphs('./data/test.json', './data_processed_test')