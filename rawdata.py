'''Raw data processor

'''
import pandas as pd
import numpy as np

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

def create_node_features(seq, loop=None):
    if loop is None:
        return [RNA_ONEHOT[letter_nuc] for letter_nuc in seq]
    else:
        return [RNA_ONEHOT[letter_nuc] + LOOP_ONEHOT[letter_loop] for letter_nuc, letter_loop in zip(seq, loop)]

def create_node_y_features(df):
    ret = []
    df_new = pd.DataFrame([df.loc[label] for label in Y_NAMES], index=Y_NAMES)
    for column in df_new:
        ret.append(df_new[column].to_list())

    return ret

def create_node_nbond(props_x, e1, e2, edge_props):

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

def create_seqdist_nodefeature(props_x, e1, e2):

    val = {}
    for ee1, ee2 in zip(e1, e2):
        dee = abs(ee1 - ee2)
        if dee > 1:
            val[ee1] = dee

    ret = []
    for node_id, node_prop in enumerate(props_x):
        if node_id in val:
            to_add = [val[node_id]]
        else:
            to_add = [0]
        ret.append(node_prop + to_add)

    return ret

def _rna_bond(n1, n2):
    [m1, m2] = sorted([n1, n2])
    if m1 == 'A' and m2 == 'U':
        return [1.0,0.0,0.0]
    elif m1 == 'C' and m2 == 'G':
        return [0.0,1.0,0.0]
    else:
        return [0.0,0.0,1.0]

def _seq_dist_bin(d1, d2):
    dd = abs(d1 - d2)
    if dd <= 6:
        return [1.0,0.0,0.0,0.0]
    elif dd >= 7 and dd <= 12:
        return [0.0,1.0,0.0,0.0]
    elif dd >=13 and dd <=18:
        return [0.0,0.0,1.0,0.0]
    else:
        return [0.0,0.0,0.0,1.0]

def create_graph_matrix_bpp(id, seq):

    array = np.load('/Users/andersohrn/PycharmProjects/rna_graph/data/bpps/{}.npy'.format(id))
    feature_vecs = []
    for k_nuc, (nuc, pairprob) in enumerate(zip(seq, array)):
        nuc_vec = RNA_ONEHOT[nuc]
        bond_vecs = []
        bond_dists = []
        for k_pp, pp in enumerate(pairprob):
            if pp > 0.0:
                bond_vecs.append([x * pp for x in _rna_bond(nuc, seq[k_pp])])
                bond_dists.append([x * pp for x in _seq_dist_bin(k_nuc, k_pp)])

        bond_vec = [0.0, 0.0, 0.0]
        for bv in bond_vecs:
            bond_vec[0] += bv[0]
            bond_vec[1] += bv[1]
            bond_vec[2] += bv[2]
        seq_vec = [0.0, 0.0, 0.0, 0.0]
        for sv in bond_dists:
            seq_vec[0] += sv[0]
            seq_vec[1] += sv[1]
            seq_vec[2] += sv[2]
            seq_vec[3] += sv[3]

        feature_vecs.append(nuc_vec + bond_vec + seq_vec)
    return feature_vecs

def vanilla_props_(df):
    for row_id, row in df.iterrows():
        f_vecs = create_graph_matrix_bpp(row.id, row.sequence)

        if Y_NAMES[0] in row.index:
            node_features_y = create_node_y_features(row.loc[Y_NAMES])
        else:
            node_features_y = None

        if 'signal_to_noise' in row.index:
            sn_val = row['signal_to_noise']
            sn_filter = row['SN_filter']
        else:
            sn_val = None
            sn_filter = None

        yield row['id'], \
              f_vecs, node_features_y, \
              row['seq_scored'], \
              sn_val, sn_filter

def graph_props_x_(df, consider_loop_type):

    for row_id, row in df.iterrows():
        if consider_loop_type:
            node_features_x = create_node_features(row.sequence, row.predicted_loop_type)
        else:
            node_features_x = create_node_features(row.sequence)

        #e_index_1, e_index_2, edge_prop = create_graph_matrix(row.structure, row.sequence)
        print (row.structure)
        e_index_1, e_index_2, edge_prop = create_graph_matrix_bpp(row.id, row.sequence)
        raise RuntimeError

        if Y_NAMES[0] in row.index:
            node_features_y = create_node_y_features(row.loc[Y_NAMES])

        else:
            node_features_y = None

        if 'signal_to_noise' in row.index:
            sn_val = row['signal_to_noise']
            sn_filter = row['SN_filter']

        else:
            sn_val = None
            sn_filter = None

        yield row['id'], \
              node_features_x, node_features_y, \
              e_index_1, e_index_2, edge_prop, \
              row['seq_scored'], \
              sn_val, sn_filter