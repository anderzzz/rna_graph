import sys
import torch
from torch.utils.data import DataLoader

from rna_dataset_vanilla import RNADataset
from model import Deep1D_incept, Res1D
from rawdata import Y_NAMES

def format_for_submission_(x, label, dec=6):

    for res_id, res_data in enumerate(x.squeeze(0)):
        col_0 = '{}_{}'.format(label, res_id)
        col_x = ','.join([str(round(x, dec)) for x in res_data.tolist()])
        yield '{},{}'.format(col_0, col_x)

def eval_vanilla(model, model_state, data, eval_save):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.load_state_dict(torch.load(model_state)['model_state_dict'])
    model.eval()

    print ('id_seqpos,' + ','.join(Y_NAMES), file=eval_save)
    for k_batch, data_batch in enumerate(data):
        data_x = data_batch['x'].unsqueeze(0)
        data_x = data_x.to(device)

        out = model(data_x)
        for csv_row in format_for_submission_(out, data_batch['id_label']):
            print (csv_row, file=eval_save)


def print_dict(dd, fout=sys.stdout):
    for key, value in dd.items():
        print ('{} : {}'.format(key, value), file=fout)

def run4():
    rna_dataset_vanilla = {'file_in': './data/test.json',
                           'tmp_file_dir' : './tmp_out_eval',
                           'filter_noise': True,
                           'nonbond_as_node_feature': True,
                           'consider_loop_type': False,
                           'consider_seqdist': True,
                           'create_data': True}
    print_dict(rna_dataset_vanilla)
    rna_data = RNADataset(**rna_dataset_vanilla)
    print (rna_data)
    print (len(rna_data))
    ddd = DataLoader(rna_data, batch_size=None, shuffle=False)

    deeper_1d = {'n_init_features': [64, 128, 128, 128],
                 'n_hidden_channels': 64,
                 'drop_rate': 0.05,
                 'n_blocks': 16,
                 'glu_act': True,
                 'n_in_channels': rna_data.n_node_dim,
                 'n_out_channels': 5}
    print_dict(deeper_1d)
    model = Deep1D_incept(**deeper_1d)

    with open('out_eval.csv', 'w') as fout:
        eval_vanilla(model, './trainer_save_9.pt', ddd, fout)

def run5():
    rna_dataset_vanilla = {'file_in': './data/test.json',
                           'tmp_file_dir' : './tmp_out_eval',
                           'filter_noise': True,
                           'nonbond_as_node_feature': True,
                           'consider_loop_type': False,
                           'consider_seqdist': True,
                           'create_data': True}
    print_dict(rna_dataset_vanilla)
    rna_data = RNADataset(**rna_dataset_vanilla)
    print (rna_data)
    print (len(rna_data))
    ddd = DataLoader(rna_data, batch_size=None, shuffle=False)

    deeper_1d = {'in_channels' : rna_data.n_node_dim,
                 'out_channels' : 5,
                 'nblocks': 20,
                 'hidden_progression' : [256, 256, 256, 256,
                                         256, 256, 256, 256,
                                         512, 512, 512, 512,
                                         512, 512, 512, 512,
                                         1024, 1024, 1024, 1024]}
    print_dict(deeper_1d)
    model = Res1D(**deeper_1d)
    with open('out_eval.csv', 'w') as fout:
        eval_vanilla(model, './trainer_save_7.pt', ddd, fout)

run4()
