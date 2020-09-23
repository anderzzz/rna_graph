import sys
import torch
from torch.utils.data import DataLoader

from rna_dataset_vanilla import RNADataset
from model import Deep1D_incept
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

    print ('id_seqpos,' + ','.join(Y_NAMES))
    for k_batch, data_batch in enumerate(data):
        data_x = data_batch['x'].unsqueeze(0)
        data_x = data_x.to(device)

        out = model(data_x)
        for csv_row in format_for_submission_(out, data_batch['id_label']):
            print (csv_row)

        raise RuntimeError

def print_dict(dd, fout=sys.stdout):
    for key, value in dd.items():
        print ('{} : {}'.format(key, value), file=fout)

def run4():
    rna_dataset_vanilla = {'file_in': './data/test.json',
                           'tmp_file_dir' : './tmp_out_eval',
                           'filter_noise': True,
                           'nonbond_as_node_feature': True,
                           'consider_loop_type': False,
                           'create_data': True}
    print_dict(rna_dataset_vanilla)
    rna_data = RNADataset(**rna_dataset_vanilla)
    print (rna_data)
    print (len(rna_data))
    ddd = DataLoader(rna_data, batch_size=None, shuffle=False)

    deeper_1d = {'n_init_features': [16, 32, 0, 0],
                 'n_hidden_channels': 64,
                 'drop_rate': 0.0,
                 'n_blocks': 10,
                 'glu_act': True,
                 'n_in_channels': rna_data.n_node_dim,
                 'n_out_channels': 5}
    print_dict(deeper_1d)
    model = Deep1D_incept(**deeper_1d)

    eval_vanilla(model, './trainer_0_save_a.pt', ddd, 'tmp_eval.csv')
    raise RuntimeError

run4()
