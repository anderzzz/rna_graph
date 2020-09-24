'''Main for Graph training

'''
import sys
import numpy as np
from numpy.random import shuffle
import copy

import torch
from torch import optim
from torch.utils.data import Subset
import torch_geometric

from model import DeeperGCN, DenseDeep1D, DenseDeep1D_incept, Deep1D_incept
from rna_dataset_graph import RNADatasetGraph
from rna_dataset_vanilla import RNADataset
from loss import MCRMSELoss

def seed_all(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def eval_vanilla(model, model_state, dataloaders, eval_save):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.load_state_dict(torch.load(model_state)['model_state_dict'])
    model.eval()

    results = []
    for data_batch in dataloaders:
        out = model(data_batch)
        results.append(out)
        raise RuntimeError

def trainer_vanilla(model, dataloaders,
                    n_epochs, lr_init, scheduler_step_size, scheduler_gamma,
                    trainer_save):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    optimizer = optim.SGD(model.parameters(), lr=lr_init, momentum=0.9)
    #optimizer = optim.Adam(model.parameters(), lr=lr_init)
    criterion = MCRMSELoss()
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                 step_size=scheduler_step_size,
                                                 gamma=scheduler_gamma)
    best_loss = 1e20
    best_model = copy.deepcopy(model.state_dict())

    for k_epoch in range(n_epochs):

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            for k_batch, data_batch in enumerate(dataloaders[phase]):
                #print (k_batch)
                data_x = data_batch['x']
                data_x = data_x.to(device)

                optimizer.zero_grad()
                out = model(data_x)

                y_to_score = out[data_batch['train_mask']]
                y_ref = torch.flatten(data_batch['y'], 0, 1)
                loss = criterion(y_to_score, y_ref)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()

            if phase == 'train':
                exp_lr_scheduler.step()

            running_loss = running_loss / (k_batch + 1)
            print ('Epoch {}, phase {}, running loss: {}'.format(k_epoch, phase, running_loss))

            if phase == 'test' and running_loss < best_loss:
                best_loss = running_loss
                best_model = copy.deepcopy(model.state_dict())
                torch.save({'model_state_dict' : best_model,
                            'loss' : best_loss},
                           trainer_save + '.pt')

def trainer_graph(model, dataloaders,
                  n_epochs, lr_init, scheduler_step_size, scheduler_gamma,
                  trainer_save):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    optimizer = optim.SGD(model.parameters(), lr=lr_init, momentum=0.9)
    #optimizer = optim.Adam(model.parameters(), lr=lr_init)
    criterion = MCRMSELoss()
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                 step_size=scheduler_step_size,
                                                 gamma=scheduler_gamma)
    best_loss = 1e20
    best_model = copy.deepcopy(model.state_dict())

    for k_epoch in range(n_epochs):

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            for k_batch, data_batch in enumerate(dataloaders[phase]):
                #print (k_batch)
                data_batch = data_batch.to(device)

                optimizer.zero_grad()
                out = model(data_batch)

                y_to_score = out[data_batch.train_mask]
                loss = criterion(y_to_score, data_batch.y)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()

            if phase == 'train':
                exp_lr_scheduler.step()

            running_loss = running_loss / (k_batch + 1)
            print ('Epoch {}, phase {}, running loss: {}'.format(k_epoch, phase, running_loss))

            if phase == 'test' and running_loss < best_loss:
                best_loss = running_loss
                best_model = copy.deepcopy(model.state_dict())
                torch.save({'model_state_dict' : best_model,
                            'loss' : best_loss},
                           trainer_save + '.pt')

def print_dict(dd, fout=sys.stdout):
    for key, value in dd.items():
        print ('{} : {}'.format(key, value), file=fout)

def run1():

    seed_all()
    rna_dataset_graph = {'file_in' : './data/train.json',
                         'filter_noise' : True,
                         'nonbond_as_node_feature' : False,
                         'consider_loop_type' : True}
    print_dict(rna_dataset_graph)
    rna_data = RNADatasetGraph(**rna_dataset_graph)

    frac_test = 0.1
    batch_size = 16
    all_inds = list(range(len(rna_data)))
    shuffle(all_inds)
    test_inds = all_inds[:int(frac_test * len(rna_data))]
    train_inds = all_inds[int(frac_test * len(rna_data)):]

    rna_structure_loaders = {'train' : torch_geometric.data.DataLoader(Subset(rna_data, train_inds),
                                                                       batch_size=batch_size, shuffle=True),
                             'test' : torch_geometric.data.DataLoader(Subset(rna_data, test_inds),
                                                                      batch_size=batch_size, shuffle=False)}

    deeper_gcn = {'n_hidden_channels' : 128,
                  'num_layers' : 12,
                  'n_in_edge_props' : rna_data.n_edge_dim,
                  'n_in_node_props' : rna_data.n_node_dim,
                  'n_out_node_props' : rna_data.n_pred_dim}
    print_dict(deeper_gcn)
    model = DeeperGCN(**deeper_gcn)

    opt_params = {'n_epochs' : 40,
                  'lr_init' : 0.01,
                  'scheduler_step_size' : 20,
                  'scheduler_gamma' : 0.1,
                  'trainer_save' : './trainer_0_save'}
    print_dict(opt_params)
    trainer_graph(model, rna_structure_loaders, **opt_params)

def run3():

    trainer_out_prefix = 'trainer_save'
    f_params = open('{}_params.txt'.format(trainer_out_prefix), 'w')

    seed_all(111)
    rna_dataset_vanilla = {'file_in': './data/train.json',
                           'filter_noise': True,
                           'nonbond_as_node_feature': True,
                           'consider_loop_type': False,
                           'consider_seqdist': True,
                           'create_data' : True}
    print_dict(rna_dataset_vanilla, fout=f_params)
    rna_data = RNADataset(**rna_dataset_vanilla)

    frac_test = 0.1
    batch_size = 16
    all_inds = list(range(len(rna_data)))
    shuffle(all_inds)
    test_inds = all_inds[:int(frac_test * len(rna_data))]
    train_inds = all_inds[int(frac_test * len(rna_data)):]

    rna_structure_loaders = {'train': torch.utils.data.DataLoader(Subset(rna_data, train_inds),
                                                                  batch_size=batch_size, shuffle=True),
                             'test': torch.utils.data.DataLoader(Subset(rna_data, test_inds),
                                                                 batch_size=batch_size, shuffle=False)}

    deeper_1d = {'n_init_features': [16,32,32,16],
                 'n_hidden_channels': 32,
                 'drop_rate': 0.0,
                 'n_blocks': 15,
                 'glu_act': True,
                 'n_in_channels': rna_data.n_node_dim,
                 'n_out_channels': rna_data.n_pred_dim}
    print_dict(deeper_1d, fout=f_params)
    #model = DenseDeep1D(**deeper_1d)
    model = Deep1D_incept(**deeper_1d)

    opt_params = {'n_epochs': 40,
                  'lr_init': 0.01,
                  'scheduler_step_size': 10,
                  'scheduler_gamma': 0.2,
                  'trainer_save': trainer_out_prefix}
    print_dict(opt_params, fout=f_params)
    f_params.flush()
    f_params.close()

    trainer_vanilla(model, rna_structure_loaders, **opt_params)

    print ('Done!')

run3()