'''Main

'''
import copy

import torch
from torch import optim
from torch.utils.data import Subset
from torch_geometric.data import DataLoader
import numpy as np
from numpy.random import shuffle

from rna_dataset import RNADataset
from model import DeeperGCN
from model_dummy import MPNNet
from data_creator import RNA_ONEHOT, BOND_ONEHOT, LOOP_ONEHOT

def seed_all(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class RMSELoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss

class MCRMSELoss(torch.nn.Module):
    def __init__(self, num_scored=3):
        super().__init__()
        self.rmse = RMSELoss()
        self.num_scored = num_scored

    def forward(self, yhat, y):
        score = 0
        for i in range(self.num_scored):
            score += self.rmse(yhat[:, i], y[:, i]) / self.num_scored

        return score

def trainer(model, dataloaders,
            n_epochs, lr_init, scheduler_step_size, scheduler_gamma,
            trainer_save):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #optimizer = optim.SGD(model.parameters(), lr=lr_init, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=lr_init)
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


if __name__ == '__main__':

    # Set runtime parameters
    run_n_hidden_channels = 128
    run_n_conv_layers = 12
    run_data_inp = './data/train.json'
    run_batch_size = 16
    run_frac_test = 0.1
    run_n_epochs = 100
    run_lr_init = 0.01
    run_scheduler_step_size = 20
    run_scheduler_gamma = 0.1
    run_trainer_save = './trainer_0_save'

    # PyTorch preliminaries to help debugging and analysis during development
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(1794)

    # Define the dataset and dataloader
    rna_structure_data = RNADataset(run_data_inp, filter_noise=True)
    all_inds = list(range(len(rna_structure_data)))
    shuffle(all_inds)
    test_inds = all_inds[:int(run_frac_test * len(rna_structure_data))]
    train_inds = all_inds[int(run_frac_test * len(rna_structure_data)):]
    rna_structure_loaders = {'train' : DataLoader(Subset(rna_structure_data, train_inds), batch_size=run_batch_size, shuffle=True),
                            'test' : DataLoader(Subset(rna_structure_data, test_inds), batch_size=run_batch_size, shuffle=False)}

    # Define the model
    n_in_edge = len(list(BOND_ONEHOT.values())[0])
    n_in_node = len(list(RNA_ONEHOT.values())[0]) + len(list(LOOP_ONEHOT.values())[0])
    n_out_node = rna_structure_data.n_pred_dim
    model = DeeperGCN(n_in_edge_props=n_in_edge, n_in_node_props=n_in_node,
                      n_out_node_props=n_out_node,
                      n_hidden_channels=run_n_hidden_channels,
                      num_layers=run_n_conv_layers)
#    model = MPNNet(n_in_node,128,n_out_node,10,n_in_edge)

    # Define the trainer
    trainer(model, rna_structure_loaders,
            n_epochs=run_n_epochs, lr_init=run_lr_init,
            scheduler_step_size=run_scheduler_step_size, scheduler_gamma=run_scheduler_gamma,
            trainer_save=run_trainer_save)
