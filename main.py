'''Main

'''
import torch
from torch import optim
from torch_geometric.data import DataLoader

from rna_dataset import RNADataset
from model import LatentGraph
from data_creator import RNA_ONEHOT, BOND_ONEHOT

def rudimentary_test():

    data_test = RNADataset('./data_processed_train')

    n_in_edge = len(list(BOND_ONEHOT.values())[0])
    n_in_node = len(list(RNA_ONEHOT.values())[0])
    lgcn = LatentGraph(n_in_edge_props=n_in_edge, n_in_node_props=n_in_node,
                       n_latent_edges=16,
                       n_latent_nodes=16,
                       n_out_node_props=5)

    dd = data_test.get(0)
    lgcn.forward(dd)

def trainer(n_epochs, data_root, batch_size, n_weight_versions, n_node_versions, lr_init, scheduler_step_size, scheduler_gamma):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rna_structure_data = RNADataset(data_root)
    n_in_edge = len(list(BOND_ONEHOT.values())[0])
    n_in_node = len(list(RNA_ONEHOT.values())[0])
    rna_structure_loader = DataLoader(rna_structure_data, batch_size=batch_size, shuffle=False)

    lgcn = LatentGraph(n_in_edge_props=n_in_edge, n_in_node_props=n_in_node,
                       n_latent_edges=n_weight_versions,
                       n_latent_nodes=n_node_versions,
                       n_out_node_props=5)

    optimizer = optim.SGD(lgcn.parameters(), lr=lr_init, momentum=0.9)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                 step_size=scheduler_step_size,
                                                 gamma=scheduler_gamma)

    lgcn.train()
    for k_epoch in range(n_epochs):

        for k_batch, data_batch in enumerate(rna_structure_loader):
            data_batch = data_batch.to(device)
            optimizer.zero_grad()

            out = lgcn(data_batch)
            raise RuntimeError


if __name__ == '__main__':
    rudimentary_test()