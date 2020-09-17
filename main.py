'''Main

'''
import torch
from torch import optim
import torch.nn.functional as F
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
    print (dir(dd))
    print (dd.n_seq_score)
    lgcn.forward(dd)

def trainer(n_epochs, data_root, batch_size,
            n_weight_versions, n_node_versions,
            conv_type, conv_n_iter, conv_kwargs,
            lr_init, scheduler_step_size, scheduler_gamma):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rna_structure_data = RNADataset(data_root)
    n_in_edge = len(list(BOND_ONEHOT.values())[0])
    n_in_node = len(list(RNA_ONEHOT.values())[0])
    rna_structure_loader = DataLoader(rna_structure_data, batch_size=batch_size, shuffle=False)

    lgcn = LatentGraph(n_in_edge_props=n_in_edge, n_in_node_props=n_in_node,
                       n_latent_edges=n_weight_versions,
                       n_latent_nodes=n_node_versions,
                       n_out_node_props=5,
                       conv_type=conv_type, conv_n_iter=conv_n_iter, conv_kwargs=conv_kwargs)

    optimizer = optim.SGD(lgcn.parameters(), lr=lr_init, momentum=0.9)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                 step_size=scheduler_step_size,
                                                 gamma=scheduler_gamma)

    lgcn.train()
    for k_epoch in range(n_epochs):

        running_loss = 0.0
        for k_batch, data_batch in enumerate(rna_structure_loader):
            #print (k_batch, data_batch.x.shape)
            data_batch = data_batch.to(device)

            ind_select = []
            low_ind = 0
            for low, high in zip(data_batch.n_seq_score, data_batch.n_seq_total):
                ind_select += list(range(low_ind, low_ind + low))
                low_ind += high
            ind_select = torch.tensor(ind_select, dtype=torch.int64)

            optimizer.zero_grad()
            out = lgcn(data_batch)

            y_to_score = out.y.index_select(0, ind_select)
            loss = 0.0
            for pred_type, ref_type in zip(y_to_score.permute(1, 0), data_batch.y.permute(1, 0)):
                loss += F.mse_loss(pred_type, ref_type)
            loss = loss / float(y_to_score.shape[1])

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        exp_lr_scheduler.step()

        running_loss = running_loss / (k_batch + 1)
        print ('Epoch {} with running loss (training) {}'.format(k_epoch, running_loss))


if __name__ == '__main__':
    #rudimentary_test()
    trainer(n_epochs=20, data_root='./data_processed_train', batch_size=1,
            n_weight_versions=1, n_node_versions=16,
            conv_type='GMMConv', conv_n_iter=2, conv_kwargs={'dim':4, 'kernel_size':32},
            lr_init=0.1, scheduler_step_size=10, scheduler_gamma=0.1)