'''Bla bla for now

'''
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TAGConv

from torch_geometric.data import Data

class LatentGraph(torch.nn.Module):
    '''Bla bla for now

    '''
    def __init__(self, n_in_edge_props, n_in_node_props,
                 n_latent_edges, n_latent_nodes,
                 n_out_node_props):
        super(LatentGraph, self).__init__()

        self.n_in_edge_props = n_in_edge_props
        self.n_in_node_props = n_in_node_props
        self.n_latent_edges = n_latent_edges
        self.n_latent_nodes = n_latent_nodes
        self.n_out_node_props = n_out_node_props

        self.gcn = GCNConv(in_channels=self.n_in_node_props, out_channels=self.n_latent_nodes)

        self.fc_node_prop = torch.nn.Sequential(\
                                torch.nn.Linear(in_features=self.n_latent_edges * self.n_latent_nodes,
                                                out_features=self.n_out_node_props, bias=True),
                                torch.nn.ReLU())
        self.fc_edge_latent = torch.nn.Sequential(\
                                  torch.nn.Linear(in_features=self.n_in_edge_props,
                                                  out_features=self.n_latent_edges, bias=True),
                                  torch.nn.ReLU())

    def forward(self, data_graph):

        # Transform edge attributes to plurality of edge weights
        edge_props = self.fc_edge_latent(data_graph.edge_attr)

        # Apply graph convolutions to plurality of differently weighted graphs
        collection_tensors = []
        for e_weight in edge_props.permute(1,0):
            g_conv_out = self.gcn(data_graph.x, edge_index=data_graph.edge_index, edge_weight=e_weight)
            collection_tensors.append(g_conv_out)
        graph_step_0 = torch.cat(collection_tensors, 1)

        # Transform convoluted node data to plurality of node properties
        graph_step_1 = self.fc_node_prop(graph_step_0)

        return Data(y=graph_step_1, edge_index=data_graph.edge_index, edge_attr=data_graph.edge_attr)

