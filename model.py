'''Bla bla for now

'''
import torch
from torch.nn.utils import weight_norm
from torch_geometric.nn import GCNConv, GMMConv, ChebConv, SGConv

from torch_geometric.data import Data

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class LatentGraph(torch.nn.Module):
    '''Bla bla for now

    '''
    def __init__(self, n_in_edge_props, n_in_node_props,
                 n_latent_edges, n_latent_nodes,
                 n_out_node_props,
                 edge_latent_epsilon=1e-5,
                 conv_type='GCNConv', conv_n_iter=1,
                 conv_kwargs={}):

        super(LatentGraph, self).__init__()

        self.n_in_edge_props = n_in_edge_props
        self.n_in_node_props = n_in_node_props
        self.n_latent_edges = n_latent_edges
        self.n_latent_nodes = n_latent_nodes
        self.n_out_node_props = n_out_node_props

        self.edge_latent_epsilon = edge_latent_epsilon
        self.conv_types = {'GCNConv' : GCNConv, 'GMMConv' : GMMConv, 'ChebConv' : ChebConv, 'SGConv' : SGConv}
        self.iter_graph_conv = conv_n_iter
        self.conv_kwargs = conv_kwargs

        try:
            self.graph_conv = self.conv_types[conv_type](in_channels=self.n_in_node_props,
                                                         out_channels=self.n_latent_nodes,
                                                         **self.conv_kwargs)
            #print (count_parameters(self.graph_conv))
            #raise RuntimeError
        except KeyError:
            raise RuntimeError('Unknown graph convolution type: {}'.format(conv_type))

        # MLP at the end of the model
        self.fc_node_prop = torch.nn.Sequential(\
                            torch.nn.Linear(in_features=self.n_latent_edges * self.n_latent_nodes,
                                            out_features=self.n_latent_edges * self.n_latent_nodes, bias=True),
                            torch.nn.ReLU(),
                            torch.nn.Linear(in_features=self.n_latent_edges * self.n_latent_nodes,
                                            out_features=self.n_out_node_props, bias=True))

        # Linear transformation of edge properties to edge weights. Normalize weights so only relative
        # differences in edge weights matters.
        self.fc_edge_latent = weight_norm(torch.nn.Linear(in_features=self.n_in_edge_props,
                                                          out_features=self.n_latent_edges, bias=False))

    def forward(self, data_graph):

        # Transform edge attributes to plurality of edge weights
        edge_props = self.fc_edge_latent(data_graph.edge_attr)
        edge_props += self.edge_latent_epsilon * torch.ones(edge_props.shape)

        # Apply graph convolutions to plurality of differently weighted graphs
        collection_tensors = []
        for e_weight in edge_props.permute(1,0):
            g_conv_out = self.graph_conv(data_graph.x, edge_index=data_graph.edge_index, edge_attr=data_graph.edge_attr)
            collection_tensors.append(g_conv_out)
        graph_step_ = torch.cat(collection_tensors, 1)

        # Transform convoluted node data to plurality of node properties
        graph_step_1 = self.fc_node_prop(graph_step_)

        return Data(y=graph_step_1, edge_index=data_graph.edge_index, edge_attr=data_graph.edge_attr)

    @property
    def available_conv_types(self):
        return list(self.conv_types.keys())
