'''Bla bla for now

'''
import torch
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GMMConv, ChebConv, SGConv, GENConv, DeepGCNLayer
from torch.nn import Conv1d, ReLU, Linear

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

        raise NotImplementedError('This graph convolutional network not properly implemented')

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
        self.fc_edge_latent = torch.nn.Sequential(\
                              torch.nn.Linear(in_features=self.n_in_edge_props,
                                              out_features=self.n_latent_edges, bias=False),
                              torch.nn.Sigmoid())

    def forward(self, data_graph):

        # Transform edge attributes to plurality of edge weights. Ensure no value greater than a threshold
        edge_props = self.fc_edge_latent(data_graph.edge_attr)

        # Apply graph convolutions to plurality of differently weighted graphs
        collection_tensors = []
        for e_weight in edge_props.permute(1,0):
            g_conv_out = self.graph_conv(data_graph.x, edge_index=data_graph.edge_index, edge_weight=e_weight)
            collection_tensors.append(g_conv_out)
        graph_step_ = torch.cat(collection_tensors, 1)

        # Transform convoluted node data to plurality of node properties
        graph_step_1 = self.fc_node_prop(graph_step_)

        return Data(y=graph_step_1, edge_index=data_graph.edge_index, edge_attr=data_graph.edge_attr)

    @property
    def available_conv_types(self):
        return list(self.conv_types.keys())

class DeeperGCN(torch.nn.Module):
    '''Bla bla for now

    '''
    def __init__(self, n_in_edge_props, n_in_node_props,
                 n_hidden_channels,
                 n_out_node_props,
                 num_layers):
        super(DeeperGCN, self).__init__()

        self.n_in_edge_props = n_in_edge_props
        self.n_in_node_props = n_in_node_props
        self.n_hidden_channels = n_hidden_channels
        self.n_out_node_props = n_out_node_props
        self.num_layers = num_layers

        # Transform the node and edge feature vectors into scalars in hidden channels
        self.node_encoder = torch.nn.Sequential(torch.nn.Linear(in_features=self.n_in_node_props,
                                                                out_features=self.n_hidden_channels))
        self.edge_encoder = torch.nn.Sequential(torch.nn.Linear(in_features=self.n_in_edge_props,
                                                                out_features=self.n_hidden_channels),
                                                torch.nn.Sigmoid())

        # Construct layers of graph convolutions. Use the DeepGCN architecture, which is comprised of
        # MLPs for each node, input being neighbour nodes and edge values, which is aggregated and added
        # as a residual to the previous node scalars
        self.layers = torch.nn.ModuleList()
        for i in range(self.num_layers):
            conv = GENConv(in_channels=self.n_hidden_channels, out_channels=self.n_hidden_channels,
                           num_layers=2, norm='layer', t=1.0, learn_t=True, aggr='softmax')
            norm = torch.nn.LayerNorm(self.n_hidden_channels, elementwise_affine=True)
            activation = torch.nn.ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, activation,
                                 block='res+', dropout=0.1)

            self.layers.append(layer)

        # Transform hidden channels to output properties
        self.lin_out = torch.nn.Linear(self.n_hidden_channels, self.n_out_node_props)

    def forward(self, data_graph):
        x = self.node_encoder(data_graph.x)
        edge = self.edge_encoder(data_graph.edge_attr)

        # Run first convolution layer to create x0
        x = self.layers[0].conv(x, data_graph.edge_index, edge)

        # Run remaining layers and add output as residuals to x0
        for layer in self.layers[1:]:
            x = layer(x, data_graph.edge_index, edge)

        # Run a final non-linear activation and dropout (if training)
        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)

        # Transform output
        x_pred = self.lin_out(x)

        return x_pred

class _DenseConvBlock(torch.nn.Module):
    '''Bla bla

    '''
    def __init__(self, in_channels, out_channels, kernel_size):
        super(_DenseConvBlock, self).__init__()
        self.conv1d = Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)

    def forward(self, x):
        print (x)
        out = self.conv1d(x)
        print (out.shape)
        raise RuntimeError
        return out

class DenseDeep1D(torch.nn.Module):
    '''Bla bla

    '''
    def __init__(self, in_channels, out_channels, n_blocks):
        super(DenseDeep1D, self).__init__()

        self.n_blocks = n_blocks
        self.features = torch.nn.Sequential()
        for k_block in range(n_blocks):
            self.features.add_module('block_{}'.format(k_block), _DenseConvBlock(in_channels, out_channels, 3))

    def forward(self, x):
        xx = self.features(x)


def test1():

    mm = DenseDeep1D(6, 2, 1)
    dd = torch.load('graphs_pytorch_0.pt')
    print (dd.x)
    mm(dd.x)

#test1()