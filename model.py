'''Bla bla for now

'''
from collections import OrderedDict

import torch
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GMMConv, ChebConv, SGConv, GENConv, DeepGCNLayer
from torch.nn import Conv1d, ReLU, Linear, BatchNorm1d, LayerNorm, Sequential

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

class _DenserConvBlock_bottleneck(torch.nn.Module):
    '''Bla bla

    '''
    def __init__(self, n_inp_features, growth_rate, bottle_factor, drop_rate, kernel_size):
        super(_DenserConvBlock_bottleneck, self).__init__()

        self.drop_rate = drop_rate

        self.norm_a = LayerNorm([n_inp_features, 107])
        self.act_a = ReLU(inplace=True)
        self.conv1d_a = Conv1d(in_channels=n_inp_features,
                               out_channels=growth_rate * bottle_factor,
                               kernel_size=kernel_size, padding_mode='reflect', padding=int((kernel_size-1)/2))
        self.norm_b = LayerNorm([growth_rate * bottle_factor, 107])
        self.act_b = ReLU(inplace=True)
        self.conv1d_b = Conv1d(in_channels=growth_rate * bottle_factor,
                               out_channels=growth_rate,
                               kernel_size=kernel_size, padding_mode='reflect', padding=int((kernel_size-1)/2))

    def forward(self, x):
        out = self.norm_a(x)
        out = self.act_a(out)
        out = self.conv1d_a(out)
        out = self.norm_b(out)
        out = self.act_b(out)
        out = self.conv1d_b(out)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        ret = torch.cat([x, out], dim=1)
        return ret

class _DenseConvBlock(torch.nn.Module):
    '''Bla bla

    '''
    def __init__(self, n_inp_features, growth_rate, drop_rate, kernel_size, glu_act):
        super(_DenseConvBlock, self).__init__()

        self.drop_rate = drop_rate
        self.glu_act = glu_act

        if self.glu_act:
            out_ch = growth_rate * 2
        else:
            out_ch = growth_rate

        self.conv1d = Conv1d(in_channels=n_inp_features,
                             out_channels=out_ch,
                             groups=1, kernel_size=kernel_size, padding_mode='reflect', padding=int((kernel_size-1)/2))
        self.norm = LayerNorm([growth_rate, 107])

    def forward(self, x):
        out = self.conv1d(x)
        if self.glu_act:
            out = F.glu(out, dim=1)
        else:
            out = F.relu(out)
        out = self.norm(out)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        ret = torch.cat([x, out], dim=1)
        return ret

class DenseDeep1D(torch.nn.Module):
    '''Bla bla

    '''
    def __init__(self, n_in_channels, n_out_channels, n_blocks, n_init_features, growth_rate, drop_rate, kernel_sizes, glu_act, bt_f=None):
        super(DenseDeep1D, self).__init__()

        self.n_blocks = n_blocks
        self.features = torch.nn.Sequential(OrderedDict([
            ('conv0', Conv1d(n_in_channels, n_init_features, kernel_size=kernel_sizes['conv0'],
                             padding_mode='reflect', padding=int((kernel_sizes['conv0']-1)/2))),
            ('norm0', LayerNorm([n_init_features, 107])),
            ('relu0', ReLU(inplace=True))
        ]))

        for k_block in range(n_blocks):
            if bt_f is None:
                self.features.add_module('block_{}'.format(k_block),
                                     _DenseConvBlock(n_init_features + k_block * growth_rate,
                                                     growth_rate=growth_rate,
                                                     drop_rate=drop_rate,
                                                     kernel_size=kernel_sizes['blocks'],
                                                     glu_act=glu_act))
            else:
                self.features.add_module('block_{}'.format(k_block),
                                     _DenserConvBlock_bottleneck(n_init_features + k_block * growth_rate,
                                                                 growth_rate=growth_rate,
                                                                 drop_rate=drop_rate,
                                                                 kernel_size=kernel_sizes['blocks'],
                                                                 bottle_factor=bt_f))

        self.act_final = ReLU(inplace=True)
        self.regression = Linear(n_init_features + n_blocks * growth_rate, n_out_channels)

    def forward(self, x):
        x_perm = x.permute(0,2,1)
        x1 = self.features(x_perm)
        x2 = self.act_final(x1)
        out = self.regression(x2.permute(0,2,1))
        return out

class DenseDeep1D_incept(torch.nn.Module):
    '''Bla bla

    '''
    def __init__(self, n_in_channels, n_out_channels, n_blocks, n_init_features, growth_rate, drop_rate, kernel_internal, glu_act):
        super(DenseDeep1D_incept, self).__init__()

        self.n_blocks = n_blocks
        self.init_features = torch.nn.ModuleList()
        for conv_level, vals in enumerate(n_init_features):
            if vals > 0:
                self.init_features.append(Conv1d(n_in_channels, vals, kernel_size=1 + 2 * conv_level,
                                                 padding_mode='reflect', padding=conv_level))
        total_init_features = sum(n_init_features)
        self.norm0 = LayerNorm([total_init_features, 107])
        self.act0 = ReLU(inplace=True)

        self.features = torch.nn.Sequential(OrderedDict())
        for k_block in range(n_blocks):
            self.features.add_module('block_{}'.format(k_block),
                                     _DenseConvBlock(total_init_features + k_block * growth_rate,
                                                     growth_rate=growth_rate,
                                                     drop_rate=drop_rate,
                                                     kernel_size=kernel_internal,
                                                     glu_act=glu_act))

        self.act_final = ReLU(inplace=True)
        self.regression = Linear(total_init_features + n_blocks * growth_rate, n_out_channels)

    def forward(self, x):
        x_perm = x.permute(0,2,1)
        feature_data = []
        for feature_conv in self.init_features:
            feature_data.append(feature_conv(x_perm))
        x_next = torch.cat(feature_data, dim=1)
        #print (x_next.shape)
        #raise RuntimeError
        x_next = self.norm0(x_next)
        x_next = self.act0(x_next)

        x1 = self.features(x_next)
        x2 = self.act_final(x1)
        out = self.regression(x2.permute(0,2,1))
        return out

class _ConvBlock_A(torch.nn.Module):
    '''Bla bla

    '''
    def __init__(self, n_inp_features, out_channels, drop_rate, glu_act):
        super(_ConvBlock_A, self).__init__()

        self.drop_rate = drop_rate
        self.glu_act = glu_act

        if self.glu_act:
            out_ch = out_channels * 2
        else:
            out_ch = out_channels

        self.conv_a = Conv1d(in_channels=n_inp_features,
                             out_channels=out_ch,
                             groups=1, kernel_size=1, padding_mode='reflect', padding=0)
        self.conv_b = Conv1d(in_channels=n_inp_features,
                             out_channels=out_ch,
                             groups=1, kernel_size=3, padding_mode='reflect', padding=1)
        self.conv_c = Conv1d(in_channels=n_inp_features,
                             out_channels=out_ch,
                             groups=1, kernel_size=5, padding_mode='reflect', padding=2)
        self.conv = torch.nn.ModuleList([self.conv_a, self.conv_b, self.conv_c])
        self.norm = LayerNorm(out_channels * 3)

    def forward(self, x):
        out_total = []
        for conv_op in self.conv:
            out = conv_op(x)
            if self.glu_act:
                out = F.glu(out, dim=1)
            else:
                out = F.relu(out)
            out_total.append(out)
        out = torch.cat(out_total, dim=1)
        out = self.norm(out.permute(0,2,1))
        out = out.permute(0,2,1)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)

        return out

class Deep1D_incept(torch.nn.Module):
    '''Bla bla

    '''
    def __init__(self, n_in_channels, n_out_channels, n_hidden_channels, n_blocks, n_init_features, drop_rate, glu_act):
        super(Deep1D_incept, self).__init__()

        self.n_blocks = n_blocks
        self.init_features = torch.nn.ModuleList()
        for conv_level, vals in enumerate(n_init_features):
            if vals > 0:
                self.init_features.append(Conv1d(n_in_channels, vals, kernel_size=1 + 2 * conv_level,
                                                 padding_mode='reflect', padding=conv_level))
        total_init_features = sum(n_init_features)
        self.norm0 = LayerNorm(total_init_features)
        self.act0 = ReLU(inplace=True)

        n_layer_channels = n_hidden_channels * 3
        self.features = torch.nn.Sequential(OrderedDict())
        self.features.add_module('block_0',
                                 _ConvBlock_A(total_init_features,
                                              out_channels=n_hidden_channels,
                                              drop_rate=drop_rate,
                                              glu_act=glu_act))
        for k_block in range(n_blocks - 1):
            self.features.add_module('block_{}'.format(k_block + 1),
                                     _ConvBlock_A(n_layer_channels,
                                                  out_channels=n_hidden_channels,
                                                  drop_rate=drop_rate,
                                                  glu_act=glu_act))

        self.act_final = ReLU(inplace=True)
        self.regression = Linear(n_layer_channels, n_out_channels)

    def forward(self, x):
        x_perm = x.permute(0,2,1)
        feature_data = []
        for feature_conv in self.init_features:
            feature_data.append(feature_conv(x_perm))
        x_next = torch.cat(feature_data, dim=1)
        x_next = self.norm0(x_next.permute(0,2,1))
        x_next = self.act0(x_next.permute(0,2,1))

        x1 = self.features(x_next)
        x2 = self.act_final(x1)
        out = self.regression(x2.permute(0,2,1))
        return out

class _ResBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, hidden_channels, kernel_types, p_dropout=0.0):
        super(_ResBlock, self).__init__()

        self.p_dropout = p_dropout
        k_params = []
        for k_type in kernel_types:
            if k_type == 'basic':
                k_size = 3
                k_stride = 1
                k_dilation = 1
                k_padding = int((k_size - 1) / 2)

            elif k_type == 'dilated':
                k_size = 3
                k_stride = 1
                k_dilation = 2
                k_padding = int((k_size - 1) / 2)

            else:
                raise IndexError('Unknown kernel type provided: {}'.format(k_type))

            k_params.append((k_size, k_stride, k_dilation, k_padding))

        self.comp_units = torch.nn.ModuleDict({
            'conv1d_0' : Conv1d(in_channels=in_channels, out_channels=hidden_channels,
                                kernel_size=k_params[0][0],
                                stride=k_params[0][1],
                                dilation=k_params[0][2],
                                padding=k_params[0][3]),
            'norm_0' : LayerNorm(hidden_channels),
            'act_0' : ReLU(inplace=True),
            'conv1d_1' : Conv1d(in_channels=hidden_channels, out_channels=out_channels,
                                kernel_size=k_params[1][0],
                                stride=k_params[1][1],
                                dilation=k_params[1][2],
                                padding=k_params[1][3]),
            'norm_1' : LayerNorm(out_channels),
            'act_final' : ReLU(inplace=True)}
        )

        if in_channels == out_channels:
            self.dim_mapper = torch.nn.Identity()
        else:
            self.dim_mapper = torch.nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        h0 = self.comp_units['conv1d_0'](x)
        h1 = self.comp_units['norm_0'](h0.permute(0, 2, 1)).permute(0, 2, 1)
        h2 = self.comp_units['act_0'](h1)
        h3 = self.comp_units['conv1d_1'](h2)
        h = self.comp_units['norm_1'](h3.permute(0, 2, 1)).permute(0, 2, 1)
        x_reshape = self.dim_mapper(x.permute(0, 2, 1)).permute(0, 2, 1)
        x_reshape = F.dropout(x_reshape, self.p_dropout)
        x_new = self.comp_units['act_final'](x_reshape + h)
        return x_new.permute(0, 2, 1)

class Res1D(torch.nn.Module):

    def __init__(self, in_channels, out_channels, nblocks, hidden_progression):
        super(Res1D, self).__init__()

        self.nblocks = nblocks

        self.blocks = Sequential(OrderedDict())
        self.blocks.add_module('block_0',
                               _ResBlock(in_channels=in_channels, out_channels=hidden_progression[0],
                                         hidden_channels=hidden_progression[0],
                                         kernel_types=('basic', 'basic')))

        for kblock in range(self.nblocks - 1):
            self.blocks.add_module('block_{}'.format(kblock + 1),
                                   _ResBlock(in_channels=hidden_progression[kblock],
                                             out_channels=hidden_progression[kblock + 1],
                                             hidden_channels=hidden_progression[kblock + 1],
                                             kernel_types=('basic', 'basic')))

        self.mlp = Sequential(Linear(in_features=hidden_progression[self.nblocks - 1],
                                     out_features=hidden_progression[self.nblocks - 1]),
                              ReLU(inplace=True),
                              Linear(in_features=hidden_progression[self.nblocks - 1],
                                     out_features=out_channels))

    def forward(self, x):

        x_feats = self.blocks(x)
        y_pred = self.mlp(x_feats)

        return y_pred