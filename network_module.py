import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
from utility import get_input_white_noise
import math
import torch.nn.init as init


class SampleNet(nn.Module):

    def __init__(self, input_dim, target_dim, n_nodes, net_structure):
        super(SampleNet, self).__init__()
        fc_dim = [input_dim] + net_structure['net_dim'] + [target_dim]
        self.layers_list = nn.ModuleList()
        # self.adj_layers_list = nn.ModuleList()
        activations = activations_to_torch(net_structure['activations'])
        self.net_type = net_structure['net_type']
        self.inner_prod = InnerProductDecoder(0.)

        # Simple fully connected layers (decoder)
        if net_structure['net_type'] == 'gnn':
            for i in range(1, len(fc_dim) - 1):
                self.layers_list.append(nn.Linear(fc_dim[i - 1], fc_dim[i]))
                self.layers_list.append(nn.BatchNorm1d(fc_dim[i]))
                self.layers_list.append(activations[0])
            self.layers_list.append(nn.Linear(fc_dim[-2], fc_dim[-1]))
            self.layers_list.append(nn.BatchNorm1d(fc_dim[-1]))
            self.layers_list.append(activations[1])


    def forward(self, noise):
        a = noise.clone()
        for layer in self.layers_list:
            a = layer(a)

        return a


class AdvNet(nn.Module):

    def __init__(self, target_dim, feature_dim, net_structure):
        super(AdvNet, self).__init__()
        fc_dim = [target_dim] + net_structure['net_dim'] + [feature_dim]
        dropout = 0.
        self._input_adv_t_net_dim = feature_dim
        self.t_sigma_num = net_structure['adv_t_sigma_num']
        self.net_type = net_structure['net_type']
        activations = activations_to_torch(net_structure['activations_a'])

        # Simple GNN models
        if net_structure['net_type'] == 'gnn':
            self.layers_list = nn.ModuleList()
            for i in range(1, len(fc_dim) - 1):
                self.layers_list.append(GraphConvolution(fc_dim[i - 1], fc_dim[i], dropout, act=activations[0]))
            self.layers_list.append(GraphConvolution(fc_dim[-2], fc_dim[-1], dropout, act=activations[1]))

        if self.t_sigma_num > 0:
            # adversarial nets for t scales
            self.t_layers_list = nn.ModuleList()
            ch_in = feature_dim
            activations = activations_to_torch(net_structure['activations_t'])
            for i in range(3):
                self.t_layers_list.append(nn.Linear(ch_in, ch_in))
                self.t_layers_list.append(nn.BatchNorm1d(ch_in))
                activation = activations[0] if i < 2 else activations[1]
                self.t_layers_list.append(activation)

        # base mean and covariance
        self._input_t_dim = net_structure['input_t_dim']
        self._input_t_batchsize = net_structure['input_t_batchsize']
        self._input_t_var = net_structure['input_t_var']

    def forward(self, noise, adj):
        a = noise
        for layer in self.layers_list:
            a = layer(a, adj)
        a = a.view(a.shape[0], -1)
        return nn.functional.normalize(a, p=2, dim=1)

    def net_t(self):
        if self.t_sigma_num > 0:
            device = 'cpu'
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
            self._t_net_input = get_input_white_noise(self._input_adv_t_net_dim, self._input_t_var,
                                                      self.t_sigma_num).detach().to(device)
            a = self._t_net_input
            for layer in self.t_layers_list:
                a = layer(a)
            a = a.repeat(int(self._input_t_batchsize / self.t_sigma_num), 1)
            self._t = get_input_white_noise(self._input_t_dim, self._input_t_var / self._input_t_dim,
                                            # dimension normalisation
                                            self._input_t_batchsize).detach().to(device)
            self._t = self._t * a
        else:
            self._t = get_input_white_noise(self._input_t_dim, self._input_t_var / self._input_t_dim,
                                            self._input_t_batchsize).detach()
            device = 'cpu'
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
            self._t = self._t.to(device)
            # add incremental functions here
        return self._t


def activations_to_torch(activations):
    for i, i_act in enumerate(activations):
        if i_act[0] == 'tanh':
            activations[i] = nn.Tanh()
        elif i_act[0] == 'sigmoid':
            activations[i] = nn.Sigmoid()
        elif i_act[0] == 'relu':
            activations[i] = nn.ReLU()
        elif i_act[0] == 'lrelu':
            activations[i] = nn.LeakyReLU(negative_slope=i_act[1])
        elif i_act[0] is None:
            pass
        else:
            raise SystemExit('Error: Unknown activation function \'{0}\''.format(i_act[0]))
    return activations

# The same version with
# https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=nn.ReLU()):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = nn.functional.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = nn.functional.normalize(z, p=2, dim=1)
        z = nn.functional.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj
