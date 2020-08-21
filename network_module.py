import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
from utility import get_input_white_noise
import math
from ResNet import *
import torch.nn.init as init


class SampleNet(nn.Module):

    def __init__(self, input_dim, target_dim, img_channel, net_structure):
        super(SampleNet, self).__init__()
        self.layers_list = nn.ModuleList()
        self.cnn_flag = False
        activations = activations_to_torch(net_structure['activations'])
        self.net_type = net_structure['net_type']

        # DCGAN
        if net_structure['net_type'] == 'dcgan':
            self.cnn_flag = True
            ch_in = input_dim
            tt_layer_number = int(math.log2(target_dim) - 1)
            top_channel_number = 64 * 2 ** (tt_layer_number - 2)

            conv_layer = nn.ConvTranspose2d(ch_in, top_channel_number, 4, 1, 0, bias=False)
            self.layers_list.append(conv_layer)
            bn_layer = nn.BatchNorm2d(top_channel_number)
            self.layers_list.append(bn_layer)
            self.layers_list.append(activations[0])

            ch_in = top_channel_number
            ch_out = top_channel_number
            for i in range(1, tt_layer_number - 1):
                ch_out = ch_out // 2
                conv_layer = nn.ConvTranspose2d(ch_in, ch_out, 4, 2, 1, bias=False)
                self.layers_list.append(conv_layer)
                bn_layer = nn.BatchNorm2d(ch_out)
                self.layers_list.append(bn_layer)
                self.layers_list.append(activations[0])
                ch_in = ch_out

            conv_layer = nn.ConvTranspose2d(ch_in, img_channel, 4, 2, 1, bias=False)
            self.layers_list.append(conv_layer)
            self.layers_list.append(activations[1])
        elif net_structure['net_type'] == 'adv-dcgan':
            self.cnn_flag = True
            ch_in = input_dim
            tt_layer_number = int(math.log2(target_dim) - 1)
            top_channel_number = 64 * 2 ** (tt_layer_number - 1)  # The only difference with DCGAN

            conv_layer = nn.ConvTranspose2d(ch_in, top_channel_number, 4, 1, 0, bias=False)
            self.layers_list.append(conv_layer)
            bn_layer = nn.BatchNorm2d(top_channel_number)
            self.layers_list.append(bn_layer)
            self.layers_list.append(activations[0])

            ch_in = top_channel_number
            ch_out = top_channel_number
            for i in range(1, tt_layer_number - 1):
                ch_out = ch_out // 2
                conv_layer = nn.ConvTranspose2d(ch_in, ch_out, 4, 2, 1, bias=False)
                self.layers_list.append(conv_layer)
                bn_layer = nn.BatchNorm2d(ch_out)
                self.layers_list.append(bn_layer)
                self.layers_list.append(activations[0])
                ch_in = ch_out

            conv_layer = nn.ConvTranspose2d(ch_in, img_channel, 4, 2, 1, bias=False)
            self.layers_list.append(conv_layer)
            self.layers_list.append(activations[1])
        elif self.net_type == 'resnet':
            self.cnn_flag = True
            self.model_gen = GoodGenerator(input_dim, target_dim)

    def forward(self, noise):
        a = noise
        if self.net_type == 'resnet':
            a = self.model_gen(a)
        else:
            if self.cnn_flag:
                a = a.unsqueeze(-1).unsqueeze(-1)
            for layer in self.layers_list:
                a = layer(a)
        return a


class AdvNet(nn.Module):

    def __init__(self, channel_in, target_dim, feature_dim, net_structure):
        super(AdvNet, self).__init__()
        self.cnn_flag = False
        self._input_adv_t_net_dim = feature_dim
        self.channel_in = channel_in
        self.t_sigma_num = net_structure['adv_t_sigma_num']
        self.net_type = net_structure['net_type']
        activations = activations_to_torch(net_structure['activations_a'])

        # DCGAN
        if net_structure['net_type'] == 'dcgan':
            self.cnn_flag = True
            self.convlayers_list = nn.ModuleList()
            tt_layer_number = int(math.log2(target_dim) - 1)
            ch_in = self.channel_in

            out_channel_number = 64
            conv_layer = nn.Conv2d(ch_in, out_channel_number, 4, 2, 1, bias=False)
            self.convlayers_list.append(conv_layer)
            self.convlayers_list.append(activations[0])
            ch_in = out_channel_number
            ch_out = out_channel_number
            for i in range(1, tt_layer_number - 1):
                ch_out = ch_out * 2
                # conv_layer = nn.Conv2d(ch_in, ch_out, 4, 2, 1, bias=False)
                # self.convlayers_list.append(conv_layer)
                # # bn_layer = nn.LayerNorm(
                # #     [ch_out, target_dim // 2 ** (i + 1), target_dim // 2 ** (i + 1)])
                # bn_layer = nn.BatchNorm2d(ch_out)
                # self.convlayers_list.append(bn_layer)
                conv_sn_layer = nn.utils.spectral_norm(nn.Conv2d(ch_in, ch_out, 4, 2, 1, bias=False))
                self.convlayers_list.append(conv_sn_layer)
                self.convlayers_list.append(activations[0])
                ch_in = ch_out
            conv_layer = nn.Conv2d(ch_in, feature_dim, 4, 1, 0, bias=False)
            self.convlayers_list.append(conv_layer)
            bn_layer = nn.LayerNorm([feature_dim, 1, 1])
            self.convlayers_list.append(bn_layer)
            self.convlayers_list.append(activations[1])
        elif self.net_type == 'resnet':
            self.model_adv = GoodDiscriminator(target_dim, feature_dim)

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

    def forward(self, noise):
        a = noise
        if self.net_type == 'resnet':
            a_compare, a = self.model_adv(a)
            return a_compare, a
        else:
            for layer in self.convlayers_list:
                a = layer(a)
            a_compare = a
            a = a.view(a.shape[0], -1)
            a_compare = a_compare.view(a_compare.shape[0], -1)
            # a_compare = self.final_norm(a_compare)
            return a_compare, a

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
