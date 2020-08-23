#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from cfgan_model import Model
# state for using GPU-1
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np

# training epoch
epoch = 300

load = False

# hyperparameters for target image data
target_source = 'citeseer'  # cora citeseer
target_batch_size = 64
feature_dim = 128
target_size = 0  # 0 means using all data samples

ae_loss_reg = 5

# hyperparameters for generating nets (Note: activations are only valid for dcgan)
generator_training_param = {'input_noise_dim': feature_dim,
                            'input_noise_var': 0.3, 'net_type': 'gnn', 'lr': 2e-4, 'inner_ite_per_batch': 1,
                            'activations': [('relu', None), ('sigmoid', None)], 'net_dim': [128, 512],
                            'weight_decay': 0, 'lr_step_size_decay': 0, 'lr_decay_gamma': 0.5}

# Hyperparameters for adversarial nets (Note: activations_a are only valid for dcgan)
adversarial_training_param = {'input_t_dim': feature_dim, 'input_t_batchsize': target_batch_size,
                              'input_t_var': 1, 'net_type': 'gnn', 'lr': 2e-4, 'inner_ite_per_batch': 1,
                              'activations_a': [('lrelu', 0.2), ('tanh', None)], 'net_dim': [512, 128],
                              'weight_decay': 0, 'lr_step_size_decay': 0, 'lr_decay_gamma': 0.5,
                              'adv_t_sigma_num': target_batch_size,
                              'activations_t': [('lrelu', 0.2), ('tanh', None)]}

# hyperparameters for CFLossFunc
loss_alpha = 1  # amplitude
loss_beta = 1  # phase
# threshold is useless when normalization is None
loss_type = {'threshold': 'origin', 'normalization': None}
loss_threshold = 1

# training
mark = '3'
if adversarial_training_param['adv_t_sigma_num'] > 0:
    model_label = 'gl_' + target_source + '_t_net_' + generator_training_param['net_type'] + '_' + mark
else:
    model_label = 'gl_' + target_source + '_t_normal_' + generator_training_param['net_type'] + '_' + mark

model = Model(model_label,
              target_source, target_size, target_batch_size,
              adversarial_training_param, generator_training_param,
              loss_type, loss_alpha, loss_beta, loss_threshold, ae_loss_reg,
              epoch)
# Comment this line for evaluation only given pre-trained models
model.train(load)

# parameters for testing
num_white_noise_test = 100

# testing
# model.test()
