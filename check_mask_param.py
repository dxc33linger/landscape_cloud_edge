"""
Author: Xiaocong Du
Date: April 2019 - May 2019
Project: Single-Net Continual Learning withProgressive Segmented Training (PST)
Description: check if the mask is correct
"""

import torch
import os
import pickle
import numpy as np
from cifar10 import model_loader


mask_file ='../../mask_library/mask_model_resnet20_lr=0.1_bs=64_task_0_top_0.5.pickle'
model_file_0 = '../../results/model_afterT0_Accu0.9702.pt'
model_file_1 = '../../results/model_afterT1_Accu0.0000.pt'
# model_file_1 = '../../results/model_mid.pt'

content = pickle.load(open(mask_file, "rb"))
print(len(content))
param_dict = content[2]

# """Check conv layers """
"""Where mask = 1, model_T0 and model_T1 should be same  """
"""Where mask = 0, model_T0 and model_T1 should be different."""
# """model_T1 and model_T1_afterFC should be same all the time since only FC retrained"""
print('-------------------------mask--------------------------------')
print(param_dict.keys())
print(param_dict['conv1.weight'][0:3, 0, :, :]) # selected kernel to print, 1 means important filters
print(param_dict['conv1.weight'].shape)
#
print('------------------------- after t0  -------------------------------- ')
model_T0 =  model_loader.load('resnet20')
model_T0.load_state_dict(torch.load(model_file_0))
print(model_T0.state_dict()['conv1.weight'][0:3, 0, :, :])

print('------------------------- after t1 -------------------------------- ')
model_T1 = model_loader.load('resnet20')
model_T1.load_state_dict(torch.load(model_file_1))
print(model_T1.state_dict()['conv1.weight'][0:3, 0, :, :])

# # print('------------------------- after FC retraining -------------------------------- ')
# model_T1_afterFC = VGG()
# model_T1_afterFC.load_state_dict(torch.load('../results/model_INITIAL_Accu0.2340.pt'))
# print(model_T1_afterFC.state_dict()['conv1_1.weight'][22:25, 0, :, :])




#
# # """Check BN layers """
# """model_T0 should be all different """
# " model_T1, model_T1_afterFC should be same"
# print('-------------------------mask--------------------------------')
# print(param_dict['layer1.1.bn1.weight'])
# print(param_dict['layer1.1.bn1.bias'])
# print(param_dict['layer1.1.bn1.running_mean'])
# print(param_dict['layer1.1.bn1.running_var'])
#
# print('------------------------- after t0  -------------------------------- ')
# model_T0 = VGG()
# model_T0.load_state_dict(torch.load(model_file_0))
# print(model_T0.state_dict()['layer1.1.bn1.weight'])
# print(model_T0.state_dict()['layer1.1.bn1.bias'])
# print(model_T0.state_dict()['layer1.1.bn1.running_mean'])
# print(model_T0.state_dict()['layer1.1.bn1.running_var'])
#
# print('------------------------- after t1 -------------------------------- ')
# model_T1 = VGG()
# model_T1.load_state_dict(torch.load(model_file_1))
# print(model_T1.state_dict()['layer1.1.bn1.weight'])
# print(model_T1.state_dict()['layer1.1.bn1.bias'])
# print(model_T1.state_dict()['layer1.1.bn1.running_mean'])
# print(model_T1.state_dict()['layer1.1.bn1.running_var'])
#
# print('------------------------- after FC retraining -------------------------------- ')
# model_T1_afterFC = VGG()
# model_T1_afterFC.load_state_dict(torch.load('../results/model_INITIAL_Accu0.2340.pt'))
# print(model_T1_afterFC.state_dict()['layer1.1.bn1.weight'])
# print(model_T1_afterFC.state_dict()['layer1.1.bn1.bias'])
# print(model_T1_afterFC.state_dict()['layer1.1.bn1.running_mean'])
# print(model_T1_afterFC.state_dict()['layer1.1.bn1.running_var'])







"""Check LINEAR layers """
"""model_T0, model_T1, model_T1_afterFC should be all different """

print('-------------------------mask--------------------------------')

print(param_dict['linear_1.weight'][:, 10:15])
# print(param_dict['linear1.bias'])

print('------------------------- after t0  -------------------------------- ')
model_T0 = VGG()
model_T0.load_state_dict(torch.load(model_file_0))
print(model_T0.state_dict()['linear_1.weight'][:, 10:15])
# print(model_T0.state_dict()['linear1.bias'])

print('------------------------- after t1 -------------------------------- ')
model_T1 = VGG()
model_T1.load_state_dict(torch.load(model_file_1))
print(model_T1.state_dict()['linear_1.weight'][:, 10:15])
# print(model_T1.state_dict()['linear1.bias'])

# print('------------------------- after FC retraining -------------------------------- ')
# model_T1_afterFC = VGG()
# model_T1_afterFC.load_state_dict(torch.load('../results/model_INITIAL_Accu0.2340.pt'))
# print(model_T1_afterFC.state_dict()['linear1.weight'][:, 10:15])
# print(model_T1_afterFC.state_dict()['linear1.bias'])
