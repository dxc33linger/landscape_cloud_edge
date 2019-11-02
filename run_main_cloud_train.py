import os
import shutil
import scipy.io
from args import *
args = parser.parse_args()

# # #
# if os.path.exists('../baseline_library'):
# 	shutil.rmtree('../baseline_library')
# os.mkdir('../baseline_library')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#'resnet20', 'resnet20_noshort',

i = 0
for model in ['resnet20']:#, 'resnet20_noshort', 'resnet56_noshort','resnet56', 'densenet121']:

	NA_C0 = 16

	epoch = {'resnet20':60, 'resnet20_noshort':120,  'resnet110':80, 'resnet110_noshort':150, 'densenet121': 125, 'vgg11': 200,
	         'resnet56': 80, 'resnet56_noshort': 150}

	command_tmp = 'python main_cloud_train.py --batch_size 64 --epoch ' + str(epoch[model]) +' --dataset cifar10 --NA_C0 '+ str(NA_C0)+ ' --model ' + model

	print('command:\n', command_tmp)
	os.system(command_tmp)


	command_tmp = 'python main_edge_train.py --gpu 1 --epoch_edge ' + str(epoch_edge[model]) +' --dataset cifar10  --model ' + model +' --epoch ' +  str(epoch[model])+' --batch_size ' + str(batch[model]) + ' --NA_C0 ' +str(NA_C0)
	# command_tmp = 'python main_edge_train.py --epoch_edge 10  --dataset cifar10  --model ' + model +' --epoch ' + str(epoch[model])+' --batch_size ' + str(batch[model])

	print('command:\n', command_tmp)

	os.system(command_tmp)

