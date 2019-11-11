import os
import shutil
import scipy.io
from args import *
args = parser.parse_args()


if os.path.exists('../../results'):
	shutil.rmtree('../../results')
os.mkdir('../../results')
if os.path.exists('../../model_library'):
	shutil.rmtree('../../model_library')
os.mkdir('../../model_library')
if os.path.exists('../../mask_library'):
	shutil.rmtree('../../mask_library')
os.mkdir('../../mask_library')

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


i = 0
for model in ['resnet20', 'resnet20_noshort', 'resnet56', 'resnet56_noshort', 'densenet121']:

	NA_C0 = 16
	task_division = '6,1'
	batch = 64

	epoch = {'resnet20':50, 'resnet20_noshort':60,  'resnet110':80, 'resnet110_noshort':150, 'densenet121': 80, 'vgg11': 200,
	         'resnet56': 60, 'resnet56_noshort': 70}


	command_tmp = 'python main_cloud_train.py --gpu 1 --epoch ' + str(epoch[model]) +' --dataset cifar10  --model ' + model + ' --batch_size ' + str(batch) + ' --NA_C0 ' +str(NA_C0) + ' --task_division ' + task_division


	print('command:\n', command_tmp)

	os.system(command_tmp)

