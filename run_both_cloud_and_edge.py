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
matplotlib.use('Agg')
import matplotlib.pyplot as plt


i = 0

# for task_division in ['1,1,1,1,1,1,1,1,1,1','5, 1, 1, 1, 1, 1', '6,1,1,1,1', '7,1,1,1', '8,1,1', '9,1']:
for task_division in ['90,10', '80,10,10', '70,10,10,10', '60,10,10,10,10', '50,10,10,10,10,10', '10,10,10,10,10,10,10,10,10,10']:
# for task_division in ['9,1', '8,1,1', '7,1,1,1', '6,1,1,1,1', '5,1,1,1,1,1', '1,1,1,1,1,1,1,1,1,1']:

	dataset = 'cifar100'
	model = 'resnet32_noshort' #, 'resnet20_noshort', 'resnet56_noshort','resnet56', 'densenet121'

	NA_C0 = 32
	batch = 64
	# epoch = int(task_division.split(',')[0])
	# epoch_edge = int(int(task_division.split(',')[1]) * 2.5)

	epoch = {'resnet32':60, 'resnet32_noshort':60, 'resnet20_noshort':15, 'resnet56':15, 'resnet56_noshort':20,
	              'resnet110':20, 'resnet110_noshort':25, 'densenet121': 25}

	epoch_edge = {'resnet32':20, 'resnet32_noshort':20, 'resnet20_noshort':15, 'resnet56':15, 'resnet56_noshort':20,
	              'resnet110':20, 'resnet110_noshort':25, 'densenet121': 25}

	#
	# command_tmp = 'python main_cloud_train.py --gpu 1 --batch_size 64 --epoch ' + str(epoch[model]) +' --dataset '+ dataset +' --NA_C0 '+ str(NA_C0)+ ' --model ' + model + ' --task_division ' + task_division
	# print('command:\n', command_tmp)
	# os.system(command_tmp)
	#
	# command_tmp = 'python main_edge_train.py --gpu 1 --epoch_edge ' + str(epoch_edge[model]) +' --dataset '+ dataset +' --model ' + model +' --epoch ' +  str(epoch[model])+' --batch_size ' + str(batch) + ' --NA_C0 ' +str(NA_C0) \
	#               +' --task_division ' + task_division
	# print('command:\n', command_tmp)
	# os.system(command_tmp)
	# i = i + 1

	command_tmp = 'python PST_main.py --gpu 1 --epoch ' + str(epoch[model]) +' --epoch_edge '+str(epoch_edge[model])+' --dataset '+ dataset +' --model ' + model +' --batch_size ' + str(batch) + ' --NA_C0 ' +str(NA_C0) +' --task_division ' + task_division
	print('command:\n', command_tmp)
	os.system(command_tmp)
	i = i + 1

	scipy.io.savemat('../../results/tuning_{}.mat'.format(i), {'i':i, 'model': model, 'epoch':epoch, 'task':args.task_division, 'dataset':args.dataset, 'NA_C0': args.NA_C0})


