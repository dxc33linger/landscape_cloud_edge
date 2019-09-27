import os
import shutil
import scipy.io
from args import *
args = parser.parse_args()

# #
if os.path.exists('../../baseline_library'):
	shutil.rmtree('../../baseline_library')
os.mkdir('../../baseline_library')

if os.path.exists('../../model_library'):
	shutil.rmtree('../../model_library')
os.mkdir('../../model_library')

if os.path.exists('../../results'):
	shutil.rmtree('../../results')
os.mkdir('../../results')

if os.path.exists('../../mask_library'):
	shutil.rmtree('../../mask_library')
os.mkdir('../../mask_library')





i = 0
for model in ['resnet56', 'resnet56_noshort', 'densenet121', 'resnet20', 'resnet20_noshort']:

	epoch_edge = {'resnet20':10, 'resnet20_noshort':20,
				  'resnet56':15, 'resnet56_noshort':15,
	              'resnet110':15, 'resnet110_noshort':20,
	              'densenet121': 25}

	epoch = {'resnet20':60, 'resnet20_noshort':120,
			 'resnet56':100, 'resnet56_noshort':200,
	         'resnet110':80, 'resnet110_noshort':150,
	         'densenet121': 200}
	batch = {'resnet20':64, 'resnet20_noshort':64,
			 'resnet56':128, 'resnet56_noshort':128,
	         'resnet110':64, 'resnet110_noshort':64,
	         'densenet121': 64}
	# command_tmp = 'python main_edge_train.py --epoch_edge ' + str(epoch_edge[model]) +' --dataset cifar10  --model ' + model
	command_tmp = 'python main_edge_train.py --epoch_edge 1 --dataset cifar10  --model ' + model +' --epoch ' + str(epoch[model])+' --batch_size ' + str(batch[model])

	print('command:\n', command_tmp)

	os.system(command_tmp)
	i = i + 1
	scipy.io.savemat('../../results/tuning_edge_{}_{}_cifar10.mat'.format(model, i), {'model': model, 'epoch':epoch_edge, 'task':args.task_division})
