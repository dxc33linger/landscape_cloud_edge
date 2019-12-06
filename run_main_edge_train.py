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
for model in ['densenet121']: #'resnet20', 'resnet20_noshort','resnet56', 'resnet56_noshort',

	epoch_edge = {'resnet20':15, 'resnet20_noshort':20,
				  'resnet56':20, 'resnet56_noshort':20,
	              'resnet110':20, 'resnet110_noshort':20,
	              'densenet121': 25}

	epoch = {'resnet20':60, 'resnet20_noshort':120,  'resnet110':80, 'resnet110_noshort':150, 'densenet121': 125, 'vgg11': 200,
	         'resnet56': 80, 'resnet56_noshort': 150}


	batch = {'resnet20':64, 'resnet20_noshort':64,
			 'resnet56':64, 'resnet56_noshort':64,
	         'resnet110':64, 'resnet110_noshort':64,
	         'densenet121': 64}

	NA_C0 = 32
	command_tmp = 'python main_edge_train.py --gpu 1 --epoch_edge ' + str(epoch_edge[model]) +' --dataset cifar10  --model ' + model +' --epoch ' +  str(epoch[model])+' --batch_size ' + str(batch[model]) + ' --NA_C0 ' +str(NA_C0)


	print('command:\n', command_tmp)

	os.system(command_tmp)
	i = i + 1
	scipy.io.savemat('../../results/tuning_edge_{}_{}_cifar10.mat'.format(model, i), {'model': model, 'epoch':epoch_edge, 'task':args.task_division})
