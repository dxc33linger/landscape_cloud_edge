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
# for task_division in ['90,10', '80,10,10', '70,10,10,10', '60,10,10,10,10', '50,10,10,10,10,10', '10,10,10,10,10,10,10,10,10,10']:
# for task_division in ['1,1,1,1,1,1,1,1,1,1','5,1,1,1,1,1', '6,1,1,1,1', '7,1,1,1', '8,1,1', '9,1']:
# for task_division in ['1,1', '2,1', '3,1', '4,1', '5,1', '6,1', '7,1', '8,1', '9,1']:
# for task_division in ['90,10', '80,10', '70,10', '60,10', '50,10', '40,10', '30,10', '20,10', '10,10']:

for model in ['vgg16', 'resnet20','resnet20_noshort', 'resnet56','resnet56_noshort', 'densenet121']:
	i = 0
	if not os.path.exists('../../results/{}'.format(model)):
		os.mkdir('../../results/{}'.format(model))

	for task_division in ['5,1,1,1,1,1']:

		dataset = 'cifar10'

		# ratio_c2e = int(int(task_division.split(',')[0]) / int(task_division.split(',')[1]))
		NA_C0 = 32 # not for vgg
		batch = 64
		epoch = 40# max(40, int(int(task_division.split(',')[0])* 10))
		epoch_edge = 15 #max(50, int(int(task_division.split(',')[0])+int(task_division.split(',')[1])* 1.0))


		command_tmp = 'python PST_main.py --gpu 0 --seed 1 --epoch ' + str(epoch) +' --epoch_edge '+str(epoch_edge)+' --dataset '+ dataset +' --model ' + model +' --batch_size ' + str(batch) + ' --NA_C0 ' +str(NA_C0) +' --task_division ' + task_division
		print('command:\n', command_tmp)
		os.system(command_tmp)


		i = int(task_division.split(',')[0]) if dataset == 'cifar10' else int(int(task_division.split(',')[0])/10)

		scipy.io.savemat('../../results/{}/tuning_{}.mat'.format(model,i), {'i':i, 'model': model, 'epoch':epoch, 'task':args.task_division, 'dataset':args.dataset, 'NA_C0': args.NA_C0, 'epoch_edge':epoch_edge})


