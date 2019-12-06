import os
import shutil
import scipy.io
from args import *
args = parser.parse_args()
import logging
import sys
#
# if os.path.exists('../../results'):
# 	shutil.rmtree('../../results')
# os.mkdir('../../results')
# if os.path.exists('../../model_library'):
# 	shutil.rmtree('../../model_library')
# os.mkdir('../../model_library')
# if os.path.exists('../../mask_library'):
# 	shutil.rmtree('../../mask_library')
# os.mkdir('../../mask_library')

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

log_path = 'run_main_cloud_train.txt'.format()
log_format = '%(asctime)s   %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M%p')
fh = logging.FileHandler(os.path.join('../../results/',log_path))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info("---------------------------------------------------------------------------------------------")
logging.info("                           run_main_cloud_train.py                                             ")
logging.info("---------------------------------------------------------------------------------------------")


i = 0
for model in ['densenet121']:

	NA_C0 = 16
	task_division = '9,1'
	batch_size = 64

	epoch = {'resnet20':50, 'resnet20_noshort':60,  'resnet110':80, 'resnet110_noshort':150, 'densenet121': 80, 'vgg11': 200, 'resnet56': 90, 'resnet56_noshort': 90}

	"""
	------------------------------------
	Run cloud model and save checkpoint
	------------------------------------
	"""

	command_tmp = 'python main_cloud_train.py --gpu 0 --epoch ' + str(epoch[model]) +' --dataset cifar10  --model ' + model + ' --batch_size ' + str(batch_size) + ' --NA_C0 ' +str(NA_C0) + ' --task_division ' + task_division
	logging.info('command: %s\n', command_tmp)
	os.system(command_tmp)

	print('command:\n', command_tmp)

	os.system(command_tmp)

