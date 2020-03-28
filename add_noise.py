
import logging
import os
import pickle
import sys
import scipy.io as scio
import continualNN
from load_cifar import *
from utils_tool import count_parameters_in_MB
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from args import parser
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu


log_path = 'log_add_noise.txt'.format()
log_format = '%(asctime)s   %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M%p')
fh = logging.FileHandler(os.path.join('../../results/{}'.format(args.model),log_path))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info("*************************************************************************************************")
logging.info("                                         add_noise.py                                             ")
logging.info("*************************************************************************************************")
logging.info("args = %s", args)

method = continualNN.ContinualNN()
method.initial_single_network(init_weights = True)

# -----------------------------------------
# Prepare dataset
# -----------------------------------------


task_list, _ = method.create_task()
logging.info('Task list %s: ', task_list)

task_division = []
for item in args.task_division.split(","):
	task_division.append(int(item))
total_task = len(task_division)
logging.info('task_division %s', task_division)
if args.dataset == 'cifar10':
	num_classes = 10
	assert sum(task_division) <= 10

elif args.dataset == 'cifar100':
	num_classes = 100
	assert sum(task_division) <= 100


cloud_class = task_division[0]
task_id = 1
cloud_list = task_list[0 : task_division[0]]
total = 0
for i in range(task_id+1):
	total += task_division[i]
current_edge_list = task_list[task_division[0]: total]
all_list = task_list[0 : total]

all_data_list = []
all_data_list.append(cloud_list)


train_cloud, test_cloud = get_dataset_cifar(cloud_list, 0)
for batch_idx, (data, target) in enumerate(train_cloud):
	logging.info('Batch: {} CLOUD re-assigned train label: {}\n'.format(batch_idx, np.unique(target)))
	break
for batch_idx, (data, target) in enumerate(test_cloud):
	logging.info('Batch: {} CLOUD re-assigned test label: {}\n'.format(batch_idx, np.unique(target)))
	break
# #