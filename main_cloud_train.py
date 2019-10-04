"""
Author: Xiaocong Du
Date: August 2019
Project: Efficient edge learning with landscape visualization
Description: main function
"""

import logging
import os
import pickle
import sys
from builtins import enumerate, range

import scipy.io as scio
import continualNN
from load_cifar import *
from utils_tool import count_parameters_in_MB
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from args import parser
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu



log_path = 'log_main_cloud.txt'.format()
log_format = '%(asctime)s   %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M%p')
fh = logging.FileHandler(os.path.join('../../results/',log_path))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info("---------------------------------------------------------------------------------------------")
logging.info("                           Cloud: main.py                                             ")
logging.info("---------------------------------------------------------------------------------------------")
logging.info("args = %s", args)

method = continualNN.ContinualNN()
method.initial_single_network(init_weights = True )
method.initialization(args.lr, args.lr_step_size, args.weight_decay)

# -----------------------------------------
# Prepare dataset
# -----------------------------------------
task_list, total_task = method.create_task()
logging.info('Task list %s: ', task_list)

cloud_class = args.task_division[0]
task_id = 1 ## cloud task_id = 0, edge_task_id starts from 1
cloud_list = task_list[0 : args.task_division[0]]
total = 0
for i in range(task_id+1):
	total += args.task_division[i]
current_edge_list = task_list[args.task_division[0]: total]
all_list = task_list[ 0 : total]


train_cloud, test_cloud = get_dataset_cifar(cloud_list, 0)
for batch_idx, (data, target) in enumerate(train_cloud):
	logging.info('CLOUD re-assigned label: %s\n', np.unique(target))
	break

train_edge, test_edge = get_dataset_cifar(current_edge_list, args.task_division[0]+ (task_id-1)*args.task_division[task_id] )
for batch_idx, (data, target) in enumerate(train_edge):
	logging.info('EDGE re-assigned label: %s\n', np.unique(target))
	break

train_all, test_all = get_dataset_cifar(all_list, 0)
for batch_idx, (data, target) in enumerate(train_all):
	logging.info('ALL re-assigned label: %s\n', np.unique(target))
	break


# -----------------------------------------
#  Train cloud model
# -----------------------------------------
train_acc = []  #training accuracy. Length = number of total epochs
test_acc = []

for epoch in range(args.epoch):
	train_acc.append(method.train(epoch, train_cloud))
	test_acc.append(method.test(test_cloud))
	logging.info('train_acc {0:.4f}'.format(train_acc[-1]))
	logging.info('test_acc {0:.4f}\n\n'.format(test_acc[-1]))


logging.info('---------------------------------------------------------------------')
logging.info('test on EDGE data {0:.4f}'.format(method.test(test_edge)))
logging.info('test on CLOUD data {0:.4f}'.format(method.test(test_cloud)))
logging.info('test on CLOUD+EDGE data {0:.4f}\n\n\n'.format(method.test(test_all)))
logging.info('---------------------------------------------------------------------')

logging.info('Freeze {} weight of model {}'.format(args.task_division[0]/10, method.save_folder))
current_mask_list, current_threshold_dict, mask_dict_pre, maskR_dict_pre, current_taylor_dict = method.sensitivity_rank_taylor_filter(args.task_division[0]/10)
with open('../../mask_library/mask_model_' + method.save_folder + '.pickle', "wb") as f:
	pickle.dump((current_mask_list, current_threshold_dict, mask_dict_pre, maskR_dict_pre, current_taylor_dict), f)

# -----------------------------------------
#  Record model
# -----------------------------------------
# method.save_model(0, 0)
scio.savemat('../../results/cloud_training_model_{}.mat'.format(args.model), {'train_acc':train_acc, 'test_acc':test_acc, 'cloud_list':cloud_list, 'current_edge_list':current_edge_list})

avg_test = sum(test_acc[-10:])/10

x = np.linspace(0, len(test_acc), len(test_acc))
plt.figure(figsize=(20,10))
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(x, train_acc, 'k', alpha=0.5, label = 'Training accuracy, cloud')
plt.plot(x, test_acc, 'b',  alpha=0.5, label = 'Testing accuracy, cloud')
plt.yticks(np.arange(0, 1.0, step=0.1))
plt.grid(color='b', linestyle='-', linewidth=0.1)
plt.legend(loc='best')
plt.title('Learning curve')
plt.savefig('../../results/cloud_learning_curve_model_{}_acc{:.4f}.png'.format(args.model, avg_test))
param = count_parameters_in_MB(method.net)
logging.info('Param: %s MB',param)

# plt.show()
logging.info("args = %s", args)



