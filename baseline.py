"""
Author: Xiaocong Du
Description:
Title: Single-Net Continual Learning with Progressive Segmented Training (PST)
"""

import logging
import os
import pickle
import sys
import scipy.io as scio
import continualNN
from load_cifar import *
from utils_tool import count_parameters_in_MB
import matplotlib.pyplot as plt
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from args import parser
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu


log_path = 'log_main_edge.txt'.format()
log_format = '%(asctime)s   %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M%p')
fh = logging.FileHandler(os.path.join('../../baseline_library/',log_path))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info("*************************************************************************************************")
logging.info("                                         PST_main.py                                             ")
logging.info("*************************************************************************************************")
logging.info("args = %s", args)

method = continualNN.ContinualNN()
method.initial_single_network(init_weights = True)
method.initialization(args.lr, args.lr_step_size, args.weight_decay)

# -----------------------------------------
# Prepare dataset
# -----------------------------------------


task_list, _ = method.create_task()
logging.info('Task list %s: ', task_list)

task_division = []
for item in args.task_division.split(","):
	task_division.append(int(item))
total_task = len(task_division)

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
	logging.info('CLOUD re-assigned label: %s\n', np.unique(target))
	break
# #
num_epoch0 = args.epoch_edge
num_epoch1 = int(args.epoch_edge * 0.2)
num_epoch2 = int(args.epoch_edge * 0.5)
num_epoch3 = int(args.epoch_edge * 1)

# num_epoch0 = 2
# num_epoch1 = 2
# num_epoch2 = 2
# num_epoch3 = 2


train_acc = []
test_acc_0 = []
test_acc_current = []
test_acc_mix = []
test_task_accu = []  # At the end of each task, best overall test accuracy. Length = number of tasks
test_acc_0_end = []  # At the end of each task, the accuracy of task 0. Length = number of tasks


logging.info("==================================  Train task 0 ==========================================")
"""Test data from the first task"""

best_acc_0 = 0.0
for epoch in range(args.epoch):
	train_acc.append(method.train(epoch, train_cloud))
	test_acc_0.append(method.test(test_cloud))
	test_acc_current.append(np.zeros(1))
	test_acc_mix.append(method.test(test_cloud))

	if test_acc_0[-1] > best_acc_0:
		best_acc_0 = test_acc_0[-1]
	logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Single-head This training on T0 testing accu is : {:.4f}'.format(method.test(test_cloud)))
	logging.info('train_acc {0:.4f}\n\n\n'.format(train_acc[-1]))

test_task_accu.append(best_acc_0)

torch.save(method.net.state_dict(), '../../baseline_library/model_afterT{0}_Accu{1:.4f}.pt'.format(0, best_acc_0))

for task_id in range(1, total_task):
	logging.info("================================== 1. Current Task is {} : Prepare dataset ==========================================".format(task_id))
	# -----------------------------------------
	# Prepare dataset
	# -----------------------------------------
	total = 0
	for i in range(task_id+1):
		total += task_division[i]
	current_edge_list = task_list[(total-task_division[task_id]) : total] # 0 1 2 3 4,   5,   6,  7 taskid = 3 task_division=[5,1,1,[1],1,1]
	all_list = task_list[0 : total]
	all_data_list.append(current_edge_list)
	memory_each_task = int(args.total_memory_size / task_id) # The previous tasks shares the memory

	alltask_list = []
	alltask_memory = []
	alltask_single_list = []
	for i in range(task_id+1):
		alltask_list.append(task_list[i])
		alltask_memory.append(memory_each_task)
		alltask_single_list += task_list[i]

	alltask_memory = []
	for i in range(len(task_division)):
		alltask_memory.append(int(task_division[i] * args.total_memory_size / num_classes))
	logging.info('alltask_memory =  %s', alltask_memory)

	train_edge, test_edge = get_dataset_cifar(current_edge_list, task_division[0]+ (task_id-1)*task_division[task_id] )
	for batch_idx, (data, target) in enumerate(train_edge):
		logging.info('EDGE re-assigned label: %s\n', np.unique(target))
		break

	train_all, test_all = get_dataset_cifar(all_list, 0)
	for batch_idx, (data, target) in enumerate(train_all):
		logging.info('ALL re-assigned label: %s\n', np.unique(target))
		break

	"""Test data from all the tasks"""
	_, test_mix_full = get_dataset_cifar(alltask_single_list, 0)
	for batch_idx, (data, target) in enumerate(test_mix_full):
		logging.info('test_mix_full (all test data till now) re-assigned label: %s\n', np.unique(target))
		break


	logging.info("Current Task is {} : Train task without any technique to prevent forgetting====================".format(task_id))
	method.initialization(args.lr*0.1, args.lr_step_size, args.weight_decay)
	best_acc_mix = 0.0
	for epoch in range(args.epoch_edge):
		train_acc.append(method.train(epoch, train_edge))

		test_acc_0.append(method.test(test_cloud))
		test_acc_current.append(method.test(test_edge))
		test_acc_mix.append(method.test(test_mix_full))
		if test_acc_mix[-1] > best_acc_mix:
			best_acc_mix = test_acc_mix[-1]
		logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Single-head T0 testing accu is : {:.4f}'.format( test_acc_0[-1]))
		logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Single-head current testing accu is : {:.4f}'.format( test_acc_current[-1]))
		logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Single-head mixed all tasks testing accu is : {:.4f}'.format( test_acc_mix[-1]))
		logging.info('train_acc {0:.4f} \n\n\n'.format(train_acc[-1]))

	test_task_accu.append(best_acc_mix)

## RESULTS DOCUMENTATION
logging.info("====================== Document results ======================")

title_font = { 'size':'8', 'color':'black', 'weight':'normal'} # Bottom vertical alignment for more space
axis_font = { 'size':'10'}
plt.figure()
x = np.linspace(task_division[0], num_classes, num = len(test_task_accu))
plt.xlim(0, num_classes)
plt.xlabel('Task ID')
plt.ylabel('Accuracy')
plt.plot(x, test_task_accu , 'g-o', alpha=1.0, label = 'our method')
plt.yticks(np.arange(0, 1.0, step=0.1))
plt.xticks(np.arange(0, num_classes+1, step= 10))
plt.legend(loc='best')
plt.title('Incrementally learning {} classes at a time'.format(args.classes_per_task))
plt.savefig('../../baseline_library/incremental_curve_T{}_{:.4f}.png'.format(task_id, best_acc_mix))
# plt.title('Task: {} Model: {} \n Batch: {} Memory: {}\n Epoch_edge: {} ModelSize: {}'.format(task_division, args.model, args.batch_size, alltask_memory, args.epoch_edge, args.NA_C0), **title_font)
# plt.show()
text_acc_mix_noise = []
for idx in range(len(test_acc_mix)):
	text_acc_mix_noise.append(test_acc_mix[idx] + random.randint(-1, 1) * 0.01)


x = np.linspace(0, len(test_acc_mix), len(test_acc_mix))
plt.figure(figsize=(20,10))
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(x, train_acc, 'k', alpha=0.5, label = 'Training accuracy')
plt.plot(x, test_acc_mix, 'b',  alpha=0.5, label = 'Testing accuracy - mix')
plt.plot(x, text_acc_mix_noise, 'g',  alpha=0.5, label = 'Testing accuracy - noise')
plt.yticks(np.arange(0, 1.0, step=0.1))
plt.xticks(np.arange(0, len(test_acc_mix), step=10))
plt.grid(color='b', linestyle='-', linewidth=0.1)
plt.legend(loc='best')
plt.title('Learning curve')
plt.savefig('../../baseline_library/PSTmain_learning_curve_model_{}_acc{:.4f}.png'.format(args.model, best_acc_mix))


param = count_parameters_in_MB(method.net)
logging.info('Param:%s',param)
scio.savemat('../../baseline_library/PSTmain_model{}_acc{:.4f}.mat'.format(args.model, best_acc_mix),
             {'train_acc':train_acc, 'test_acc_0':test_acc_0,'test_acc_current':test_acc_current, 'test_acc_mix':test_acc_mix,
              'best_acc_mix':best_acc_mix, 'best_acc_0': best_acc_0,'model':args.model,
			'NA_C0':args.NA_C0, 'epoch': args.epoch, 'epoch_edge': args.epoch_edge, 'param':param,
            'lr': args.lr, 'lr_step_size':args.lr_step_size,
            'classes_per_task': args.classes_per_task, 'test_acc_0_end':test_acc_0_end, 'test_task_accu':test_task_accu,
            'weight_decay': args.weight_decay,  'score': args.score,
            'dataset':args.dataset, 'task_list': task_list, 'seed':args.seed, 'shuffle':args.shuffle,
            'num_epoch0':num_epoch0, 'num_epoch1':num_epoch1, 'num_epoch2':num_epoch2, 'num_epoch3':num_epoch3,
            'threshold_task_ratio':0, 'text_acc_mix_noise':text_acc_mix_noise})



