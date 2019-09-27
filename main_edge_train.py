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
import scipy.io as scio
import continualNN
from load_cifar import *
from utils_tool import count_parameters_in_MB
import matplotlib.pyplot as plt

from args import parser
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu



log_path = 'log_main_edge.txt'.format()
log_format = '%(asctime)s   %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M%p')
fh = logging.FileHandler(os.path.join('../../results/',log_path))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info("---------------------------------------------------------------------------------------------")
logging.info("                                   Edge: main.py                                             ")
logging.info("---------------------------------------------------------------------------------------------")
logging.info("args = %s", args)

method = continualNN.ContinualNN()
method.initial_single_network()
method.initialization(args.lr, args.lr_step_size, args.weight_decay)

# -----------------------------------------
# Prepare dataset
# -----------------------------------------
task_list, total_task = method.create_task()
logging.info('Task list %s: ', task_list)

assert sum(args.task_division) == 10
cloud_class = args.task_division[0]
cloud_list = task_list[0 : args.task_division[0]]

train_cloud, test_cloud = get_dataset_cifar(cloud_list, 0)
for batch_idx, (data, target) in enumerate(train_cloud):
	logging.info('CLOUD re-assigned label: %s\n', np.unique(target))
	break

test_task_accu = []
all_data_list = []
all_data_list.append(cloud_list)


# -----------------------------------------
#  Load cloud model
# -----------------------------------------
task_id = 0 ## cloud
save_folder = str(args.model) + '_lr=' + str(args.lr)+'_bs=' + str(args.batch_size)
model_file = '../loss-landscape/cifar10/trained_nets/' + save_folder + '_model_epoch' + str(args.epoch-1) + '.t7'

logging.info('==> Resuming from checkpoint..and test\n file path: {}'.format(model_file))
checkpoint = torch.load(model_file)
method.net.load_state_dict(checkpoint['state_dict'])

test_cloud = method.test(train_cloud)
logging.info('test on cloud data {0:.4f}\n'.format(test_cloud))
test_task_accu.append(test_cloud)

# -----------------------------------------
#  segment cloud model
# -----------------------------------------
ratio = args.task_division[0] / 10
logging.info('ratio: {}'.format(ratio))

logging.info('Freeze {} weight of model {}'.format(args.task_division[0] / 10, save_folder))
current_mask_list, current_threshold_dict, mask_dict_pre, maskR_dict_pre, current_taylor_dict = method.sensitivity_rank_taylor_filter(ratio	)
with open('../../mask_library/mask_model_' + save_folder + '_task_'+ str(task_id)+ '_top_'+ str(args.task_division[task_id] / 10) +'.pickle', "wb") as f:
	pickle.dump((current_mask_list, current_threshold_dict, mask_dict_pre, maskR_dict_pre, current_taylor_dict), f)

torch.save(method.net.state_dict(), '../../results/model_afterT{0}_Accu{1:.4f}.pt'.format(task_id, test_cloud))
logging.info('test on cloud data {0:.4f}\n'.format(test_cloud))


#
# print(method.net.state_dict()['conv1.weight'][0:3, 0, :, :])
# print(method.net.state_dict()['linear.weight'][:, 0:3])

train_acc = []  #training accuracy. Length = number of total epochs
test_acc = []

for task_id in range(1, len(args.task_division)):
	logging.info('---------------------Edge training task {} -----------------------'.format(task_id))
	# -----------------------------------------
	# Prepare dataset
	# -----------------------------------------
	total = 0
	for i in range(task_id+1):
		total += args.task_division[i]
	current_edge_list = task_list[(total-args.task_division[task_id]) : total] # 0 1 2 3 4,   5,   6,  7 taskid = 3 task_division=[5,1,1,[1],1,1]
	all_list = task_list[0 : total]
	all_data_list.append(current_edge_list)
	print(all_data_list)

	alltask_memory = [500, 100, 100, 100, 100, 100]
	train_edge, test_edge = get_dataset_cifar(current_edge_list, args.task_division[0]+ (task_id-1)*args.task_division[task_id] )
	for batch_idx, (data, target) in enumerate(train_edge):
		logging.info('EDGE re-assigned label: %s\n', np.unique(target))
		break

	train_all, test_all = get_dataset_cifar(all_list, 0)
	for batch_idx, (data, target) in enumerate(train_all):
		logging.info('ALL re-assigned label: %s\n', np.unique(target))
		break

	train_bm, _ = get_partial_dataset_cifar(0, all_data_list, num_images = alltask_memory)
	for batch_idx, (data, target) in enumerate(train_bm):
		print('train_bm (balanced memory) re-assigned label: \n', np.unique(target))
		break

	# -----------------------------------------
	#  Train edge model
	# -----------------------------------------

	for epoch in range(args.epoch_edge):
		train_acc.append(method.train_with_frozen_filter(epoch, train_edge, mask_dict_pre, maskR_dict_pre))
		test_acc.append(method.test(train_edge))
		logging.info('test_acc {0:.4f}\n'.format(test_acc[-1]))

		# print(method.net.state_dict()['conv1.weight'][0:3, 0, :, :])
		# print(method.net.state_dict()['linear.weight'][:, 0:3])

		logging.info('task {}'.format(task_id))
		logging.info('test on cloud data {0:.4f}\n'.format(method.test(train_cloud)))
		logging.info('test on edge data {0:.4f}\n'.format(method.test(train_edge)))
		logging.info('test on cloud+edge data {0:.4f}\n'.format(method.test(train_all)))


	print("Current Task is {} : Finetune FC with balanced memory =================================".format(task_id))
	# method.initialization(args.lr*0.5, int(num_epoch3*0.7), args.weight_decay)

	for epoch in range(1):
		train_acc.append(method.train_fc(epoch, train_bm))
		test_acc.append(method.test(train_edge))

		logging.info('task {}'.format(task_id))
		logging.info('test on cloud data {0:.4f}\n'.format(method.test(train_cloud)))
		logging.info('test on edge data {0:.4f}\n'.format(method.test(train_edge)))
		logging.info('test on cloud+edge data {0:.4f}\n'.format(method.test(train_all)))


	torch.save(method.net.state_dict(), '../../results/model_afterT{0}_Accu{1:.4f}.pt'.format(task_id, sum(test_acc[-3:-1]) / 3))
	# print(method.net.state_dict()['conv1.weight'][0:3, 0, :, :])
	# print(method.net.state_dict()['linear.weight'][:, 0:3])

	#
# 	# -----------------------------------------
# 	#  Segment model
# 	# -----------------------------------------
#
	method.mask_frozen_weight(maskR_dict_pre)
	current_mask_list, current_threshold_dict, mask_dict_current, maskR_dict_current, current_taylor_dict = method.sensitivity_rank_taylor_filter(
		args.task_division[task_id] / 10)
	mask_dict_pre, maskR_dict_pre = method.AND_twomasks(mask_dict_pre, mask_dict_current, maskR_dict_pre, maskR_dict_current)

	with open('../../mask_library/mask_model_' + save_folder + '_task_'+ str(task_id)+ '_top_'+ str(args.task_division[task_id] / 10) +'.pickle', "wb") as f:
		pickle.dump((current_mask_list, current_threshold_dict, mask_dict_pre, maskR_dict_pre, current_taylor_dict, mask_dict_current, maskR_dict_current), f)

	test_task_accu.append(sum(test_acc[-3:-1]) / 3)


	# break

## RESULTS DOCUMENTATION
print("====================== Document results ======================")

plt.figure()
x = np.linspace(0, len(test_task_accu), num = len(test_task_accu))
plt.xlim(0, len(test_task_accu)-1)
plt.xlabel('Task ID')
plt.ylabel('Accuracy')
plt.plot(x, test_task_accu , 'g-o', alpha=1.0, label = 'our method')
plt.yticks(np.arange(0, 1.0, step=0.1))
plt.legend(loc='best')
plt.title('Incrementally learning {} classes at a time'.format(args.classes_per_task))
plt.savefig('../../results/incremental_curve_model_{}_{:.4f}.png'.format(args.model, test_task_accu[-1]))
# plt.show()


#
scio.savemat('../../results/edge_training_model_{}.mat'.format(args.model), {'train_acc':train_acc, 'test_acc':test_acc, 'cloud_list':cloud_list, 'current_edge_list':current_edge_list, 'test_task_accu':test_task_accu})
#