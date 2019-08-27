"""
Author: Xiaocong Du
Date: April 2019 - May 2019
Project: Single-Net Continual Learning with Progressive Segmented Training (PST)
Description: main function
"""

import logging
import os
import pickle
import sys
import scipy.io as scio
import continualNN
from learning_curve import *
from load_cifar import *
from select_memory import Memorymaker
from utils_tool import count_parameters_in_MB
import matplotlib.pyplot as plt

from args import parser
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu

method = continualNN.ContinualNN()


log_path = 'log_SNC_main.txt'.format()
# if os.path.exists(os.path.join('../results/',log_path)):
# 	os.remove(os.path.join('../results/',log_path))
log_format = '%(asctime)s   %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M%p')
fh = logging.FileHandler(os.path.join('../results/',log_path))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info("*************************************************************************************************")
logging.info("                                         SNC_main.py                                             ")
logging.info("*************************************************************************************************")
logging.info("args = %s", args)

method.initial_single_network(init_weights = True )
method.initialization(args.lr, args.lr_step_size, args.weight_decay)

# torch.save(method.net,'../model_library/saved_network_for_visualization')
task_list, total_task = method.create_task()
logging.info('Task list %s: ', task_list)


num_epoch0 = args.num_epoch
num_epoch1 = int(args.num_epoch * 0.2)
num_epoch2 = int(args.num_epoch * 0.6)
num_epoch3 = int(args.num_epoch * 0.2)

train_acc = []  #training accuracy. Length = number of total epochs
test_acc_0 = []  # testing accuracy of task 0 , single-headed.Length = number of total epochs
test_acc_current = [] # testing accuracy of current task, single-headed.Length = number of total epochs
test_acc_mix = []  # overall accuracy, i.e., testing accuracy of full test data including all the previous dataset, single-headed. Length = number of total epochs
test_multihead_0 = [] # testing accuracy of task 0 , multi-headed. Length = number of total epochs
test_multihead_current = [] # testing accuracy of current task, multi-headed. Length = number of total epochs
test_task_accu = []  # At the end of each task, best overall test accuracy. Length = number of tasks
test_acc_0_end = []  # At the end of each task, the accuracy of task 0. Length = number of tasks
NME_accu_mix = []
NME_accu_0 =[]
NME_accu_current = []
logging.info("==================================  Train task 0 ==========================================")
"""Test data from the first task"""
task_id = 0
train_0, test_0 = get_dataset_cifar(task_list[task_id], 0*args.classes_per_task)
for batch_idx, (data, target) in enumerate(train_0):
	logging.info('task 0 %s\n', np.unique(target))
	break

best_acc_0 = 0.0
for epoch in range(num_epoch0):
	train_acc.append(method.train(epoch, train_0))
	test_acc_0.append(method.test(test_0))
	test_acc_current.append(np.zeros(1))
	test_acc_mix.append(np.zeros(1))
	test_multihead_0.append(np.zeros(1))
	test_multihead_current.append(np.zeros(1))

	if test_acc_0[-1] > best_acc_0:
		best_acc_0 = test_acc_0[-1]
	logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Single-head This training on T0 testing accu is : %.4f', method.test(test_0))
	logging.info('train_acc {0:.4f}\n\n\n'.format(train_acc[-1]))
method.save_model(0, 0)

current_mask_list, current_threshold_dict, mask_dict_pre, maskR_dict_pre, current_taylor_dict = method.sensitivity_rank_taylor_filter(args.threshold_task)
with open('../mask_library/mask_task{}_threshold{}_acc{:.4f}.pickle'.format(0, args.threshold_task, best_acc_0), "wb") as f:
	pickle.dump((current_mask_list, current_threshold_dict, mask_dict_pre, maskR_dict_pre, current_taylor_dict), f)
# torch.save(method.net.state_dict(), '../results/model_beforeT{0}_Accu{1:.4f}.pt'.format(0, best_acc_0))
# method.PCA_feature(train_0, task_id = 0,
#                    title = '../results/PCA_feature_task{}_acc{:.3f}'.format(task_id, best_acc_0))

memory_img_list, memory_target_list = method.initial_memory()
batch_size_ = int(50000 / total_task)
train_current = get_dataset_cifar_noAug(task_list[0],  task_id*args.classes_per_task,  batch_size_, shuffle_ = False)
memory_img_list, memory_target_list = method.select_memory_for_currentTask(train_current, memory_img_list, memory_target_list, task_id = 0)

logging.info("================================== task 0: Clear cut 0 of task 0 ==========================================")
method.initial_single_network(init_weights = True)
method.initialization(args.lr, args.lr_step_size, args.weight_decay)
method.load_model_random_initial(maskR_dict_pre, mask_dict_pre, model_id=0, task_id= 0)  # initialize top important args.threshold_task0 weights and retrain from scratch

for epoch in range(num_epoch2):
	train_acc.append(method.train_with_frozen_filter(epoch, train_0, maskR_dict_pre, mask_dict_pre))
	test_acc_0.append(method.test(test_0))
	test_acc_current.append(np.zeros(1))
	test_acc_mix.append(np.zeros(1))
	test_multihead_0.append(np.zeros(1))
	test_multihead_current.append(np.zeros(1))
	logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Single-head This training on T0 testing accu is : %.4f', method.test(test_0))
	logging.info('Cut T0 train_acc {0:.4f}\n\n\n'.format(train_acc[-1]))
	if test_acc_0[-1] > best_acc_0:
		best_acc_0 = test_acc_0[-1]
test_task_accu.append(best_acc_0)
test_acc_0_end.append(best_acc_0)

torch.save(method.net.state_dict(), '../results/model_afterT{0}_Accu{1:.4f}.pt'.format(0, best_acc_0))



for task_id in range(1, total_task):
	logging.info("================================== 1. Current Task is {} : Prepare dataset ==========================================".format(task_id))

	"""Current trainset, e.g. task id = 2, the 3rd task"""
	# task_id = 1
	train_current, test_current = get_dataset_cifar(task_list[task_id], task_id*args.classes_per_task)
	for batch_idx, (data, target) in enumerate(train_current):
		logging.info('train_current re-assigned label: %s\n', np.unique(target))
		break

	# """Balance memory: same amounts of images from previous tasks and current task"""
	memory_each_task = int(args.total_memory_size / task_id) # The previous tasks shares the memory
	alltask_list = []
	alltask_memory = []
	alltask_single_list = []
	for i in range(task_id+1):
		alltask_list.append(task_list[i]) # alltask_list = [task_list[0], task_list[1], ...task_list[task_id]]
		alltask_memory.append(memory_each_task) # E.g., alltask_memory = [1000, ....1000]
		alltask_single_list += task_list[i]
	logging.info('Memory capacity %s', alltask_memory)
	train_bm, _ = get_partial_dataset_cifar(0, alltask_list, num_images = alltask_memory)
	for batch_idx, (data, target) in enumerate(train_bm):
		logging.info('train_memory (balanced memory) re-assigned label: %s\n', np.unique(target))
		break

	"""Test data from all the tasks"""
	_, test_mix_full = get_dataset_cifar(alltask_single_list, 0)
	for batch_idx, (data, target) in enumerate(test_mix_full):
		logging.info('test_mix_full (all test data till now) re-assigned label: %s\n', np.unique(target))
		break

	# memory_img_array, memory_target_array= method.select_balanced_memory(memory_img_list, memory_target_list, task_id)
	# memory_trainset = Memorymaker(memory_img_array, memory_target_array)
	# train_memory = method.memory_to_dataloader(memory_trainset)
	# for batch_idx, (data, target) in enumerate(train_memory):
	# 	logging.info('train_memory with nearest distance re-assigned label: %s\n', np.unique(target))
		# break
	## memory_for_NME = get_memory_cifar(0, alltask_list, num_images = alltask_memory, batch_size = args.batch_size)

	logging.info("=============================== 2. Current Task is {} : Memory-assisted balancing ==================================".format(task_id))
	method.initialization(args.lr, int(num_epoch1), args.weight_decay)
	for epoch in range(num_epoch1):
		train_acc.append(method.train_with_frozen_filter(epoch, train_bm, mask_dict_pre, maskR_dict_pre))
		test_acc_0.append(method.test(test_0))
		test_acc_current.append(method.test(test_current))
		test_acc_mix.append(method.test(test_mix_full))
		test_multihead_0.append(method.test_multihead(0, test_0))
		test_multihead_current.append(method.test_multihead(task_id, test_current))
		logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Single-head T0 testing accu is : %.4f', test_acc_0[-1])
		logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Multi-headed T0 testing accu is : %.4f', test_multihead_0[-1])
		logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Single-head current testing accu is : %.4f', test_acc_current[-1])
		logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Multi-headed current testing accu is : %.4f', test_multihead_current[-1])
		logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Single-head mixed all tasks testing accu is : %.4f', test_acc_mix[-1])
		logging.info('train_acc {0:.4f} \n\n\n'.format(train_acc[-1]))


	logging.info("Current Task is {} : Train task 1 w/ injecting memory iteratively ===================================".format(task_id))
	method.initialization(args.lr, args.lr_step_size, args.weight_decay)
	for epoch in range(num_epoch2):
		if epoch % 3 == 0 or epoch % 5 == 0 or epoch > num_epoch2 - 3:
			# train_acc.append(method.train_with_frozen_filter(epoch, train_memory, mask_dict_pre, maskR_dict_pre))
			train_acc.append(method.train_with_frozen_filter(epoch, train_bm, mask_dict_pre, maskR_dict_pre))

		else:
			train_acc.append(method.train_with_frozen_filter(epoch, train_current, mask_dict_pre, maskR_dict_pre))

		test_acc_0.append(method.test(test_0))
		test_acc_current.append(method.test(test_current))
		test_acc_mix.append(method.test(test_mix_full))
		test_multihead_0.append(method.test_multihead(0, test_0))
		test_multihead_current.append(method.test_multihead(task_id, test_current))

		logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Single-head T0 testing accu is : %.4f', test_acc_0[-1])
		logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Multi-headed T0 testing accu is : %.4f', test_multihead_0[-1])
		logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Single-head current testing accu is : %.4f', test_acc_current[-1])
		logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Multi-headed current testing accu is : %.4f', test_multihead_current[-1])
		logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Single-head mixed all tasks testing accu is : %.4f', test_acc_mix[-1])
		logging.info('train_acc {0:.4f} \n\n\n'.format(train_acc[-1]))


	logging.info("Current Task is {} : Retrain FC with balanced memory ===================================".format(task_id))
	method.initialization(args.lr*0.5, int(num_epoch3), args.weight_decay)
	best_acc_mix = 0.0

	for epoch in range(num_epoch3):
		train_acc.append(method.train_fc(epoch, train_bm))

		test_acc_0.append(method.test(test_0))
		test_acc_current.append(method.test(test_current))
		test_acc_mix.append(method.test(test_mix_full))
		test_multihead_0.append(method.test_multihead(0, test_0))
		test_multihead_current.append(method.test_multihead(task_id, test_current))
		if test_acc_mix[-1] > best_acc_mix:
			best_acc_mix = test_acc_mix[-1]
		logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Single-head T0 testing accu is : %.4f', test_acc_0[-1])
		logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Multi-headed T0 testing accu is : %.4f', test_multihead_0[-1])
		logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Single-head current testing accu is : %.4f', test_acc_current[-1])
		logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Multi-headed current testing accu is : %.4f', test_multihead_current[-1])
		logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Single-head mixed all tasks testing accu is : %.4f', test_acc_mix[-1])
		logging.info('train_acc {0:.4f} \n\n\n'.format(train_acc[-1]))

	test_task_accu.append(best_acc_mix)
	test_acc_0_end.append(test_acc_0[-1])

	logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> At the end of task {}, T0 accu is {:.4f}'.format(task_id, test_acc_0[-1]))
	method.save_model(0, task_id)
	torch.save(method.net.state_dict(), '../results/model_afterT{0}_Accu{1:.4f}.pt'.format(task_id, best_acc_mix))

	# method.PCA_feature(train_current, task_id,
	#                    title='../results/PCA_feature_task{}_acc{:.3f}'.format(task_id, best_acc_0))



	# train_current = get_dataset_cifar_noAug(task_list[task_id], task_id * args.classes_per_task, batch_size_, shuffle_=False)
	# memory_img_list, memory_target_list = method.select_memory_for_currentTask(train_current, memory_img_list, memory_target_list, task_id)
	# train_current, test_current = get_dataset_cifar(task_list[task_id], task_id*args.classes_per_task)





	if task_id != total_task-1:
		logging.info("===================================== 3.  Current Task is {} : importance sampling ====================================".format(task_id))
		method.mask_frozen_weight(maskR_dict_pre)

		current_mask_list, current_threshold_dict, mask_dict_current, maskR_dict_current, current_taylor_dict = method.sensitivity_rank_taylor_filter(args.threshold_task)
		with open('../mask_library/mask_task{}_threshold{}_acc{:.4f}.pickle'.format(task_id, args.threshold_task, best_acc_mix), "wb") as f:
			pickle.dump((current_mask_list, current_threshold_dict, mask_dict_current, maskR_dict_current, current_taylor_dict, mask_dict_pre, maskR_dict_pre), f)


		logging.info("===================================== 4. Current Task is {} : model segmentation ==========================================".format(task_id))
		method.initial_single_network(init_weights=True)
		method.initialization(args.lr, args.lr_step_size-10, args.weight_decay)
		method.load_model_random_initial(maskR_dict_current, mask_dict_current, model_id=0, task_id = task_id)  # initialize top important args.threshold_task0 weights and retrain from scratch

		for epoch in range(num_epoch2):
			if epoch % 3 == 0 or epoch % 5 == 0 or  epoch > num_epoch2 - 3:
				train_acc.append(method.train_with_frozen_filter(epoch, train_bm, maskR_dict_current, mask_dict_current))
			else:
				train_acc.append(method.train_with_frozen_filter(epoch, train_current, maskR_dict_current, mask_dict_current))

			test_acc_0.append(method.test(test_0))
			test_acc_current.append(method.test(test_current))
			test_acc_mix.append(method.test(test_mix_full))
			test_multihead_0.append(method.test_multihead(0, test_0))
			test_multihead_current.append(method.test_multihead(task_id, test_current))

			logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Single-head T0 testing accu is : %.4f', test_acc_0[-1])
			logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Multi-headed T0 testing accu is : %.4f', test_multihead_0[-1])
			logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Single-head current testing accu is : %.4f', test_acc_current[-1])
			logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Multi-headed current testing accu is : %.4f', test_multihead_current[-1])
			logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Single-head mixed all tasks testing accu is : %.4f', test_acc_mix[-1])


		logging.info("Current Task is {} : Combine masks  ==========================================".format(task_id))
		mask_dict_pre, maskR_dict_pre = method.AND_twomasks(mask_dict_pre, mask_dict_current, maskR_dict_pre, maskR_dict_current)

	# break


## RESULTS DOCUMENTATION
logging.info("====================== Document results ======================")
train_acc = np.asarray(train_acc)
test_acc_0 = np.asarray(test_acc_0)
test_acc_current = np.asarray(test_acc_current)
test_acc_mix = np.asarray(test_acc_mix)
test_acc = np.stack((test_acc_0, test_acc_current, test_acc_mix))

test_multihead_0 = np.asarray(test_multihead_0)
test_multihead_current = np.asarray(test_multihead_current)
test_multihead = np.stack((test_multihead_0, test_multihead_current))
logging.info('Each task has epoch:{}'.format(num_epoch2*2+num_epoch3+num_epoch1))
logging.info('Total num_epochs : {}, current task ID: {} '.format(test_acc.shape[1], task_id))
file_name = '../results/SNC_Accu_T0_{0:.4f}-Task{1}_[{2:.4f}]'.format(best_acc_0, task_id, best_acc_mix)

x = np.linspace(0, test_acc.shape[1], test_acc.shape[1])
plt.figure(figsize=(20,10))
plt.xlabel('Epoch')
plt.ylabel('Testing Accuracy')
plt.plot(x, train_acc[:] , 'g', alpha=0.5, label = 'Training accuracy')
plt.plot(x, test_acc[0,:], 'y',  alpha=0.5, label = 'Testing accuracy on T0')
plt.plot(x, test_acc[1,:], 'm',  alpha=0.5, label = 'Testing accuracy on current task')
plt.plot(x, test_acc[2,:], 'b',  alpha=1.0, label = 'Testing accuracy on all the tasks')
plt.yticks(np.arange(0, 1.0, step=0.1))
plt.grid(color='b', linestyle='-', linewidth=0.1)
plt.legend(loc='best')
plt.title('Learning curve')
plt.savefig('../results/learning_curve_T0_{0:.4f}-Task{1}_{2:.4f}.png'.format(best_acc_0, task_id, best_acc_mix))
# plt.show()
plt.figure()
x = np.linspace(0, len(test_task_accu), num = len(test_task_accu))
plt.xlim(0, 100/args.classes_per_task)
plt.xlabel('Task ID')
plt.ylabel('Accuracy')
plt.plot(x, test_task_accu , 'g-o', alpha=1.0, label = 'our method')
plt.yticks(np.arange(0, 1.0, step=0.1))
plt.legend(loc='best')
plt.title('Incrementally learning 10 classes at a time')
plt.savefig('../results/incremental_curve_Task{}_Accu{:.4f}.png'.format(task_id, best_acc_mix))
# plt.show()
param = count_parameters_in_MB(method.net)
logging.info('Param:%s',param)
scio.savemat(file_name+'.mat', {'train_acc':train_acc, 'test_acc':test_acc,'best_acc_mix':best_acc_mix, 'best_acc_0': best_acc_0,
								'NA_C0':args.NA_C0, 'num_epoch': args.num_epoch, 'param':param,
                                'lr': args.lr, 'lr_step_size':args.lr_step_size, 'test_multihead':test_multihead,
                                'classes_per_task': args.classes_per_task, 'test_acc_0_end':test_acc_0_end, 'test_task_accu':test_task_accu,
                                'weight_decay': args.weight_decay,  'score': args.score,
                                'dataset':args.dataset, 'task_list': task_list, 'seed':args.seed, 'shuffle':args.shuffle,
                                'num_epoch0':num_epoch0, 'num_epoch1':num_epoch1, 'num_epoch2':num_epoch2, 'num_epoch3':num_epoch3,
                                'threshold_task':args.threshold_task,
                                'NME_accu_mix':NME_accu_mix, 'NME_accu_0':NME_accu_0, 'NME_accu_current':NME_accu_current})



