"""
Author: Xiaocong Du
Date: April 2019 - May 2019
Project: Single-Net Continual Learning withProgressive Segmented Training (PST)
Description: functions
"""

import torch
import logging
import os
import random
import re
from collections import OrderedDict
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import torch.optim as optim
from sklearn import decomposition
from sklearn.manifold import TSNE
from torch.autograd import Variable
import pickle
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from args import parser
from scipy.spatial.distance import cdist
from utils_tool import progress_bar
from cifar10 import model_loader



args = parser.parse_args()


class ContinualNN(object):
	def __init__(self):
		self.batch_size = args.batch_size
		self.model_path = '../../model_library'
		self.baseline_path = '../../baseline_library/'
		if not os.path.exists(self.baseline_path):
			os.mkdir(self.baseline_path)
		if not os.path.exists('../../results/'):
			os.mkdir('../../results')
		if not os.path.exists('../../mask_library/'):
			os.mkdir('../../mask_library')

	def initial_single_network(self, NA_C0 = args.NA_C0, init_weights = True):
		# self.net = resnet32(NA_C0, init_weights)
		# # self.net = resnet18(NA_C0, init_weights)
		# logging.info('Network: ResNet32, init_weights=True')
		self.net = model_loader.load(args.model)
		print(self.net)
		return self.net


	def initialization(self, lr, lr_step_size, weight_decay):
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.net = self.net.to(self.device)
		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = optim.SGD(self.net.parameters(),  lr = lr,  momentum = 0.9, weight_decay = weight_decay)
		self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size = lr_step_size, gamma= args.lr_gamma)


	def create_task(self):
		if args.dataset == 'cifar10':
			total_classes = 10
		elif args.dataset == 'cifar100':
			total_classes = 100
		# random select label
		a = list(range(0, total_classes))
		if args.shuffle:
			random.seed(args.seed)
			random.shuffle(a)
		else:
			a = a
		task_list = []
		for i in range(0, len(a), args.classes_per_task):
			task_list.append(a[i:i + args.classes_per_task])
		self.task_list = task_list
		self.total_num_task = int(total_classes / args.classes_per_task)
		return self.task_list, self.total_num_task


	def train(self, epoch, trainloader):
		logging.info('\nEpoch: %d lr: %s' % (epoch, self.scheduler.get_lr()))
		self.net.train()
		train_loss = 0.0
		correct = 0
		total = 0
		self.optimizer.step()

		self.scheduler.step()

		for batch_idx, (inputs, targets) in enumerate(trainloader):
			inputs, targets = inputs.to(self.device), targets.to(self.device)
			inputs_var = Variable(inputs)
			targets_var = Variable(targets)

			self.optimizer.zero_grad()
			outputs = self.net(inputs_var)
			loss = self.criterion(outputs, targets_var)

			loss.backward()
			self.optimizer.step()

			train_loss += loss.item()
			_, predicted = outputs.max(1) # outputs.shape: (batch, classes)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()
			self.loss = train_loss
			acc = 100.*correct/total
			progress_bar(batch_idx, len(trainloader), 'Loss:%.3f|Acc:%.3f%% (%d/%d)--Train' % (train_loss/(batch_idx+1), acc, correct, total))

		if epoch == args.epoch - 1:
			self.save_checkpoint_t7(epoch, acc, train_loss)
		return correct/total

	def save_checkpoint_t7(self, epoch, acc, loss, postfix = '', path_postfix=''):
		self.save_folder = self.name_save_folder(args)
		state = {
			'acc': acc,
			'loss': loss,
			'epoch': epoch,
			'state_dict': self.net.module.state_dict() if args.ngpu > 1 else self.net.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),

		}
		opt_state = {
			'optimizer': self.optimizer.state_dict()
		}
		path = 'trained_nets' + path_postfix + '/'
		self.model_file = '../loss-landscape/cifar10/'+path+self.save_folder+'_model_epoch'+ str(epoch) + postfix +'.t7'
		logging.info('Saving checkpiont to ' + self.model_file)
		torch.save(state, self.model_file)
		# torch.save(opt_state, '../loss-landscape/cifar10/trained_nets/'+self.save_folder+'_opt_state_epoch' + str(epoch) + '.t7')


	def name_save_folder(self, args):
		save_folder = args.model + '_lr=' + str(args.lr)
		# if args.lr_decay != 0.1:
		# 	self.save_folder += '_lr_decay=' + str(args.lr_decay)
		save_folder += '_bs=' + str(args.batch_size)
		# self.save_folder += '_wd=' + str(args.weight_decay)
		# self.save_folder += '_mom=' + str(args.momentum)
		# self.save_folder += '_save_epoch=' + str(args.save_epoch)
		return save_folder
		# if args.loss_name != 'crossentropy':
		# 	self.save_folder += '_loss=' + str(args.loss_name)
		# if args.noaug:
		# 	self.save_folder += '_noaug's
		# if args.raw_data:
		# 	self.save_folder += '_rawdata'
		# if args.label_corrupt_prob > 0:
		# 	self.save_folder += '_randlabel=' + str(args.label_corrupt_prob)
		# if args.ngpu > 1:
		# 	self.save_folder += '_ngpu=' + str(args.ngpu)
		# if args.idx:
		# 	self.save_folder += '_idx=' + str(args.idx)
	def train_fc(self, epoch, trainloader):
		for name, param in self.net.named_parameters():
			if re.search('conv', name) or re.search('bn', name):
				param.requires_grad = False
			elif re.search('linear', name):
				param.requires_grad = True

		self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

		logging.info('\nEpoch: %d lr: %s' % (epoch, self.scheduler.get_lr()))
		self.optimizer.step()
		self.scheduler.step()
		self.net.train()
		train_loss = 0.0
		correct = 0
		total = 0

		for batch_idx, (inputs, targets) in enumerate(trainloader):
			inputs, targets = inputs.to(self.device), targets.to(self.device)
			inputs_var = Variable(inputs)
			targets_var = Variable(targets)
			self.optimizer.zero_grad()
			outputs = self.net(inputs_var)
			loss = self.criterion(outputs, targets_var)
			loss.backward()
			self.optimizer.step()
			train_loss += loss.item()
			_, predicted = outputs.max(1)  # outputs.shape: (batch, classes)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()
			self.loss = train_loss
			progress_bar(batch_idx, len(trainloader), 'Loss:%.3f|Acc:%.3f%% (%d/%d)--Train' % (
			train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
		return correct / total


	def initialize_fc(self):
		param_old = OrderedDict([(k, None) for k in self.net.state_dict().keys()])
		for layer_name, param in self.net.state_dict().items():
			if re.search('linear', layer_name):
				param_old[layer_name] = nn.init.normal_(param.clone(), 0, 0.01)
			else:
				param_old[layer_name] = param.clone()
		self.net.load_state_dict(param_old)


	def test(self, testloader):
		self.net.eval()
		test_loss = 0.0
		correct = 0
		total = 0
		with torch.no_grad():
			for batch_idx, (inputs, targets) in enumerate(testloader):
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				inputs = Variable(inputs)
				targets = Variable(targets)

				outputs = self.net(inputs)
				loss = self.criterion(outputs, targets)

				test_loss += loss.item()
				_, predicted = outputs.max(1)
				total += targets.size(0)
				correct += predicted.eq(targets).sum().item()

				progress_bar(batch_idx, len(testloader), 'Loss:%.3f|Acc:%.3f%% (%d/%d)--Test'
							 % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

			# print('target', targets[0:13])
			# print('predicted', predicted[0:13])
			# logging.info('output %s', outputs[3, 0:20])
			# logging.info('output_tmp %s', output_tmp[3, 0:20])
		return correct/total



	def save_model(self, model_id, task_id, threshold_task):
		if not os.path.isdir(self.model_path):
			os.mkdir(self.model_path)

		file_name = "Task{}_model{}_classes{}_shf{}_top{}.pickle".format(task_id, model_id, args.classes_per_task, args.shuffle, threshold_task)
		path = os.path.join(self.model_path, file_name)
		pickle.dump(self.net.state_dict(), open(path, 'wb'))

		logging.info('save_model: model {} of Task {} is saved in [{}]\n'.format(model_id, task_id, path) )


	def save_baseline(self):
		file_name = "Baseline_ResNet32_Cifar100_NAC0{}_classes{}_shf{}.pickle".format(args.NA_C0, args.classes_per_task,args.shuffle)
		path = os.path.join(self.baseline_path, file_name)
		pickle.dump(self.net.state_dict(), open(path, 'wb'))
		logging.info('Baseline is saved in [{}]\n'.format(path) )


	def load_model(self, model_id, task_id, to_net, threshold_task):
		file_name = "Task{}_model{}_classes{}_shf{}_top{}.pickle".format(task_id, model_id, args.classes_per_task, args.shuffle, threshold_task)
		path = os.path.join(self.model_path, file_name)
		param_model_dict = pickle.load(open(path, 'rb'))
		# assert param_model_dict['linear.weight'].get_device() == self.net.state_dict()['linear.weight'].get_device(), "parameter and net are not in same device"
		to_net.load_state_dict(param_model_dict)
		logging.info('load_model: Loading {}....'.format(path) )



	def load_model_random_initial(self, save_mask_file, save_mask_fileR, model_id, task_id, threshold_task):
		try:
			mask_dict = pickle.load(open(save_mask_file, "rb"))
			mask_reverse_dict = pickle.load(open(save_mask_fileR, "rb"))
		except TypeError:
			mask_dict = save_mask_file
			mask_reverse_dict = save_mask_fileR

		param_random = OrderedDict([(k, None) for k in self.net.state_dict().keys()])

		for layer_name, param in self.net.state_dict().items():
			param_random[layer_name] = Variable(param.type(torch.cuda.FloatTensor).clone(), requires_grad = True)

		param_processed = OrderedDict([(k,None) for k in self.net.state_dict().keys()])

		self.load_model(model_id, task_id, self.net, threshold_task)

		for layer_name, param_model in self.net.state_dict().items():
			param_model = Variable(param_model.type(torch.cuda.FloatTensor), requires_grad = True)
			if layer_name in mask_dict.keys(): # if layer_name from networkA, load model with mask, randomly initialize the rest
				param_processed[layer_name] = Variable(torch.mul(param_model, mask_dict[layer_name]) + torch.mul(param_random[layer_name], mask_reverse_dict[layer_name]), requires_grad = True)
			else: # if new network, randomly initialize
				param_processed[layer_name] = Variable(param_random[layer_name], requires_grad = True)
		assert param_processed[layer_name].get_device() == self.net.state_dict()[layer_name].get_device(), "parameter and net are not in same device"

		self.net.load_state_dict(param_processed)
		logging.info('load_model_random_initial: Random initialize masked weights.\n')
		return self.net


	def hist_accuracy(self, y, title, x_lim, x_label='Accuracy', bin=200):
		plt.figure()
		plt.hist(y, bins=bin, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
		plt.xlabel(x_label)
		plt.ylabel("Count")
		plt.xlim(x_lim[0], x_lim[1])
		# plt.ylim(0, 700)
		plt.title("Histogram: "+ title)
		plt.show()
		plt.savefig('../results/histogram'+title+'.png')


	def train_with_frozen_filter(self, epoch, trainloader, mask_dict, mask_dict_R, path_postfix=''):

		param_old_dict = OrderedDict([(k, None) for k in self.net.state_dict().keys()])
		for layer_name, param in self.net.state_dict().items():
			param_old_dict[layer_name] = param.clone()

		self.net.train()
		logging.info('\nEpoch: %d lr: %s' % (epoch, self.scheduler.get_lr()))
		train_loss = 0.0
		correct = 0
		total = 0
		self.optimizer.step()
		self.scheduler.step()

		for batch_idx, (inputs, targets) in enumerate(trainloader):
			inputs, targets = inputs.to(self.device), targets.to(self.device)
			inputs_var = Variable(inputs)
			targets_var = Variable(targets)

			self.optimizer.zero_grad()
			outputs = self.net(inputs_var)
			loss = self.criterion(outputs, targets)

			loss.backward()
			self.optimizer.step()

			train_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()
			acc = 100. * correct / total
			# apply mask
			param_processed = OrderedDict([(k, None) for k in self.net.state_dict().keys()])
			for layer_name, param_new in self.net.state_dict().items():
				param_new = param_new.type(torch.cuda.FloatTensor)
				param_old_dict[layer_name] = param_old_dict[layer_name].type(torch.cuda.FloatTensor)
				# print(layer_name)
				if re.search('conv', layer_name):
					param_processed[layer_name] = Variable(torch.mul(param_old_dict[layer_name], mask_dict[layer_name]) +
														   torch.mul(param_new, mask_dict_R[layer_name]), requires_grad=True)



					# print('new\n', param_new[0:3, 0, :, :])

				elif re.search('shortcut', layer_name):
					if len(param_new.shape) == 4:  # conv in shortcut
						param_processed[layer_name] = Variable(torch.mul(param_old_dict[layer_name], mask_dict[layer_name]) +
														   torch.mul(param_new, mask_dict_R[layer_name]), requires_grad=True)
					else:
						param_processed[layer_name] = Variable(param_new, requires_grad=True)
				elif re.search('linear', layer_name):
					param_processed[layer_name] = Variable(torch.mul(param_old_dict[layer_name], mask_dict[layer_name]) +
						torch.mul(param_new, mask_dict_R[layer_name]), requires_grad=True)

				else:
					param_processed[layer_name] = Variable(param_new, requires_grad=True)  # num_batches_tracked

			# print('old\n', param_old_dict['conv1.weight'][0:3, 0, :, :])
			# print('mask\n', mask_dict['conv1.weight'][0:3, 0, :, :])
			# print('mask_R\n', mask_dict_R['conv1.weight'][0:3, 0, :, :])
			# print('param_processed\n', param_processed['conv1.weight'][0:3, 0, :, :])



			self.net.load_state_dict(param_processed)
			progress_bar(batch_idx, len(trainloader), 'Loss:%.3f|Acc:%.3f%% (%d/%d)--Train' % (
			train_loss / (batch_idx + 1), acc, correct, total))


		if epoch == 0 or epoch == args.epoch_edge - 1 or epoch == args.epoch_edge // 2:
			self.save_checkpoint_t7(epoch, acc, train_loss, '_edge_model', path_postfix,)
		return correct / total






	def test_multihead(self, task_id, testloader):
		self.net.eval()
		test_loss = 0.0
		correct = 0
		total = 0
		with torch.no_grad():
			for batch_idx, (inputs, targets) in enumerate(testloader):
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				inputs = Variable(inputs)
				targets = Variable(targets)

				outputs = self.net(inputs)
				loss = self.criterion(outputs, targets)

				test_loss += loss.item()
				_, predicted = outputs[:, args.classes_per_task*task_id:args.classes_per_task*(task_id+1)].max(1)
				total += targets.size(0)
				correct += (predicted+args.classes_per_task*task_id).eq(targets).sum().item()

				progress_bar(batch_idx, len(testloader), 'Loss:%.3f|Acc:%.3f%% (%d/%d)--Test'
							 % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

		return correct/total



	def sensitivity_rank_taylor_filter(self, threshold):
		self.net.eval()
		mask_list_4d = []
		mask_list_R_4d = []
		threshold_list = []
		gradient_list = []
		weight_list = []
		taylor_list = []
		i = 0
		logging.info("Obtain top {} position according to {} ........".format(threshold, args.score))

		for m in self.net.modules():
			# print(m)
			if type(m) != nn.Sequential and i != 0:
				if isinstance(m, nn.Conv2d):
					total_param = m.weight.data.shape[0]
					weight_copy = m.weight.data.abs().clone().cpu().numpy()
					if args.score == 'abs_w':
						taylor = np.sum(weight_copy,  axis=(1,2,3))
					elif args.score == 'abs_grad':
						grad_copy = m.weight.grad.data.abs().clone().cpu().numpy()
						taylor = np.sum(grad_copy, axis=(1,2,3))
					elif args.score == 'grad_w':
						grad_copy = m.weight.grad.data.abs().clone().cpu().numpy()
						taylor = np.sum(weight_copy*grad_copy, axis=(1, 2, 3))

					num_keep = int(total_param * threshold)
					arg_max = np.argsort(taylor) # Returns the indices sort an array. small->big
					arg_max_rev = arg_max[::-1][:num_keep]
					thre = taylor[arg_max_rev[-1]]
					mask = np.zeros(weight_copy.shape)
					mask_R = np.ones(weight_copy.shape)
					mask[arg_max_rev.tolist(), :, :, :] = 1.0 ## mask = 0 means postions to be updated
					mask_R[arg_max_rev.tolist(), :, :, :] = 0.0  ## mask = 0 means postions to be updated

					mask_list_4d.append(mask)  # 0 is more
					mask_list_R_4d.append(mask_R)  # 1 is more
					threshold_list.append(thre)
					if args.score in ['abs_grad', 'grad_w']:
						gradient_list.append(m.weight.grad.data.clone().cpu().numpy())
					weight_list.append(m.weight.data.clone().cpu().numpy())
					taylor_list.append(taylor)

				elif isinstance(m, nn.BatchNorm2d):
					# bn weight
					total_param = m.weight.data.shape[0]
					weight_copy = m.weight.data.abs().clone().cpu().numpy()
					if args.score == 'abs_w':
						taylor = weight_copy# * weight_copy
					elif args.score == 'abs_grad':
						grad_copy = m.weight.grad.data.abs().clone().cpu().numpy()
						taylor = grad_copy  # * weight_copy
					elif args.score == 'grad_w':
						grad_copy = m.weight.grad.data.abs().clone().cpu().numpy()
						taylor = weight_copy*grad_copy  #
					num_keep = int(total_param * threshold)
					arg_max = np.argsort(taylor)  # Returns the indices sort an array. small->big
					arg_max_rev = arg_max[::-1][:num_keep]
					thre = taylor[arg_max_rev[-1]]
					mask = np.zeros(weight_copy.shape)
					mask_R = np.ones(weight_copy.shape)
					mask[arg_max_rev.tolist()] = 1.0
					mask_R[arg_max_rev.tolist()] = 0.0
					mask_list_4d.append(mask)  # 0 is more
					mask_list_R_4d.append(mask_R)
					threshold_list.append(thre)
					if args.score in ['abs_grad', 'grad_w']:
						gradient_list.append(m.weight.grad.data.clone().cpu().numpy())
					weight_list.append(m.weight.data.clone().cpu().numpy())
					taylor_list.append(taylor)

					##bn bias
					total_param = m.bias.data.shape[0]
					weight_copy = m.bias.data.abs().clone().cpu().numpy()
					if args.score == 'abs_w':
						taylor = weight_copy# * weight_copy
					elif args.score == 'abs_grad':
						grad_copy = m.bias.grad.data.abs().clone().cpu().numpy()
						taylor = grad_copy  # * weight_copy
					elif args.score == 'grad_w':
						grad_copy = m.bias.grad.data.abs().clone().cpu().numpy()
						taylor = weight_copy*grad_copy  #
					num_keep = int(total_param * threshold)
					arg_max = np.argsort(taylor)  # Returns the indices sort an array. small->big
					arg_max_rev = arg_max[::-1][:num_keep]
					thre = taylor[arg_max_rev[-1]]
					mask = np.zeros(weight_copy.shape)
					mask_R = np.ones(weight_copy.shape)
					mask[arg_max_rev.tolist()] = 1.0
					mask_R[arg_max_rev.tolist()] = 0.0
					mask_list_4d.append(mask)
					mask_list_R_4d.append(mask_R)
					threshold_list.append(thre)
					if args.score in ['abs_grad', 'grad_w']:
						gradient_list.append(m.bias.grad.data.clone().cpu().numpy())
					weight_list.append(m.bias.data.clone().cpu().numpy())
					taylor_list.append(taylor)

					# # running_mean
					total_param = m.running_mean.data.shape[0]
					weight_copy = m.running_mean.data.abs().clone().cpu().numpy()
					num_keep = int(total_param * threshold)
					arg_max = np.argsort(taylor)  # Returns the indices sort an array. small->big
					arg_max_rev = arg_max[::-1][:num_keep]
					thre = taylor[arg_max_rev[-1]]
					mask = np.zeros(weight_copy.shape)
					mask_R = np.ones(weight_copy.shape)
					mask[arg_max_rev.tolist()] = 1.0
					mask_R[arg_max_rev.tolist()] = 0.0
					mask_list_4d.append(mask)
					mask_list_R_4d.append(mask_R)
					threshold_list.append(thre)
					if args.score in ['abs_grad', 'grad_w']:
						gradient_list.append(m.bias.grad.data.clone().cpu().numpy())
					weight_list.append(m.bias.data.clone().cpu().numpy())
					taylor_list.append(taylor)

					total_param = m.running_var.data.shape[0]
					weight_copy = m.running_var.data.abs().clone().cpu().numpy()
					taylor = weight_copy  # * weight_copy
					num_keep = int(total_param * threshold)
					arg_max = np.argsort(taylor)  # Returns the indices sort an array. small->big
					arg_max_rev = arg_max[::-1][:num_keep]
					thre = taylor[arg_max_rev[-1]]
					mask = np.zeros(weight_copy.shape)
					mask_R = np.ones(weight_copy.shape)
					mask[arg_max_rev.tolist()] = 1.0
					mask_R[arg_max_rev.tolist()] = 0.0
					mask_list_4d.append(mask)
					mask_list_R_4d.append(mask_R)
					threshold_list.append(thre)
					if args.score in ['abs_grad', 'grad_w']:
						gradient_list.append(m.bias.grad.data.clone().cpu().numpy())
					weight_list.append(m.bias.data.clone().cpu().numpy())
					taylor_list.append(taylor)

					# if torch.__version__ == '1.0.1.post2': # torch 1.0 bn.num_tracked
					mask_list_4d.append(np.zeros(1))
					mask_list_R_4d.append(np.zeros(1))
					threshold_list.append(np.zeros(1))
					gradient_list.append(np.zeros(1))
					weight_list.append(np.zeros(1))
					taylor_list.append(taylor)

				elif isinstance(m, nn.Linear): # neuron-wise
					# print('linear', m)
					#linear weight
					weight_copy = m.weight.data.abs().clone().cpu().numpy()
					if args.score == 'abs_w':
						taylor = np.sum(weight_copy, axis = 1)
					elif args.score == 'abs_grad':
						grad_copy = m.weight.grad.data.abs().clone().cpu().numpy()
						taylor = np.sum(grad_copy, axis = 1)
					elif args.score == 'grad_w':
						grad_copy = m.weight.grad.data.abs().clone().cpu().numpy()
						taylor = np.sum(weight_copy*grad_copy, axis = 1)
					num_keep = int(m.weight.data.shape[0] * threshold * args.FC_decay)
					arg_max = np.argsort(taylor)  # Returns the indices that would sort an array. small->big
					arg_max_rev = arg_max[::-1][:num_keep]
					thre = taylor[arg_max_rev[-1]]
					mask = np.zeros(weight_copy.shape)
					mask_R = np.ones(weight_copy.shape)
					mask[arg_max_rev.tolist(), :] = 1.0
					mask_R[arg_max_rev.tolist(), :] = 0.0
					mask_list_4d.append(mask)  # 0 is more
					mask_list_R_4d.append(mask_R)  # 1 is more
					threshold_list.append(thre)
					if args.score in ['abs_grad', 'grad_w']:
						gradient_list.append(m.weight.grad.data.clone())
					weight_list.append(m.weight.data.clone())
					taylor_list.append(taylor)

					# linear bias
					weight_copy = m.bias.data.abs().clone().cpu().numpy()
					if args.score == 'abs_w':
						taylor = weight_copy# * weight_copy
					elif args.score == 'abs_grad':
						grad_copy = m.bias.grad.data.abs().clone().cpu().numpy()
						taylor = grad_copy  # * weight_copy
					elif args.score == 'grad_w':
						grad_copy = m.bias.grad.data.abs().clone().cpu().numpy()
						taylor = weight_copy*grad_copy  #
					arg_max = np.argsort(taylor)
					arg_max_rev = arg_max[::-1][:num_keep]
					thre = taylor[arg_max_rev[-1]]
					mask = np.zeros(weight_copy.shape[0])
					mask_R = np.ones(weight_copy.shape[0])
					mask[arg_max_rev.tolist()] = 1.0
					mask_R[arg_max_rev.tolist()] = 0.0
					mask_list_4d.append(mask)
					mask_list_R_4d.append(mask_R)
					threshold_list.append(thre)
					if args.score in ['abs_grad', 'grad_w']:
						gradient_list.append(m.bias.grad.data.clone())
					weight_list.append(m.bias.data.clone())
					taylor_list.append(taylor)
			i += 1
		all_mask = []
		all_mask.append(mask_list_4d)
		all_mask.append(mask_list_R_4d)
		logging.info('Got some lists: mask/maskR/threshold/gradient/weight/{}'.format(args.score))
		logging.info('mask length: {} // threshold_list length:{} // gradient list: length {} // weight list: length {} // taylor_list: length {}'.
					 format(len(mask_list_4d), len(threshold_list), len(gradient_list), len(weight_list), len(taylor_list)))  # 33

		gradient_dict, threshold_dict, mask_dict, mask_R_dict, taylor_dict = self.convert_list_to_dict(gradient_list, threshold_list, all_mask, taylor_list)
		return all_mask, threshold_dict, mask_dict, mask_R_dict, taylor_dict



	def convert_list_to_dict(self, gradient_list, threshold_list, mask_file, taylor_list): # test drift range of the rest parameters
		threshold_dict = OrderedDict([(k, None) for k in self.net.state_dict().keys()])
		gradient_dict = OrderedDict([(k, None) for k in self.net.state_dict().keys()])
		mask_dict = OrderedDict([(k, None) for k in self.net.state_dict().keys()])
		mask_R_dict = OrderedDict([(k, None) for k in self.net.state_dict().keys()])
		taylor_dict = OrderedDict([(k, None) for k in self.net.state_dict().keys()])
		# print(threshold_dict.keys())
		# print(len(threshold_dict))
		assert len(threshold_list) == len(threshold_dict), 'Dictionary <-> list does not match'

		idx = 0

		mask_list = []
		mask_list_R = []
		for i in range(len(mask_file[0])):
			mask_list.append(torch.from_numpy(mask_file[0][i]).type(torch.cuda.FloatTensor))
			mask_list_R.append(torch.from_numpy(mask_file[1][i]).type(torch.cuda.FloatTensor))

		for layer_name, param in self.net.state_dict().items():
			# print(layer_name, param.shape)
			# print(idx)
			# print(threshold_list[idx])
			threshold_dict[layer_name] = threshold_list[idx]
			if args.score in ['abs_grad', 'grad_w']:
				gradient_dict[layer_name] = gradient_list[idx]
			mask_dict[layer_name] = mask_list[idx]
			mask_R_dict[layer_name] = mask_list_R[idx]
			taylor_dict[layer_name] = taylor_list[idx]
			idx += 1
		# for i, key in enumerate(mask_dict): # check if threshold loading into correct dictionary
		#     assert threshold_list[i] == threshold_dict[key], 'Threshold loading incorrect'
		#     assert taylor_list[i].all() == taylor_dict[key].all(), 'Taylor loading incorrect'
		#     assert gradient_list[i].all() == gradient_dict[key].all(), 'Gradient loading incorrect'
		logging.info('Several lists are converted into dictionaries (in torch.cuda)\n\n')
		return  gradient_dict, threshold_dict, mask_dict, mask_R_dict, taylor_dict



	def mask_frozen_weight(self, maskR):

		param_processed = OrderedDict([(k, None) for k in self.net.state_dict().keys()])

		for layer_name, param in self.net.state_dict().items():
				param_processed[layer_name] = Variable(torch.mul(param, maskR[layer_name]), requires_grad=False)
		self.net.load_state_dict(param_processed)


	def AND_twomasks(self, mask_dict_1, mask_dict_2, maskR_dict_1, maskR_dict_2):
		maskR_processed = OrderedDict([(k, None) for k in maskR_dict_1.keys()])
		mask_processed = OrderedDict([(k, None) for k in maskR_dict_1.keys()])
		for layer_name, mask in maskR_dict_1.items():
			maskR_processed[layer_name] = torch.mul(maskR_dict_1[layer_name], maskR_dict_2[layer_name])
			mask_processed[layer_name] =  torch.add(mask_dict_1[layer_name], mask_dict_2[layer_name])
		return mask_processed, maskR_processed




	def compute_features(self, memory_loader, task_id):
		self.net.eval()

		num_classes = (task_id + 1) * args.classes_per_task
		logging.info('num_classes %s', num_classes)

		lists = [[] for _ in range(num_classes)]
		labels = [[] for _ in range(num_classes)]
		logging.info('Calculating feature center......\n')

		for batch_idx, (inputs, targets) in enumerate(memory_loader):
			logging.info('Extracting features...batch %s / label: %s', batch_idx, np.unique(targets.cpu()))
			logging.info('[batch, image_channel, image_w, image_h] shape %s \n', inputs.shape)

			inputs, targets = inputs.to(self.device), targets.to(self.device)

			inputs = Variable(inputs)
			targets = Variable(targets)

			for i in range(targets.shape[0]):
				_, feature_vec = self.net(inputs[i].unsqueeze_(0))
				for label in range(num_classes):
					if targets[i] == label:
						lists[label].append(feature_vec.detach().cpu().numpy().squeeze())
						labels[label].append(label)

		logging.info('feature map shape %s', feature_vec.shape)
		logging.info('lists length: {0:d}, {1:d} samples per class\n'.format(len(lists), len(lists[0])))
		# print(labels[0][0].shape)
		norm_feature_all = np.zeros((num_classes, feature_vec.shape[1]))

		for j in range(len(lists)):
			vec = np.asarray(lists[j])
			vec = np.mean(vec, axis = 0)
			vec /= np.linalg.norm(vec)
			norm_feature_all[j, :] = vec
		logging.info('feature_all shape - [#classes, feature dimension] - %s', norm_feature_all.shape)
		return norm_feature_all, num_classes, lists, labels


	def test_NME(self, feature_vec, num_classes, testloader):
		self.net.eval()
		test_loss = 0
		correct = 0
		total = 0
		dist_score = []
		logging.info('Testing with Nearest-Mean-Exampler.....it takes some time.....')
		with torch.no_grad():

			for batch_idx, (inputs, targets) in enumerate(testloader):
				inputs, targets = inputs.to(self.device), targets.to(self.device)

				inputs = Variable(inputs)
				targets =Variable(targets)
				total += targets.size(0)
				for i in range(targets.shape[0]):

					_, feature_vec_test = self.net(inputs[i].unsqueeze_(0))

					feature_vec_test = feature_vec_test.cpu().numpy()

					feature_vec_test /= np.linalg.norm(feature_vec_test)

					for j in range(num_classes):
						val = np.expand_dims(feature_vec[j], axis=1)
						distance = cdist(feature_vec_test, val.T, 'sqeuclidean')
						dist_score.append(distance)

					score = np.asarray(dist_score).squeeze()
					predicted = np.argmin(score)
					if (predicted == targets[i]):
						correct += 1
					dist_score.clear()
		logging.info('>>>>>>>>>>>>>>>>>>>>>> NME testing accuracy is %s', correct/total)
		return correct/total
		# Usage:
		# logging.info("current task {} Test with NME ==========================================".format(task_id))
		#	## memory_for_NME = get_memory_cifar(0, alltask_list, num_images = alltask_memory, batch_size = args.batch_size)
		#	feature_vec, num_classes = method.compute_features(memory_for_NME, task_id)
		# NME_accu_mix.append(method.test_NME(feature_vec, num_classes, test_mix_full))
		# NME_accu_0.append(method.test_NME(feature_vec, num_classes, test_0))
		# NME_accu_current.append(method.test_NME(feature_vec, num_classes, test_current))

	def prune_secondary(self, mask_dict):
		param_processed = OrderedDict([(k, None) for k in self.net.state_dict().keys()])

		for layer_name, param in self.net.state_dict().items():
			param = param.type(torch.cuda.FloatTensor)
			param_processed[layer_name] = Variable(torch.mul(param, mask_dict[layer_name]),requires_grad=True)

		self.net.load_state_dict(param_processed)


	def PCA_feature(self, train_dataset, task_id, title):
		_, _, feature_all_list, labels = self.compute_features(train_dataset, task_id)
		total_image = len(feature_all_list[0]) * len(feature_all_list)
		feature_array = np.zeros((total_image, feature_all_list[0][0].shape[0]))
		labels_array = np.zeros(total_image)

		for i in range(feature_array.shape[0]):
			index = int(i // len(feature_all_list[0]))
			feature_array[i, :] = feature_all_list[index][i % len(feature_all_list[0])]
			labels_array[i] = labels[index][i % len(feature_all_list[0])]
		print(feature_array.shape)
		print(labels_array.shape)

		with open(title+'.pickle', 'wb') as f:
			pickle.dump(feature_array, f)

		pca = decomposition.PCA(n_components=10)
		X_training_reduced = pca.fit_transform(feature_array)
		tsne = TSNE(n_components=3)
		X_training_reduced_tsne = tsne.fit_transform(X_training_reduced)
		logging.info('Dimension decreases from %s to %s', feature_array.shape, X_training_reduced_tsne.shape)
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		mk = [',', 'o', 'v', '^', '<', '>', '1', '2', '3', 'p', 'P', '*', 'h', '+', 'x', 'D', 'd', '4', '8', 's']

		ax.scatter(X_training_reduced_tsne[:, 0], X_training_reduced_tsne[:, 1], X_training_reduced_tsne[:, 2],
				   c=labels_array, cmap='tab10', marker = mk[task_id])
		plt.savefig(title + '.png')


	def compute_distance(self, feature_vec, dataloader, task_id):
		self.net.eval()
		test_loss = 0
		correct = 0
		total = 0
		dist_score = []
		num_classes = args.classes_per_task
		start_idx = args.classes_per_task * task_id
		distance = [[] for _ in range(num_classes)]

		logging.info('feature_vec.shape %s', feature_vec.shape)  # 20,120
		logging.info('Calculating distance......')

		with torch.no_grad():

			for batch_idx, (inputs, targets) in enumerate(dataloader):
				inputs, targets = inputs.to(self.device), targets.to(self.device)

				inputs = Variable(inputs)
				targets = Variable(targets)
				total += targets.size(0)
				for i in range(targets.shape[0]):
					# print('targets.shape', targets.shape)   #128
					outputs, feature_vec_test = self.net(inputs[i].unsqueeze_(0))
					_, predicted = outputs.max(1)
					feature_vec_test = feature_vec_test.cpu().numpy()
					feature_vec_test /= np.linalg.norm(feature_vec_test)

					for label in range(num_classes):
						# print(targets[i], start_idx)
						if targets[i] == start_idx + label:
							val = np.expand_dims(feature_vec[label+start_idx], axis=1)  #120, 1, val.T: 1,120
							dis = cdist(feature_vec_test, val.T, 'sqeuclidean')
							distance[label].append(dis)
							# print('batch_idx {} image {} target{}/predicted{} distance {}'.format(batch_idx,i, targets[i], predicted.item(), dis))

			total_image = 0
			distance_list = []
			for i in range(len(distance)):
				# print(i, len(distance[i]))
				distance_list += distance[i]
				total_image += len(distance[i])
			distance_array0 = np.zeros(total_image)
			for i in range(distance_array0.shape[0]):
				distance_array0[i] = distance_list[i]

			assert distance_array0[-1] == distance[-1][-1], 'Error when converting list to array, distance'
			with open('../results/distance_Task{}.pickle'.format(task_id), 'wb') as f:
				pickle.dump((distance_array0), f)

			return distance_array0


	def initial_memory(self, size = args.total_memory_size):
		empty_memory_list = []
		memory_target_list = []
		for _ in range(self.total_num_task):
			empty_memory_list.append(torch.zeros(size, 3, 32, 32))
			memory_target_list.append(torch.zeros(size))

		logging.info('Empty memory list created: Total container: {}. Size of each container ({},3,32,32)'.format(len(empty_memory_list), size))

		return empty_memory_list, memory_target_list


	def select_memory_for_currentTask(self, data_loader, memory_img_list, memory_target_list, task_id):

		feature_center, _, _, _ = self.compute_features(data_loader, task_id)
		distance_array = self.compute_distance(feature_center, data_loader, task_id)
		# print(feature_center[10])

		logging.info('Min and Max distance: {0:.4f} {1:.4f}'.format(min(distance_array), max(distance_array)))

		sort_distance_idx = np.argsort(distance_array.squeeze())
		selected_index = sort_distance_idx[::-1][0:args.total_memory_size]
		assert distance_array[sort_distance_idx[-1]] == distance_array[selected_index[0]], 'check distance sorting'
		assert distance_array[sort_distance_idx[-args.total_memory_size]] == distance_array[selected_index[-1]], 'check distance sorting'

		j = 0
		for batch_idx, (inputs, targets) in enumerate(data_loader):
			for i in selected_index:
				memory_img_list[task_id][j, :, :, :] = inputs[i, :, :, :]
				memory_target_list[task_id][j] = targets[i]
				j += 1
		logging.info('Memory selection for current task -  done, according to furthest feature distance\n')

		return memory_img_list, memory_target_list


	def select_balanced_memory(self, memory_img_list, memory_target_list, task_id):
		each_size = int(args.total_memory_size / task_id) if task_id != 0 else 0

		memory_img_array = torch.zeros(args.total_memory_size, 3, 32, 32)
		memory_target_array = torch.zeros(args.total_memory_size)
		for i in range(task_id):
			start = each_size * i
			end = each_size * (i+1)
			# print(start, end)
			memory_img_array[start : end, :, :, :] = memory_img_list[i][0:each_size, :, :, :]
			memory_target_array[start : end] = memory_target_list[i][0:each_size]
		logging.info('\nMemory selection from previous tasks - done, shape: %s', memory_img_array.shape)
		return memory_img_array, memory_target_array


	def memory_to_dataloader(self, memoryset):
		memory_dataloader = DataLoader(memoryset, batch_size=args.batch_size, shuffle=True, drop_last=False,
									   num_workers=1)
		logging.info('Memory wrapped to dataloader \n')

		return memory_dataloader