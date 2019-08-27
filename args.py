"""
Author: Xiaocong Du
Date: April 2019 - May 2019
Project: Single-Net Continual Learning withProgressive Segmented Training (PST)
Description: parser work
"""

import argparse

parser = argparse.ArgumentParser(description='Single-Net Continual Learning (SNC) with Segmented Training, xiaocong du, April 2019')
parser.add_argument('--gpu', type=str, default = '1', help='GPU')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use for each rank, useful for data parallel evaluation')
parser.add_argument('--model', default='resnet18', help='model name')

parser.add_argument('--seed', type=int, default = 333, help='random seed')
parser.add_argument('--resume', type=bool, default = False, help='resume checkpoint')


parser.add_argument('--dataset', default = 'cifar10', type=str, choices=['cifar10','cifar100'])
parser.add_argument('--classes_per_task', type=int, default = 1, choices=[1], help='class per task')
# parser.add_argument('--num_classes', type=int, default = 12, help='final FC size')
parser.add_argument('--shuffle', type=bool, default = True, help='dataset shuffle')
parser.add_argument('--optimizer', default='sgd', help='optimizer: sgd | adam')

parser.add_argument('--epoch', type=int, default = 80, help='training epochs')
# parser.add_argument('--num_epoch_edge', type=int, default = 20, help='training epochs')
parser.add_argument('--save_epoch', type=int, default = 10, help='training epochs')

parser.add_argument('--NA_C0', type=int, default = 20, help='size of first channel in resnet')

parser.add_argument('--batch_size', type=int, default = 128, help='batch size')
parser.add_argument('--weight_decay', default = 5E-4, type=float, help='weight decay')

parser.add_argument('--lr', default = 0.1, type=float, help='learning rate')
parser.add_argument('--lr_step_size', default = 40, type=int, help='learning rate decay step')
parser.add_argument('--lr_gamma', default = 0.1, type=float, help='learning rate decay rate')

parser.add_argument('--score', type=str, default = 'grad_w', choices=['abs_w','abs_grad', 'grad_w'], help='importance score')

parser.add_argument('--threshold_task', default = 0.2, type=float, help='beta, importance percentage top X% ')

parser.add_argument('--FC_decay', default = 1.0, type=float, help='FC_decay * threshold is the mask rate in FC layer')
parser.add_argument('--total_memory_size', type=int, default = 1000, help='memory size')
parser.add_argument('--random_memory', type=bool, default = False, help='True: random memory, False: selective memory')


parser.add_argument('--task_division', type=list, default = [8, 2])


# parser.add_argument('--prune', type=bool, default = False, help='dataset shuffle')
