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
from learning_curve import *
from load_cifar import *
from utils_tool import count_parameters_in_MB
import matplotlib.pyplot as plt

from args import parser
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu



log_path = 'log_main.txt'.format()
log_format = '%(asctime)s   %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M%p')
fh = logging.FileHandler(os.path.join('./',log_path))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info("---------------------------------------------------------------------------------------------")
logging.info("                           Edge: main.py                                             ")
logging.info("---------------------------------------------------------------------------------------------")
logging.info("args = %s", args)

method = continualNN.ContinualNN()
method.initial_single_network(init_weights = True )
method.initialization(args.lr, args.lr_step_size, args.weight_decay)
