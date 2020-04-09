import os
import shutil
import scipy.io
from args import *
args = parser.parse_args()


if os.path.exists('../../results'):
	shutil.rmtree('../../results')
os.mkdir('../../results')

i = 0
for model in ['vgg16', 'resnet20','resnet20_noshort', 'resnet56','resnet56_noshort', 'densenet121']:
    epoch = {'resnet20':60, 'resnet20_noshort':120, 'densenet121': 125, 'vgg16': 180, 'resnet56': 80, 'resnet56_noshort': 150}
    if not os.path.exists('../../results/{}'.format(model)):
        os.mkdir('../../results/{}'.format(model))

    for alpha in [0.5, 1]:
    # for alpha in [ 1]:
        NA_C0 = 64 if model == 'vgg16' else 32

        command_tmp = 'python add_noise.py --gpu 1  --batch_size 64 --epoch ' + str(epoch[model]) + ' --alpha ' + str(alpha)+ ' --model ' + str(model) + ' --NA_C0 ' +str(NA_C0)
        print('command:\n', command_tmp)

        os.system(command_tmp)
        i = i + 1
        scipy.io.savemat('../../results/{}/tuning_addNoise_{}_{}_cifar10.mat'.format(model, model, i), {'model': model, 'alpha': alpha, 'epoch':epoch, 'task':args.task_division})
