from __future__ import print_function
import main
import argparse
from prettytable import PrettyTable
import matplotlib 
import pdb
import os
import datetime

# # creating folders
# if not os.path.isdir("runs"):
#     os.mkdir("runs")

# if not os.path.isdir("saved_model"):
#     os.mkdir("saved_model")

# if not os.path.isdir("data"):
#     os.mkdir("data")

# if not os.path.isdir("images"):
#     os.mkdir("images")

# # sanity check for some arguments
# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))

# dataset_names = sorted(name for name in datasets.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(datasets.__dict__[name]))

# transformations_names = sorted(name for name in data_transformations.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(data_transformations.__dict__[name]))

hyper_param_names = ['LR', 'BS', 'EPOCH', 'OPTIMIZER']

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='PyTorch semi supervised hyperparamters tuning')
parser.add_argument('--hyper_param', metavar='hp', default='LR', choices=hyper_param_names, help='hyperparamters:' +
						' | '.join(hyper_param_names) +
                        ' (default: LR)')

args = parser.parse_args()
class param:
	def __init__(self):
		self.batch_size = 64
		self.epochs = 200
		self.lr = 0.001
		self.momentum = 0.5
		self.seed = 1
		self.save_model = 'y'
		self.model_save_interval = 50
		self.label = 'new'
		self.arch = 'vae_net'
		self.dataset = 'cifar10'
		self.data_transforms = 'tensor_transform'
		self.hyper_param = 'LR'

current_time = str(datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
file = open("runs/run-" + current_time, "w")

if __name__ == '__main__':

	if args.hyper_param == 'LR':
		report = PrettyTable(['LR', 'Best Val loss'])
		lrs = [10.0**j for j in range(-6,1,1)]
		for lr in lrs:
			best_val_loss = main.main(param())
			report.add_row([lr, best_val_loss])
	# else if args.hyper_param == 'BS':
