from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import models
import datasets
import algorithms
import data_transformations
from prettytable import PrettyTable
import datetime
import os
import time
import pdb


# creating folders
if not os.path.isdir("runs"):
    os.mkdir("runs")

if not os.path.isdir("saved_model"):
    os.mkdir("saved_model")

if not os.path.isdir("data"):
    os.mkdir("data")

# sanity check for some arguments
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

dataset_names = sorted(name for name in datasets.__dict__
    if name.islower() and not name.startswith("__")
    and callable(datasets.__dict__[name]))

transformations_names = sorted(name for name in data_transformations.__dict__
    if name.islower() and not name.startswith("__")
    and callable(data_transformations.__dict__[name]))

algorithms_names = sorted(name for name in algorithms.__dict__
    if name.islower() and not name.startswith("__")
    and callable(algorithms.__dict__[name]))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

current_time = str(datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
file = open("runs/run-" + current_time, "w")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
    parser.add_argument('--batch-size', type=int, default=32
        , metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--save_model', type=str, default='n', metavar='D',
                        help="Do you want to save models for this run or not. (y) for saving the model")

    parser.add_argument('--optim', type=str, default='SGD', metavar='D', choices = ['Adam', 'SGD'],
                        help="Select an optimizer: ['Adam', 'SGD']")

    # Model structure
    parser.add_argument('--arch', '-a', metavar='ARCH', default='conv_net',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: conv_net)')
    # Dataset setting
    parser.add_argument('--dataset', metavar='DATASET', default='ssl_data',
                        choices=dataset_names,
                        help='Datasets: ' +
                            ' | '.join(dataset_names) +
                            ' (default: ssl_data)')
    # Data Transformation setting
    parser.add_argument('--data_transforms', metavar='DATA_TRANFORMS', default='tensor_transform',
                        choices=transformations_names,
                        help='Transformations: ' +
                            ' | '.join(transformations_names) +
                            ' (default: tensor_transform)')

    # Algorithm & Related Fields
    parser.add_argument('--algorithm', metavar='ALGO', default='supervised',
                        choices=algorithms_names,
                        help='Algorithms: ' +
                            ' | '.join(algorithms_names) +
                            ' (default: supervised)')
    parser.add_argument('--load_saved_model', type=str, default='n', metavar='LOAD_SAVED_MODEL',
                        help='Do you wish to run the model pre-trained on unlabelled dataset or not. (y) for yes')

    parser.add_argument('--saved-model-filepath', type=str, default='', metavar='SAVED_MODEL_FOLDER',
                        help='Filepath of the saved model, to be given id load_saved_model is set to y')

    parser.add_argument('--tau', type=float, default=0.95, metavar='TAU', 
        help='threshold used by proxy label algorithm rate (default: 0.95)')

    # Printing Information
    args = parser.parse_args()
    
    options = PrettyTable(['option', 'Value'])
    for key, val in vars(args).items():
        options.add_row([key, val])
    options.add_row(["save-model-folder", current_time])
    file.write(options.get_string())
    file.write("\n")
    print(options)

    # Calling the specific algorithm
    algorithms.__dict__[args.algorithm](parser.parse_args(), device = device, file = file, current_time = current_time)

    file.write("\n")
    file.close()