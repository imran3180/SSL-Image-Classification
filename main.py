from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import models
import datasets
import data_transformations
from prettytable import PrettyTable
import matplotlib 
import datetime
import os
import time
import pdb
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# creating folders
if not os.path.isdir("runs"):
    os.mkdir("runs")

if not os.path.isdir("saved_model"):
    os.mkdir("saved_model")

if not os.path.isdir("data"):
    os.mkdir("data")

if not os.path.isdir("images"):
    os.mkdir("images")

hyper_param_names = ['LR', 'BS', 'EPOCH', 'OPTIMIZER']

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def loss_function(x_hat, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(
        x_hat, x.view(-1, 1024), reduction='sum'
    )
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

current_time = str(datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
file = open("runs/run-" + current_time, "w")

def plot_grid(fig, plot_input, actual_output, row_no):
    grid = ImageGrid(fig, 121, nrows_ncols=(row_no, 2), axes_pad=0.05, label_mode="1")
    actual_output = actual_output.view(-1, 32, 32)
    for i in range(row_no):
        for j in range(2):
            if(j == 0):
                grid[i*2+j].imshow(np.transpose(plot_input[i], (1, 2, 0)), interpolation="nearest")
            if(j == 1):
                grid[i*2+j].imshow(np.transpose(actual_output[i].detach().cpu().numpy(), (0, 1)), interpolation="nearest")

def make_loader(args):
    data_transforms = data_transformations.__dict__[args.data_transforms]
    train_dataset = datasets.__dict__[args.dataset](is_train = True, supervised = True, data_transforms = data_transforms)
    val_dataset = datasets.__dict__[args.dataset](is_train = False, data_transforms = data_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    return train_loader, val_loader

def train_unsupervised(model, epoch, train_loader, optimizer):
    model.train()
    training_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        # ===================forward=====================
        optimizer.zero_grad()
        output, mu, logvar  = model(data)
        loss = loss_function(output, data, mu, logvar)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
    training_loss /= len(train_loader.dataset)
    return training_loss, len(train_loader.dataset)

def validation(model, val_loader, epoch):
    with torch.no_grad():
        model.eval()
        validation_loss = 0
        for batch_idx, (data, _) in enumerate(val_loader):
            data = data.to(device)
            output, mu, logvar = model(data)
            validation_loss += loss_function(output, data, mu, logvar).item() # sum up batch loss
            if epoch % args.model_save_interval == 0 and batch_idx == 0:
                start_time = time.time()
                mini_batch_size = 5
                plot_input = data[0:5, :]
                plot_output = output[0:5, :]
                F = plt.figure(1, (30, 60))
                F.subplots_adjust(left=0.05, right=0.95)
                plot_grid(F, plot_input, plot_output, mini_batch_size)
                images_folder = "images/" + current_time
                if not os.path.isdir(images_folder):
                    os.mkdir(images_folder)
                plt.savefig(images_folder + '/' + args.label + "_" +  str(epoch) + ".jpg")
                plt.show()
                end_time = time.time()
                print('saving time:', round(end_time-start_time, 2))
    validation_loss /= len(val_loader.dataset)
    return validation_loss, len(val_loader.dataset)

def main(args):
    torch.manual_seed(args.seed)
    nclasses = datasets.__dict__[args.dataset].nclasses
    model = models.__dict__[args.arch](latent_size = nclasses)
    model = torch.nn.DataParallel(model).to(device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,)
    train_loader, val_loader = make_loader(args)
    report = PrettyTable(['Epoch #', 'Train loss', 'Train Total', 'Val loss', 'Val Total', 'Time(secs)'])
    best_validation_loss = 0
    for epoch in range(1, args.epochs + 1):
        # ===================pretty printing and logging=====================
        per_epoch = PrettyTable(['Epoch #', 'Train loss', 'Train Total', 'Val loss', 'Val Total', 'Time(secs)'])
        start_time = time.time()
        # ===================train and validate=====================
        training_loss, train_total = train_unsupervised(model, epoch, train_loader, optimizer)
        validation_loss,val_total = validation(model, val_loader, epoch)
        # ===================pretty printing and logging=====================
        end_time = time.time()
        report.add_row([epoch, round(training_loss, 4), train_total, round(validation_loss, 4), val_total, round(end_time - start_time, 2)])
        per_epoch.add_row([epoch, round(training_loss, 4), train_total, round(validation_loss, 4), val_total, round(end_time - start_time, 2)])
        print(per_epoch)
        # ===================saving model and printing images=====================
        if args.save_model == 'y':
            val_folder = "saved_model/" + current_time
            if not os.path.isdir(val_folder):
                os.mkdir(val_folder)
            if(epoch % args.model_save_interval == 0):
                save_model_file = val_folder + '/model_' + args.label + ':' + str(epoch) +'.pth'
                torch.save(model.state_dict(), save_model_file)
        # print('\nSaved model to ' + model_file + '. You can run `python evaluate.py --model' + model_file + '` to generate the Kaggle formatted csv file')
        if best_validation_loss==0 or (validation_loss < best_validation_loss):
            best_validation_loss = validation_loss
    file.write(report.get_string())
    return best_validation_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch semi supervised example')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save_model', type=str, default='y', metavar='D',
                        help="Do you want to save models for this run or not. (y) for saving the model")
    parser.add_argument('--model_save_interval', type=int, default=50, metavar='D',
                        help="No of epochs after which you want to save models for this run")
    parser.add_argument('--label', type=str, metavar='D', required=True,
                        help="label for this run")
    
    # Model structure
    parser.add_argument('--arch', '-a', metavar='ARCH', default='vae_net',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: conv_net)')
    # Dataset setting
    parser.add_argument('--dataset', metavar='DATASET', default='cifar10',
                        choices=dataset_names,
                        help='Datasets: ' +
                            ' | '.join(dataset_names) +
                            ' (default: ssl_data)')
    # Data Transformation setting
    parser.add_argument('--data_transforms', metavar='DATA_TRANFORMS', default='tensor_transform',
                        choices=transformations_names,
                        help='Datasets: ' +
                            ' | '.join(transformations_names) +
                            ' (default: tensor_transform)')
    parser.add_argument('--hyper_param', metavar='hp', default='LR', choices=hyper_param_names, help='hyperparamters:' +
                        ' | '.join(hyper_param_names) +
                        ' (default: LR)')
    # Printing Information
    args = parser.parse_args()

    options = PrettyTable(['option', 'Value'])
    for key, val in vars(args).items():
        options.add_row([key, val])
    options.add_row(["save-model-folder", current_time])
    file.write(options.get_string())
    file.write("\n")
    print(options)

    if args.hyper_param == 'LR':
        tuning_report = PrettyTable(['LR', 'Best Val loss'])
        lrs = [10.0**j for j in range(-6,1,1)]
        for lr in lrs:
            best_val_loss = main(parser.parse_args())
            tuning_report.add_row([lr, best_val_loss])
        file.write(tuning_report.get_string())
    # best_validation_loss = main(parser.parse_args())
    file.write("\n")
    file.close()