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
import datetime
import os
import time
import pdb

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

criterion = nn.NLLLoss().to(device)

current_time = str(datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
file = open("runs/run-" + current_time, "w")

def make_loader(args):
    data_transforms = data_transformations.__dict__[args.data_transforms]
    train_dataset = datasets.__dict__[args.dataset](is_train = True, data_transforms = data_transforms)
    val_dataset = datasets.__dict__[args.dataset](is_train = False, data_transforms = data_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    return train_loader, val_loader

def train(model, epoch, train_loader, optimizer):
    model.train()
    training_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data.to(device)), Variable(target.to(device))
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        training_loss += loss.data.item()
        if batch_idx == 10:
            break
    training_loss /= len(train_loader.dataset)
    return training_loss

def validation(model, val_loader):
    model.eval()
    validation_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(val_loader):
        data, target = Variable(data.to(device), volatile=True), Variable(target.to(device))
        output = model(data)
        validation_loss += criterion(output, target).data.item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        if batch_idx == 10:
            break

    validation_loss /= len(val_loader.dataset)
    return validation_loss, correct.item(), len(val_loader.dataset)


def main(args):
    torch.manual_seed(args.seed)
    nclasses = datasets.__dict__[args.dataset].nclasses
    model = models.__dict__[args.arch](nclasses = nclasses)
    # model = torch.nn.DataParallel(model).to(device)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    train_loader, val_loader = make_loader(args)
    report = PrettyTable(['Epoch No #', 'Training loss', 'Validation loss', 'Accuracy', 'Correct', 'Total', 'Time in secs'])
    for epoch in range(1, args.epochs + 1):
        print("processing epoch {} ...".format(epoch))
        start_time = time.time()
        training_loss = train(model, epoch, train_loader, optimizer)
        validation_loss, correct, total = validation(model, val_loader)
        end_time = time.time()
        report.add_row([epoch, round(training_loss, 4), round(validation_loss, 4), "{}%".format(round(correct/total, 3)), correct, total, round(end_time - start_time, 2)])
        if args.save_model == 'y':
            val_folder = "saved_model/" + current_time
            if not os.path.isdir(val_folder):
                os.mkdir(val_folder)
            save_model_file = val_folder + '/model_' + str(epoch) +'.pth'
            torch.save(model.state_dict(), save_model_file)
        # print('\nSaved model to ' + model_file + '. You can run `python evaluate.py --model' + model_file + '` to generate the Kaggle formatted csv file')
    file.write(report.get_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=3, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save_model', type=str, default='n', metavar='D',
                        help="Do you want to save models for this run or not. (y) for saving the model")

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
                        help='Datasets: ' +
                            ' | '.join(transformations_names) +
                            ' (default: tensor_transform)')
    # Printing Information
    args = parser.parse_args()
    
    options = PrettyTable(['option', 'Value'])
    for key, val in vars(args).items():
        options.add_row([key, val])
    options.add_row(["save-model-folder", current_time])
    file.write(options.get_string())
    file.write("\n")

    # creating folders
    if not os.path.isdir("runs"):
        os.mkdir("runs")

    if not os.path.isdir("saved_model"):
        os.mkdir("saved_model")

    if not os.path.isdir("data"):
        os.mkdir("data")

    main(parser.parse_args())
    file.write("\n")
    file.close()