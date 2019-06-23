# This algorithms is using only supervised data for the prediction.
# base line model

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import models
import datasets
from prettytable import PrettyTable
import datetime
import time
import pdb
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR


__all__ = ['just_supervised']   # Only this method will be visible outside the file

def train(device, model, criterion, epoch, train_loader, optimizer):
    model.train()
    training_loss = 0
    correct = 0
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        correct += utils.accuracy(output, target, topk=(1,))[0]
        loss.backward()
        optimizer.step()
        training_loss += loss.data.item()
    training_loss /= len(train_loader.dataset)
    return training_loss, correct.item(), len(train_loader.dataset)

def validation(device, model, criterion, val_loader):
    model.eval()
    validation_loss = 0
    correct1 = 0
    correct5 = 0
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(val_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            validation_loss += criterion(output, target).data.item() # sum up batch loss
            counts = utils.accuracy(output, target, topk=(1, 5))
            correct1 += counts[0]
            correct5 += counts[1]
        validation_loss /= len(val_loader.dataset)
    return validation_loss, correct1.item(), correct5.item(), len(val_loader.dataset)

def just_supervised(args, **kwargs):
    torch.manual_seed(args.seed)
    device = kwargs['device']
    file = kwargs['file']
    current_time = kwargs['current_time']
    nclasses = datasets.__dict__[args.dataset].nclasses
    model = models.__dict__[args.arch](nclasses = nclasses)
    model = torch.nn.DataParallel(model).to(device)
    if(args.load_saved_model == 'y'):
        saved_model_dict = torch.load(args.saved_model_filepath)
        model.load_state_dict(saved_model_dict, strict=False)

    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = utils.select_optimizer(args, model)
    lr_lambda = lambda epoch: 0.02 ** (epoch//60)
    scheduler = LambdaLR(optimizer, lr_lambda)
    train_loader, _, val_loader = utils.make_loader(args)
    report = PrettyTable(['Epoch #', 'Train loss', 'Train Accuracy', 'Train Correct', 'Train Total', 'Val loss', 'Top-1 Accuracy', 'Top-5 Accuracy', 'Top-1 Correct', 'Top-5 Correct', 'Val Total', 'Time(secs)'])
    for epoch in range(1, args.epochs + 1):
        per_epoch = PrettyTable(['Epoch #', 'Train loss', 'Train Accuracy', 'Train Correct', 'Train Total', 'Val loss', 'Top-1 Accuracy', 'Top-5 Accuracy', 'Top-1 Correct', 'Top-5 Correct', 'Val Total', 'Time(secs)'])
        start_time = time.time()
        training_loss, train_correct, train_total = train(device, model, criterion, epoch, train_loader, optimizer)
        validation_loss, val1_correct, val5_correct, val_total = validation(device, model, criterion, val_loader)
        end_time = time.time()
        report.add_row([epoch, round(training_loss, 4), "{:.3f}%".format(round((train_correct*100.0)/train_total, 3)), train_correct, train_total, round(validation_loss, 4), "{:.3f}%".format(round((val1_correct*100.0)/val_total, 3)), "{:.3f}%".format(round((val5_correct*100.0)/val_total, 3)), val1_correct, val5_correct, val_total, round(end_time - start_time, 2)])
        per_epoch.add_row([epoch, round(training_loss, 4), "{:.3f}%".format(round((train_correct*100.0)/train_total, 3)), train_correct, train_total, round(validation_loss, 4), "{:.3f}%".format(round((val1_correct*100.0)/val_total, 3)), "{:.3f}%".format(round((val5_correct*100.0)/val_total, 3)), val1_correct, val5_correct, val_total, round(end_time - start_time, 2)])
        print(per_epoch)
        if args.save_model == 'y':
            val_folder = "saved_model/" + current_time
            if not os.path.isdir(val_folder):
                os.mkdir(val_folder)
            save_model_file = val_folder + '/model_' + str(epoch) +'.pth'
            torch.save(model.state_dict(), save_model_file)
        if(args.arch == 'rotnet'):
            scheduler.step()
        # print('\nSaved model to ' + model_file + '. You can run `python evaluate.py --model' + model_file + '` to generate the Kaggle formatted csv file')
    file.write(report.get_string())