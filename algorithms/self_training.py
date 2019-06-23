# Self-Training is one of most basic algorithm for the proxy label.
# We use prediction done by model used as label for unsupervised data 
# and add them to our training dataset if the confidence of the prediction
# of the model is greater than a threshold(tau)

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
from torch.utils.data import ConcatDataset

__all__ = ['self_training']   # Only this method will be visible outside the file

num_channels = 3
img_height = 32
img_width = 32

def train(device, model, criterion, epoch, train_unsupervised_loader, optimizer, batch_size, arch):
	model.train()
	training_loss = 0
	correct = 0
	for batch_idx, (data, target) in tqdm(enumerate(train_unsupervised_loader)):
		data, target = data.to(device), target.to(device)
		if(arch == "rotnet"):
			data = data.view(-1, num_channels, img_height, img_width)
			target = target.view(-1)
		optimizer.zero_grad()
		output = model(data)
		loss = criterion(output, target)
		correct += utils.accuracy(output, target)[0]
		loss.backward()
		optimizer.step()
		training_loss += loss.data.item()
	training_loss /= len(train_unsupervised_loader.dataset)
	return training_loss, correct.item(), len(train_unsupervised_loader.dataset)

def validation(device, model, criterion, val_loader, arch):
	model.eval()
	validation_loss = 0
	correct = 0
	with torch.no_grad():
		for batch_idx, (data, target) in tqdm(enumerate(val_loader)):
			data, target = data.to(device), target.to(device)
			if(arch == "rotnet"):
				data = data.view(-1, num_channels, img_height, img_width)
				target = target.view(-1)
			output = model(data)
			validation_loss += criterion(output, target).data.item() # sum up batch loss
			counts = utils.accuracy(output, target)
			correct += counts[0]
		validation_loss /= len(val_loader.dataset)
	return validation_loss, correct.item(), len(val_loader.dataset)

def self_training(args, **kwargs):
	torch.manual_seed(args.seed)
	device = kwargs['device']
	file = kwargs['file']
	current_time = kwargs['current_time']

	nclasses = datasets.__dict__[args.dataset].nclasses
	model = models.__dict__[args.arch](nclasses = nclasses)
	model = torch.nn.DataParallel(model).to(device)
	model.to(device)
	print('device:', device)
	# Multiple loss will be needed because we need in-between probabilty.
	# nn.CrossEntropyLoss criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
	# loss = utils.select_loss_function(args.arch).to(device)

	optimizer = utils.select_optimizer(args, model)
	train_supervised_dataset, _, _ = utils.get_dataset(args)	# because we need to update the dataset after each epoch
	_, train_unsupervised_loader, val_loader = utils.make_loader(args)

	report = PrettyTable(['Epoch #', 'Train loss', 'Train Accuracy', 'Train Correct', 'Train Total', 'Val loss', 'Top-1 Accuracy', 'Top-1 Correct', 'Val Total', 'Time(secs)'])
	for epoch in range(1, args.epochs + 1):
		per_epoch = PrettyTable(['Epoch #', 'Train loss', 'Train Accuracy', 'Train Correct', 'Train Total', 'Val loss', 'Top-1 Accuracy', 'Top-1 Correct', 'Val Total', 'Time(secs)'])
		start_time = time.time()
		criterion = nn.CrossEntropyLoss()
		training_loss, train_correct, train_total = train(device, model, criterion, epoch, train_unsupervised_loader, optimizer, args.batch_size, args.arch)
		validation_loss, val_correct, val_total = validation(device, model, criterion, val_loader, args.arch)
		end_time = time.time()
		report.add_row([epoch, round(training_loss, 4), "{:.3f}%".format(round((train_correct*100.0)/train_total, 3)), train_correct, train_total, round(validation_loss, 4), "{:.3f}%".format(round((val_correct*100.0)/val_total, 3)), val_correct, val_total, round(end_time - start_time, 2)])
		per_epoch.add_row([epoch, round(training_loss, 4), "{:.3f}%".format(round((train_correct*100.0)/train_total, 3)), train_correct, train_total, round(validation_loss, 4), "{:.3f}%".format(round((val_correct*100.0)/val_total, 3)), val_correct, val_total, round(end_time - start_time, 2)])
		print(per_epoch)
		if args.save_model == 'y':
			val_folder = "saved_model/" + current_time
			if not os.path.isdir(val_folder):
				os.mkdir(val_folder)
			save_model_file = val_folder + '/model_' + str(epoch) +'.pth'
			torch.save(model.state_dict(), save_model_file)
	file.write(report.get_string())