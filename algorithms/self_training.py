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
from torch.utils.data import ConcatDataset, Subset, DataLoader

__all__ = ['self_training']   # Only this method will be visible outside the file

def train(device, model, logsoftmax, nll, epoch, train_supervised_dataset, optimizer, batch_size):
	model.train()
	training_loss = 0
	correct = 0
	train_supervised_loader = DataLoader(train_supervised_dataset, batch_size = batch_size, shuffle=True, num_workers=0)
	for batch_idx, (data, target) in tqdm(enumerate(train_supervised_loader)):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = model(data)
		loss = nll(logsoftmax(output), target)
		correct += utils.accuracy(output, target, topk=(1,))[0]
		loss.backward()
		optimizer.step()
		training_loss += loss.data.item()
	training_loss /= len(train_supervised_loader.dataset)
	return training_loss, correct.item(), len(train_supervised_loader.dataset)

def validation(device, model, logsoftmax, nll, val_loader):
	model.eval()
	validation_loss = 0
	correct1 = 0
	correct5 = 0
	with torch.no_grad():
		for batch_idx, (data, target) in tqdm(enumerate(val_loader)):
			data, target = data.to(device), target.to(device)
			output = model(data)
			validation_loss += nll(logsoftmax(output), target).data.item() # sum up batch loss
			counts = utils.accuracy(output, target, topk=(1, 5))
			correct1 += counts[0]
			correct5 += counts[1]
		validation_loss /= len(val_loader.dataset)
	return validation_loss, correct1.item(), correct5.item(), len(val_loader.dataset)

def label_addition(device, model, softmax, train_supervised_dataset, train_unsupervised_dataset, tau, batch_size):
	model.eval()
	with torch.no_grad():
		proxy_data, proxy_label = None, None
		train_unsupervised_loader = DataLoader(train_unsupervised_dataset, batch_size = batch_size, shuffle = False, num_workers=1)
		selected_index_global = []
		for batch_idx, (data, _) in tqdm(enumerate(train_unsupervised_loader)):
			data = data.to(device)
			output = model(data)
			probabilities = softmax(output)
			scores, indices = probabilities.topk(1, 1, True, True)
			selected_index_in_batch = (scores > tau).squeeze().nonzero().view(-1)
			new_data = torch.index_select(data, 0, selected_index_in_batch)
			new_data_label = torch.index_select(indices, 0, selected_index_in_batch).view(-1)
			if new_data.shape[0] > 0:
				# print("batch_idx = {} selected_index_in_batch = {}".format(batch_idx, selected_index_in_batch))
				selected_index_global += ((batch_idx * batch_size) + selected_index_in_batch).tolist()
				if proxy_data is None:
					proxy_data = new_data
					proxy_label = new_data_label
				else:
					proxy_data = torch.cat((proxy_data, new_data))
					proxy_label = torch.cat((proxy_label, new_data_label))
		if proxy_data is not None:
			proxy_data = torch.tensor(proxy_data, dtype = torch.float)
			proxy_label = proxy_label.tolist()
			proxy_dataset = datasets.ProxyDataset(proxy_data, proxy_label)
			indices = list(set(list(range(len(train_unsupervised_dataset)))) - set(selected_index_global))
			return ConcatDataset([train_supervised_dataset, proxy_dataset]), Subset(train_unsupervised_dataset, indices)
		else:
			return train_supervised_dataset, train_unsupervised_dataset


def self_training(args, **kwargs):
	torch.manual_seed(args.seed)
	device = kwargs['device']
	file = kwargs['file']
	current_time = kwargs['current_time']

	nclasses = datasets.__dict__[args.dataset].nclasses
	model = models.__dict__[args.arch](nclasses = nclasses)
	model = torch.nn.DataParallel(model).to(device)
	model.to(device)

	softmax = nn.Softmax(dim = 1)
	logsoftmax = nn.LogSoftmax(dim = 1)
	nll = nn.NLLLoss().to(device)

	optimizer = utils.select_optimizer(args, model)
	train_supervised_dataset, train_unsupervised_dataset, _ = utils.get_dataset(args)	# because we need to update the dataset after each epoch
	_, train_unsupervised_loader, val_loader = utils.make_loader(args)

	report = PrettyTable(['Epoch #', 'Train loss', 'Train Accuracy', 'Train Correct', 'Train Total', 'Val loss', 'Top-1 Accuracy', 'Top-5 Accuracy', 'Top-1 Correct', 'Top-5 Correct', 'Val Total', 'Time(secs)'])
	for epoch in range(1, args.epochs + 1):
		per_epoch = PrettyTable(['Epoch #', 'Train loss', 'Train Accuracy', 'Train Correct', 'Train Total', 'Val loss', 'Top-1 Accuracy', 'Top-5 Accuracy', 'Top-1 Correct', 'Top-5 Correct', 'Val Total', 'Time(secs)'])
		start_time = time.time()

		training_loss, train_correct, train_total = train(device, model, logsoftmax, nll, epoch, train_supervised_dataset, optimizer, args.batch_size)
		validation_loss, val1_correct, val5_correct, val_total = validation(device, model, logsoftmax, nll, val_loader)
		if epoch%args.proxy_interval == 0:
			train_supervised_dataset, train_unsupervised_dataset = label_addition(device, model, softmax, train_supervised_dataset, train_unsupervised_dataset, args.tau, args.batch_size)

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
	file.write(report.get_string())