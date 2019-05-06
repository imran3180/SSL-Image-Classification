# -*- coding:utf-8 -*-
from __future__ import print_function 
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from functional import log_sum_exp
from torch.utils.data import DataLoader,TensorDataset
import torchvision.utils as vutils
import sys
import argparse
from Nets import Generator, Discriminator
from Datasets import *
import pdb
import tensorboardX
import os



class ImprovedGAN(object):

    def __init__(self, G, D, labeled, unlabeled, test, args):

        # load model from folder if exists, else create new modles
        if os.path.exists(args.savedir):
            print('Loading model from ' + args.savedir)
            self.G = torch.load(os.path.join(args.savedir, 'G.pkl'))
            self.D = torch.load(os.path.join(args.savedir, 'D.pkl'))
        else:
            os.makedirs(args.savedir)
            self.G = G
            self.D = D
            torch.save(self.G, os.path.join(args.savedir, 'G.pkl'))
            torch.save(self.D, os.path.join(args.savedir, 'D.pkl'))

        # set up logging
        self.writer = tensorboardX.SummaryWriter(log_dir=args.logdir)

        # enable cuda for models
        if args.cuda:
            self.G.cuda()
            self.D.cuda()

        # pass in training (S, U) and val data
        self.labeled = labeled
        self.unlabeled = unlabeled
        self.test = test

        # set up optimizers
        self.Doptim = optim.Adam(self.D.parameters(), lr=args.lr, betas= (args.momentum, 0.999))
        self.Goptim = optim.Adam(self.G.parameters(), lr=args.lr, betas = (args.momentum,0.999))

        # pass in args
        self.args = args

    def trainD(self, x_label, y, x_unlabel):

        x_label, x_unlabel, y = Variable(x_label), Variable(x_unlabel), Variable(y, requires_grad = False)
        if self.args.cuda:
            x_label, x_unlabel, y = x_label.cuda(), x_unlabel.cuda(), y.cuda()

        # discriminator output label based on S, U, and fake data
        output_label  = self.D(x_label, cuda=self.args.cuda)
        output_unlabel = self.D(x_unlabel, cuda=self.args.cuda)

        input_fake = self.G(x_unlabel.size()[0], cuda = self.args.cuda).view(x_unlabel.size()).detach()
        output_fake = self.D(input_fake, cuda=self.args.cuda)

        logz_label, logz_unlabel, logz_fake = log_sum_exp(output_label), log_sum_exp(output_unlabel), log_sum_exp(output_fake) # log ∑e^x_i
        prob_label = torch.gather(output_label, 1, y.unsqueeze(1)) # log e^x_label = x_label 

        # supervised loss
        loss_supervised = -torch.mean(prob_label) + torch.mean(logz_label)

        # unsupervised loss
        loss_unsupervised = 0.5 * (-torch.mean(logz_unlabel) + torch.mean(F.softplus(logz_unlabel))  + # real_data: log Z/(1+Z)
                            torch.mean(F.softplus(logz_fake)) ) # fake_data: log 1/(1+Z)

        # total loss
        loss = loss_supervised + self.args.unlabel_weight * loss_unsupervised

        # accuracy
        acc = torch.mean((output_label.max(1)[1] == y).float())

        # backprop and step
        self.Doptim.zero_grad()
        loss.backward()
        self.Doptim.step()

        return loss_supervised.item(), loss_unsupervised.item(), acc
    
    def trainG(self, x_unlabel):

        x_unlabel = Variable(x_unlabel)
        if self.args.cuda:
            x_unlabel = x_unlabel.cuda()

        fake = self.G(x_unlabel.size()[0], cuda = self.args.cuda).view(x_unlabel.size())
#        fake.retain_grad()

        # feature matching
        mom_gen, output_fake = self.D(fake, feature=True, cuda=self.args.cuda)
        mom_unlabel, _ = self.D(x_unlabel, feature=True, cuda=self.args.cuda)
        mom_gen = torch.mean(mom_gen, dim = 0)
        mom_unlabel = torch.mean(mom_unlabel, dim = 0)
        loss_fm = torch.mean((mom_gen - mom_unlabel) ** 2)
        #loss_adv = -torch.mean(F.softplus(log_sum_exp(output_fake)))
        loss = loss_fm #+ 1. * loss_adv        

        # backprop and step
        self.Goptim.zero_grad()
        self.Doptim.zero_grad()
        loss.backward()
        self.Goptim.step()

        return loss.item()

    def train(self):

        assert self.unlabeled.__len__() > self.labeled.__len__()
        assert type(self.labeled) == TensorDataset

        times = int(np.ceil(self.unlabeled.__len__() * 1. / self.labeled.__len__()))
        t1 = self.labeled.tensors[0].clone() # S data
        t2 = self.labeled.tensors[1].clone() # S labels
        tile_labeled = TensorDataset(t1.repeat(times,1,1,1),t2.repeat(times)) # equalize size of labelled and unlabelled
        gn = 0

        for epoch in range(self.args.epochs):

            self.G.train()
            self.D.train()

            # data loaders
            unlabel_loader1 = DataLoader(self.unlabeled, batch_size = self.args.batch_size, shuffle=True, drop_last=True, num_workers = 4)
            unlabel_loader2 = DataLoader(self.unlabeled, batch_size = self.args.batch_size, shuffle=True, drop_last=True, num_workers = 4).__iter__()
            label_loader = DataLoader(tile_labeled, batch_size = self.args.batch_size, shuffle=True, drop_last=True, num_workers = 4).__iter__()

            loss_supervised = loss_unsupervised = loss_gen = accuracy = 0.
            batch_num = 0

            for (unlabel1, _label1) in unlabel_loader1:
#                pdb.set_trace()
                batch_num += 1
                unlabel2, _label2 = unlabel_loader2.next()
                x, y = label_loader.next()
                if args.cuda:
                    x, y, unlabel1, unlabel2 = x.cuda(), y.cuda(), unlabel1.cuda(), unlabel2.cuda()

                ls, lu, acc = self.trainD(x, y, unlabel1)

                # self.dump_tensors()

                loss_supervised += ls
                loss_unsupervised += lu
                accuracy += acc

                lg = self.trainG(unlabel2)
                if epoch > 1 and lg > 1:
#                    pdb.set_trace()
                    lg = self.trainG(unlabel2)
                loss_gen += lg

                if (batch_num + 1) % self.args.log_interval == 0:
                    print('Training: %d / %d' % (batch_num + 1, len(unlabel_loader1)))
                    gn += 1
                    # self.writer.add_scalars('loss', {'loss_supervised':ls, 'loss_unsupervised':lu, 'loss_gen':lg}, gn)
                    # self.writer.add_histogram('real_feature', self.D(Variable(x, volatile = True), cuda=self.args.cuda, feature = True)[0], gn)
                    # self.writer.add_histogram('fake_feature', self.D(self.G(self.args.batch_size, cuda = self.args.cuda), cuda=self.args.cuda, feature = True)[0], gn)
                    # self.writer.add_histogram('fc3_bias', self.G.fc3.bias, gn)
                    # self.writer.add_histogram('D_feature_weight', self.D.layers[-1].weight, gn)
#                    self.writer.add_histogram('D_feature_bias', self.D.layers[-1].bias, gn)
                    print('Eval: correct %d/%d, %.4f' % (self.eval(), self.test.__len__(), acc))

                    # why???
                    self.D.train()
                    self.G.train()

                    torch.cuda.empty_cache()


            loss_supervised /= batch_num
            loss_unsupervised /= batch_num
            loss_gen /= batch_num
            accuracy /= batch_num
            print("Iteration %d, loss_supervised = %.4f, loss_unsupervised = %.4f, loss_gen = %.4f train acc = %.4f" % (epoch, loss_supervised, loss_unsupervised, loss_gen, accuracy))

            torch.cuda.empty_cache()


            sys.stdout.flush()

            if (epoch + 1) % self.args.eval_interval == 0:
                print("Eval: correct %d / %d"  % (self.eval(), self.test.__len__()))
                torch.save(self.G, os.path.join(args.savedir, 'G.pkl'))
                torch.save(self.D, os.path.join(args.savedir, 'D.pkl'))

    def predict(self, x):

        x = Variable(x)
        if self.args.cuda:
            x = x.cuda()

        return torch.max(self.D(x, cuda=self.args.cuda), 1)[1].data

    def eval(self):

        self.G.eval()
        self.D.eval()

        d, l = [], []
        for (datum, label) in self.test:
            d.append(datum)
            l.append(label)

        x, y = torch.stack(d), torch.LongTensor(l)
        if self.args.cuda:
            x, y = x.cuda(), y.cuda()
        pred = self.predict(x)
        return torch.sum(pred == y)

    def draw(self, batch_size):
        self.G.eval()
        return self.G(batch_size, cuda=self.args.cuda)

    def pretty_size(size):
        """Pretty prints a torch.Size object"""
        assert (isinstance(size, torch.Size))
        return " × ".join(map(str, size))

    def dump_tensors(gpu_only=True):
        """Prints a list of the Tensors being tracked by the garbage collector."""
        import gc
        total_size = 0
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    if not gpu_only or obj.is_cuda:
                        print("%s:%s%s %s" % (type(obj).__name__,
                                              " GPU" if obj.is_cuda else "",
                                              " pinned" if obj.is_pinned else "",
                                              pretty_size(obj.size())))
                        total_size += obj.numel()
                elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                    if not gpu_only or obj.is_cuda:
                        print("%s → %s:%s%s%s%s %s" % (type(obj).__name__,
                                                       type(obj.data).__name__,
                                                       " GPU" if obj.is_cuda else "",
                                                       " pinned" if obj.data.is_pinned else "",
                                                       " grad" if obj.requires_grad else "",
                                                       " volatile" if obj.volatile else "",
                                                       pretty_size(obj.data.size())))
                        total_size += obj.data.numel()
            except Exception as e:
                pass
        print("Total size:", total_size)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Improved GAN')
    parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                        help='learning rate (default: 0.003)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--eval-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before evaling training status')
    parser.add_argument('--unlabel-weight', type=float, default=1, metavar='N',
                        help='scale factor between labeled and unlabeled data')
    parser.add_argument('--logdir', type=str, default='./logfile', metavar='LOG_PATH', help='logfile path, tensorboard format')
    parser.add_argument('--savedir', type=str, default='./models', metavar='SAVE_PATH', help = 'saving path, pickle format')
    args = parser.parse_args()

    args.cuda = args.cuda and torch.cuda.is_available()
    np.random.seed(args.seed)

    device = torch.device("cuda:0" if args.cuda else "cpu")
    cudnn.benchmark = True

    gan = ImprovedGAN(Generator(100), Discriminator(), MnistLabel(10), MnistUnlabel(), MnistVal(), args)

    # gan = ImprovedGAN(Generator(100, output_dim = 64 * 64 * 3),
    #                   Discriminator(input_dim = 64 * 64 * 3, output_dim = 1000),
    #                   ImageNetLabel(1000, 2), ImageNetUnlabel(), ImageNetVal(),
    #                   args)

    # gan = ImprovedGAN(Generator(z_dim=100, nc=3).to(device),
    #                   Discriminator(nc = 3, output_units = 1000).to(device),
    #                   ImageNetLabel(1000, 2), ImageNetUnlabel(), ImageNetVal(),
    #                   args)

    gan.train()
    
