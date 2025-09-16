# -*- coding: utf-8 -*-
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import math
import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm
from utils.cutout import Cutout
import models.proxy_ann_model as model
from utils.autoaugment import CIFAR10Policy

nonspike = math.inf

class Snnsystem(object):
    def __init__(self, arguments):
        super(Snnsystem, self).__init__()

        self.numclass = arguments['num_class']
        self.model_name = arguments['model_name']
        self.batch_size = arguments['bz']
        self.lr = arguments['lr']
        self.device = arguments['device']
        self.itNum = arguments['iterationNum']
        self.trainrefreshrate = arguments['barfreshnum']
        self.maxA = 0
        self.correct = 0
        self.total = 0

        # cal the sparsity
        self.sparsity = []

        if arguments['dataset'] == 'cifar10':
            self.dataName = 'cifar10'
            transform_train = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(32,padding=4,fill=128),
                                                              torchvision.transforms.RandomHorizontalFlip(),
                                                              CIFAR10Policy(),
                                                              torchvision.transforms.ToTensor(),
                                                              Cutout(n_holes=1,length=16),
                                                              torchvision.transforms.Normalize(
                                                                    (0.4914,    0.4822,   0.4465),
                                                                    (0.2023,    0.1994,   0.2010)),
                                                                 ])
            train_dataset = torchvision.datasets.CIFAR10(root='../datasets/cifar10/', train=True,
                                                         download=True,
                                                         transform=transform_train)
            self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,
                                                            num_workers=10,drop_last=True)
            test_dataset = torchvision.datasets.CIFAR10(root='../datasets/cifar10/', train=False,
                                                        download=True,
                                                        transform=torchvision.transforms.Compose([
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize(
                                                                    (0.4914,    0.4822,    0.4465),
                                                                    (0.2023,    0.1994,    0.2010))
                                                        ]))
            self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size,
                                                           num_workers=10,  drop_last=True)
                                                           
        if arguments['dataset'] == 'cifar100':
            self.dataName = 'cifar100'
            transform_train = torchvision.transforms.Compose([ torchvision.transforms.RandomCrop(32,padding=4,fill=128),
                                                               torchvision.transforms.RandomHorizontalFlip(),
                                                               CIFAR10Policy(),
                                                               torchvision.transforms.ToTensor(),
                                                               Cutout(n_holes=1,length=16),
                                                               torchvision.transforms.Normalize(
                                                                # (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                                                                   np.array([125.3, 123.0, 113.9]) / 255.0,
                                                                   np.array([63.0, 62.1, 66.7]) / 255.0)
                                                               ])
            train_dataset = torchvision.datasets.CIFAR100(root='/data/dataset/CIFAR100/', train=True, download=True,
                                                          transform=transform_train)
            self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                                            num_workers=10,drop_last=True)
            test_set = torchvision.datasets.CIFAR100(root='/data/dataset/CIFAR100/', train=False, download=True,
                                                     transform=torchvision.transforms.Compose([
                                                         torchvision.transforms.ToTensor(),
                                                         torchvision.transforms.Normalize(
                                                                # (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                                                             np.array([125.3, 123.0, 113.9]) / 255.0,
                                                             np.array([63.0, 62.1, 66.7]) / 255.0)
                                                     ]))
            self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size, shuffle=False,
                                                           num_workers=10,drop_last=True)


        self.model = model.SpikeVGG('VGG16').to(self.device)
        print(self.model)

        if 'pre-trained' in arguments.keys():
            path = arguments['pre-trained']
            pretrained_dict = torch.load(path,map_location='cpu')
            model_dict = self.model.state_dict()
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
            print('load success')

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        # self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[120,160,240])
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 150, 240], gamma=0.4)

        self.pbar = tqdm()

    def updatecorrect(self, prediction, labels):
        self.correct += float(prediction.eq(labels).sum().item())
        self.total += float(labels.size(0))

    def refresh(self, loss, train_accumu, epoch):
        self.pbar.desc = '%d th epoch, loss: %f' % (epoch, loss / self.total)
        self.pbar.update(train_accumu)
        self.pbar.refresh()

    def trainupdate(self):
        print('Training accuracy is %.4f'% (self.correct / self.total))
        self.resetcorrect()

    def testupdate(self):
        acc = self.correct / self.total
        self.maxA = max(self.maxA, acc)
        print('Testing accuracy is %.4f' % (acc))
        print('maxAcc is %.4f' % (self.maxA))

        if acc == self.maxA:
            torch.save(self.model.state_dict(),'./'+self.dataName+'best.pkl')
            print('save the best model params')

        self.resetcorrect()

    def resetcorrect(self):
        self.correct = 0
        self.total = 0

    def trainEpoch(self, epoch):

        c_loss = 0

        self.model.train()
        self.resetcorrect()

        lastfreshTotal = 0.
        iterable = range(len(self.train_loader))
        self.pbar.reset(total=len(iterable))
        for i, (images, labels) in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            images, labels = images.to(self.device), labels.to(self.device)

            outputs = self.model(images)
            _, predicted = outputs.cpu().max(1)
            self.updatecorrect(predicted.to(self.device), labels)

            self.loss = self.criterion(outputs, labels)
            self.loss.backward()
            self.optimizer.step()

            c_loss += (self.loss.detach()) * images.shape[0]

            train_accumu = self.total - lastfreshTotal
            if train_accumu > self.trainrefreshrate:
                self.refresh(c_loss, train_accumu, epoch)
                lastfreshTotal = self.total

        self.trainupdate()

    def testEpoch(self):
        self.model.eval()
        self.resetcorrect()
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.test_loader):
                # print('spikes:',self.model.spikes)
                # print('neurons:',self.model.neurons)
                self.model.spikes = 0.
                self.model.neurons = 0.
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                self.sparsity.append(self.model.spikes/self.model.neurons)
                _, predicted = outputs.cpu().max(1)
                self.updatecorrect(predicted.to(self.device), labels)

        self.testupdate()
        # print(sum(self.sparsity)/len(self.sparsity))

    def training(self):
        for epoch in range(self.itNum):
            print('current lr {:.5e}'.format(self.optimizer.param_groups[0]['lr']))
            self.trainEpoch(epoch=epoch)
            self.testEpoch()
            self.lr_scheduler.step()


Arguments = {'bz': 256, 'lr': 1e-4, 'device': torch.cuda.current_device(),
             'iterationNum': 300, 'barfreshnum': 256, 'num_class':10,
             'dataset': 'cifar10', 'model_name': 'VGG16', 'pre-trained': './pretrained/cifar10-ann-94.79VGG16.pkl'}


system = Snnsystem(Arguments)
system.training() # neurons: 284682(VGG16); 152576(VGG11)