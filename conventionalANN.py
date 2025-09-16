# -*- coding: utf-8 -*-
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torchvision
import torchvision.datasets as datasets
import torch.nn as nn
from tqdm import tqdm
from utils.cutout import Cutout
from utils.autoaugment import CIFAR10Policy,ImageNetPolicy
from ptflops import get_model_complexity_info

cfg_fc = {
    'CNN': [512,800,128,10],
    'VGG11': [512, 4096, 4096,10],
    'VGG16': [512, 4096, 4096,10],
}

cfg_conv = {
    'CNN': [16, 'M', 32, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}

class activity(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x): # ReLU1
        big = 1.
        small = 0.

        x[x <= small] = small
        x[x >= big] = big

        valid_mask = x.lt(big) * x.gt(small)

        ctx.save_for_backward(valid_mask)
        return x

    @staticmethod
    def backward(ctx,grad_input,):
        valid_mask = ctx.saved_tensors[0]

        return grad_input * valid_mask.float()

class act(nn.Module):
    def __init__(self):
        super(act, self).__init__()

    def forward(self,input):
        return activity.apply(input)

class MLPblock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(MLPblock, self).__init__()

        self.linear = nn.Linear(in_features=input_channel,out_features=output_channel,bias=False)
        nn.init.kaiming_normal_(self.linear.weight, a=0, mode='fan_in', nonlinearity='relu')

    def forward(self, inputs):
        output = self.linear(inputs)
        return output

class CNNblock(nn.Module):
    def __init__(self,input_channel, output_channel, kernel_size, stride, bias=False, padding=0):
        super(CNNblock, self).__init__()

        self.cnn = nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, bias=bias,
                             padding=padding)
        nn.init.kaiming_normal_(self.cnn.weight, a=0, mode='fan_in', nonlinearity='relu')

    def forward(self, inputs):
        output = self.cnn(inputs)
        return output

class CNN(nn.Module):
    def __init__(self,vggname):
        super(CNN, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.activition = act()
        self.vggname = vggname

        self.num_class = cfg_fc[self.vggname][-1]

        self.fc, self.save_sign_fc = self.construct_fc()
        self.conv, self.save_sign_conv = self.construct_conv()
        self.dd = [[] for _ in range(11)]

    def construct_fc(self):
        layers = []
        save_tensor = []

        input_dim = cfg_fc[self.vggname][0]
        for x in cfg_fc[self.vggname][1:]:
            if x != self.num_class:
                layers += [MLPblock(input_dim, x),]
                save_tensor.append('pre_activation')
                input_dim = x
                layers += [self.activition]
                save_tensor.append('activation')
            else:
                layers += [ MLPblock(input_dim, x) ]
                save_tensor.append('last_preActi')

        return nn.Sequential(*layers), save_tensor


    def construct_conv(self):
        layers = []
        input_channel = 3
        save_tensor = []

        for x in cfg_conv[self.vggname]:
            if x == 'M':
                layers += [self.pool]
                save_tensor.append('pool')
            else:
                layers += [ CNNblock(input_channel, x, kernel_size=3, padding=1, bias=False, stride=1),
                            self.activition]
                save_tensor.append('pre_activation')
                save_tensor.append('activation')
                input_channel = x

        return nn.Sequential(*layers), save_tensor

    def forward(self,input):
        x = input
        for i in range(len(self.save_sign_conv)):
            x = self.conv[i](x)
        x = x.view(x.shape[0], -1)
        for i in range(len(self.save_sign_fc)):
            x = self.fc[i](x)

        return x


class Snnsystem(object):
    def __init__(self, arguments):
        super(Snnsystem, self).__init__()
        self.maxA = 0
        self.batch_size = arguments['bz']                   # 256
        self.lr = arguments['lr']                           # 0.001
        self.device = arguments['device']
        self.itNum = arguments['iterationNum']              # 200 epoch
        self.trainrefreshrate = arguments['barfreshnum']    # 256


        if arguments['dataset'] == 'mnist':
            self.dataName = 'mnist'
            transform_train = torchvision.transforms.Compose([
                                                             torchvision.transforms.ToTensor()])
            train_dataset = torchvision.datasets.MNIST(root='./data/mnist/', train=True, download=True,
                                                       transform=transform_train)
            self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                                            num_workers=0)
            test_set = torchvision.datasets.MNIST(root='./data/mnist/', train=False, download=True,
                                                  transform=torchvision.transforms.ToTensor())
            self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size, shuffle=False,
                                                           num_workers=0)

        if arguments['dataset'] == 'cifar10':
            self.dataName = 'cifar10'

            transform_train = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(32,padding=4,fill=128),
                                                              torchvision.transforms.RandomHorizontalFlip(),
                                                              CIFAR10Policy(),
                                                              torchvision.transforms.ToTensor(),
                                                              Cutout(n_holes=1,length=16),
                                                              torchvision.transforms.Normalize((0.4914, 0.4822,
                                                                                0.4465), (0.2023, 0.1994, 0.2010))
                                                              ])
            train_dataset = torchvision.datasets.CIFAR10(root='./data/cifar10/', train=True, download=True,
                                                         transform=transform_train)
            self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                                            num_workers=10,drop_last=True)
            test_set = torchvision.datasets.CIFAR10(root='./data/cifar10/', train=False, download=True,
                                                    transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.4914, 0.4822,
                                                                            0.4465), (0.2023, 0.1994, 0.2010))
                                                    ]))
            self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size, shuffle=False,
                                                           num_workers=10,drop_last=True)


        self.trainSize = len(self.train_loader)
        self.model = CNN('VGG16')
        # self.model = resnet20()

        if 'param' in arguments.keys():
            self.load_pretrain(arguments['param'])
            print(arguments['param'])

        self.model.to(self.device)
        print(self.model)
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr,momentum=0.9,weight_decay=5e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[15,30,40])
        self.pbar = tqdm()
        self.correct = 0
        self.total = 0
        self.trainacc = [0.]
        self.testacc = [0.]
        self.ratio = []
        self.weight_sum = {}

        self.nepoch = 0
        self.round = 1
        self.ratio_out_lt_one = 0
        self.div = 10000

    def load_pretrain(self, path):
        pretrained_dict = torch.load(path)

        snn_iter = iter(self.model.parameters())
        for k in pretrained_dict:
            if 'bias' not in k :
                with torch.no_grad():
                    next(snn_iter).copy_(self.model.state_dict()[k])

    def updatecorrect(self, prediction, labels):
        self.correct += float(prediction.eq(labels).sum().item())
        self.total += float(labels.size(0))

    def refresh(self, loss, closs, train_accumu, epoch):
        self.pbar.desc = '%d th epoch, the average tloss: %f, closs: %f' % (epoch, loss / self.total, closs / self.total)
        self.pbar.update(train_accumu)
        self.pbar.refresh()

    def trainupdate(self):
        self.trainacc.append(self.correct / self.total)
        print('Training accuracy is %.4f'% (self.trainacc[-1]))
        self.resetcorrect()


    def testupdate(self):
        acc = self.correct / self.total
        self.maxA = max(self.maxA, acc)
        print('Testing accuracy is %.4f' % (acc))
        print('maxAcc is %.4f' % (self.maxA))

        if acc == self.maxA:
            torch.save(self.model.state_dict(),'mnist-ann.pkl')
            print('save the best model params')

        self.testacc.append(acc)
        self.resetcorrect()

    def resetcorrect(self):
        self.correct = 0
        self.total = 0

    def trainEpoch(self, epoch):

        c_loss = 0
        self.model.train()
        self.resetcorrect()
        lastfreshTotal = 0.
        iterable = range(self.trainSize)
        self.pbar.reset(total=len(iterable))
        for i, (images, labels) in enumerate(self.train_loader):
            self.model.zero_grad()

            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)

            self.optimizer.zero_grad()
            self.loss = self.criterion(outputs, labels)

            _, predicted = outputs.cpu().max(1)
            self.updatecorrect(predicted, labels.cpu())

            self.loss.backward()
            self.optimizer.step()

            c_loss += self.loss.detach() * images.shape[0]
            running_loss = c_loss

            train_accumu = self.total - lastfreshTotal
            if train_accumu > self.trainrefreshrate:
                self.refresh(running_loss, c_loss, train_accumu, epoch)
                lastfreshTotal = self.total

        self.trainupdate()

    def testEpoch(self):
        self.model.eval()
        self.resetcorrect()
        self.ratio_out_lt_one = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.cpu().max(1)
                self.updatecorrect(predicted, targets)

        self.testupdate()

    def training(self):
        for epoch in range(self.itNum):
            print('current lr {:.5e}'.format(self.optimizer.param_groups[0]['lr']))
            self.trainEpoch(epoch=epoch)
            self.lr_scheduler.step()
            self.testEpoch()


Arguments = {'bz': 256, 'lr': 1e-2, 'device': torch.cuda.current_device(),
             'iterationNum': 200, 'barfreshnum': 256, 'dataset': 'cifar10',
             }

system = Snnsystem(Arguments)
system.training()