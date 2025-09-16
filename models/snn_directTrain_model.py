# !usr/bin/env python
# -*- coding:utf-8 _*-
# @Time  :2022/12/26 15:20
# @Author: wwj
# @File  :snn_model.py
import torch
import torch.nn as nn
import math


cfg_fc = {
    'VGG11': [512, 4096, 4096,10],
    'VGG16': [512, 4096, 4096,100],
}

cfg_conv = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}

nonspike = math.inf

class piecewise_function_layer1(torch.autograd.Function):
    # The activation function of the coding (/first) layer
    @staticmethod
    def forward(ctx, x):
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
        dE = grad_input * valid_mask.float()
        return dE

class SNNactivation_layer1(nn.Module):
    def __init__(self):
        super(SNNactivation_layer1, self).__init__()

    def forward(self, input):
        return piecewise_function_layer1.apply(input)

class piecewise_function(torch.autograd.Function):
    # The activation function of other layers (other than the first layer)
    @staticmethod
    def forward(ctx, t_f, pre_acti, w_sum):

        t_f[t_f<0] = nonspike
        t_f[t_f>2] = nonspike
        t_f[w_sum >= 1 + pre_acti] = 1.0
        t_f = t_f - 1
        t_f[t_f<0] = nonspike

        valid_mask = t_f.lt(1) * t_f.gt(0)
        ctx.save_for_backward(valid_mask)
        return t_f

    @staticmethod
    def backward(ctx, grad_input):
        valid_mask = ctx.saved_tensors[0]
        dE = grad_input * valid_mask.float()
        return dE, None, None

class SNNactivation(nn.Module):
    def __init__(self):
        super(SNNactivation, self).__init__()

    def forward(self, t_f, pre_acti, w_sum):
        return piecewise_function.apply(t_f, pre_acti, w_sum)

class gradients_clip(torch.autograd.Function):
    # Gradient clipping to avoid gradient explosion
    @staticmethod
    def forward(ctx, valid_weight_sum, I):
        t_f = (2 + I)/(1 + valid_weight_sum)
        ctx.save_for_backward(valid_weight_sum, I)
        return t_f

    @staticmethod
    def backward(ctx, grad_input):
        bigger_clip = 1e5

        valid_weight_sum, I = ctx.saved_tensors

        grad_weight_sum = -(2 + I)/((1 + valid_weight_sum)**2)     # 激活值关于两项的导数
        grad_I = 1 / (1 + valid_weight_sum)

        grad_weight_sum[grad_weight_sum > bigger_clip] = bigger_clip
        grad_weight_sum[grad_weight_sum < -bigger_clip] = -bigger_clip

        grad_I[grad_I > bigger_clip] = bigger_clip
        grad_I[grad_I < -bigger_clip] = -bigger_clip

        return grad_input*grad_weight_sum, grad_input*grad_I

grad_clip = gradients_clip.apply

class Beginblock(nn.Module):
    # Coding layer
    def __init__(self, input_channel, output_channel, kernel_size, stride=1, padding=0, bias=False):
        super(Beginblock, self).__init__()
        self.cnn = nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, bias=bias,
                             padding=padding)

        nn.init.kaiming_normal_(self.cnn.weight, a=0, mode='fan_in', nonlinearity='relu')

    def forward(self, input):
        cnn_out = self.cnn(input)
        return cnn_out,0,0

class CNNblock(nn.Module):
    # Conv layer
    def __init__(self, input_channel, output_channel, kernel_size, stride=1, padding=0, bias=False):
        super(CNNblock, self).__init__()
        self.cnn = nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, bias=bias,
                             padding=padding)

        nn.init.kaiming_normal_(self.cnn.weight, a=0, mode='fan_in', nonlinearity='relu')
        self.stride = stride
        self.padding = padding


    def forward(self, t_f):
        valid_t_f = (t_f != nonspike).detach().float()
        valid_weight_sum = self.cnn(valid_t_f)

        t_f[t_f == nonspike] = 0.0
        cnn_out = self.cnn(t_f)
        outputs = grad_clip(valid_weight_sum,cnn_out)

        return outputs, cnn_out, valid_weight_sum

class MLPblock(nn.Module):
    # FC layer
    def __init__(self, input_channel, output_channel):
        super(MLPblock, self).__init__()
        self.linear = nn.Linear(input_channel, output_channel, bias=False)

        nn.init.kaiming_normal_(self.linear.weight.data, a=0, mode='fan_in', nonlinearity='relu')


    def forward(self, t_f):
        valid_t_f = (t_f != nonspike).detach().float()
        valid_weight_sum = self.linear(valid_t_f)

        t_f[t_f == nonspike] = 0.0
        mlp_out = self.linear(t_f)
        outputs = grad_clip(valid_weight_sum,mlp_out)

        return outputs, mlp_out, valid_weight_sum

class Finalblock(nn.Module):
    # Output layer
    def __init__(self, input_channel, output_channel):
        super(Finalblock, self).__init__()
        self.linear = nn.Linear(input_channel, output_channel, bias=False)

        nn.init.kaiming_normal_(self.linear.weight.data, a=0, mode='fan_in', nonlinearity='relu')

    def forward(self, t_f):
        valid_t_f = (t_f != nonspike).detach().float()
        valid_weight_sum = self.linear(valid_t_f)

        win_begin_in = 1 - t_f
        win_begin_in[win_begin_in == -nonspike] = 0.0
        win_begin_membrane = self.linear(win_begin_in)
        outputs = valid_weight_sum + win_begin_membrane

        return outputs

class SpikeVGG(nn.Module):
    def __init__(self, vggname, ):
        # Build model
        super(SpikeVGG, self).__init__()
        self.vggname = vggname

        self.num_class = cfg_fc[vggname][-1]
        self.pool = nn.MaxPool2d(2, 2)
        self.activation1 = SNNactivation_layer1()
        self.activation = SNNactivation()

        self.fc, self.save_sign_fc = self.construct_fc()
        self.conv, self.save_sign_conv = self.construct_conv()

        # cal sparsity
        self.spikes = 0.
        self.neurons = 0.

    def construct_fc(self):
        # Build the FC layers of the model
        layers = []
        save_tensor = []
        input_dim = cfg_fc[self.vggname][0]

        for x in cfg_fc[self.vggname][1:]:
            if x != self.num_class:
                # Build hidden layer
                layers += [MLPblock(input_dim, x)]
                save_tensor.append('pre_activation')
                input_dim = x
                layers += [self.activation]
                save_tensor.append('activation')
            else:
                # Build output layer
                layers += [
                    Finalblock(input_dim, x),
                           ]
                save_tensor.append('last_preActi')

        return nn.Sequential(*layers), save_tensor

    def construct_conv(self):
        # Build the Conv layers of the model
        layers = []
        input_channel = 3
        save_tensor = []
        begin_flag = 1

        for x in cfg_conv[self.vggname]:
            if x == 'M':
                layers += [self.pool]
                save_tensor.append('pool')
            else:
                if begin_flag==1:
                    begin_flag = 0
                    layers += [
                            # Build coding layer: use direct coding
                            Beginblock(input_channel, x, kernel_size=3, padding=1, stride=1, bias=False),
                            self.activation1
                        ]

                else:
                    layers += [
                        # Build hidden layer
                        CNNblock(input_channel, x, kernel_size=3, padding=1, stride=1, bias=False),
                        self.activation,
                            ]
                save_tensor.append('pre_activation')
                save_tensor.append('activation')
                input_channel = x

        return nn.Sequential(*layers), save_tensor

    def forward(self, inputs):

        # Forward of the Conv layer
        for i in range(len(self.save_sign_conv)):

            if self.save_sign_conv[i] == 'pre_activation':
                inputs, pre_act, w_sum = self.conv[i](inputs)
            elif self.save_sign_conv[i] == 'activation':
                if i == 1:
                    inputs = self.conv[i](inputs)
                    inputs = 1 - inputs
                    inputs[inputs==1] = nonspike
                else:
                    inputs = self.conv[i](inputs, pre_act, w_sum)

                self.neurons += inputs.numel()
                self.spikes += (inputs!=nonspike).float().sum()
            else:
                inputs = -self.conv[i](-inputs)
                pass

        # Flatten
        inputs = inputs.view(inputs.shape[0], -1)

        # Forward of the FC layer
        for i in range(len(self.save_sign_fc)):

            if self.save_sign_fc[i] == 'pre_activation':
                inputs, pre_act, w_sum = self.fc[i](inputs)
            elif self.save_sign_fc[i] == 'activation':
                inputs = self.fc[i](inputs, pre_act, w_sum)
                self.neurons += inputs.numel()
                self.spikes += (inputs!=nonspike).float().sum()
            elif self.save_sign_fc[i] == 'last_preActi':
                inputs = self.fc[i](inputs)
                self.neurons += inputs.numel()
                self.spikes += inputs.numel()
            else:
                inputs = -self.fc[i](-inputs)
                pass

        return inputs