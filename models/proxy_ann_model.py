import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg_fc = {
    'CNN': [512,800,128,10],
    'VGG11': [512, 4096, 4096,10],
    'VGG16': [512, 4096, 4096,10],
    # 'VGG16': [25088, 4096, 4096,1000],
}

cfg_conv = {
    'CNN': [16, 'M', 32, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}

class piecewise_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, pre_acti, w_sum):
        correct_maximum = 1.

        input[pre_acti < -w_sum] = 0.
        input[pre_acti >= correct_maximum] = correct_maximum

        out = input
        if len(torch.where(input<0)[0]) != 0:
            print('有小于0的激活值')
        if len(torch.where(input>correct_maximum)[0]) != 0:
            print('有大于1的激活值')

        valid_mask = pre_acti.lt(correct_maximum) * pre_acti.gt(-w_sum)

        ctx.save_for_backward(valid_mask)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        mask = ctx.saved_tensors[0]
        dE = grad_output * mask.float()
        return dE, None, None, None, None

class SNNactivation(nn.Module):
    def __init__(self):
        super(SNNactivation, self).__init__()

    def forward(self, input, pre_acti, w_sum):
        return piecewise_function.apply(input, pre_acti, w_sum)


class gradients_clip(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight_sum,ann_out):


        z = (weight_sum + ann_out)/(1 + weight_sum)

        valid_mask = ann_out.lt(1.) * ann_out.gt(-weight_sum)

        ctx.save_for_backward(weight_sum,ann_out,valid_mask)

        return z

    @staticmethod
    def backward(ctx, grad_input):
        bigger_clip = 11

        weight_sum, ann_out, valid_mask = ctx.saved_tensors

        grad_weight_sum = (1 - ann_out)/((1+weight_sum)**2)     # 激活值关于两项的导数
        grad_ann_out = 1 / (1 + weight_sum)

        grad_weight_sum[grad_weight_sum > bigger_clip] = bigger_clip
        grad_weight_sum[grad_weight_sum < -bigger_clip] = -bigger_clip

        grad_ann_out[grad_ann_out > bigger_clip] = bigger_clip
        grad_ann_out[grad_ann_out < -bigger_clip] = -bigger_clip

        grad_weight_sum = grad_weight_sum * valid_mask
        grad_ann_out = grad_ann_out * valid_mask

        return grad_input*grad_weight_sum, grad_input*grad_ann_out

grad_clip = gradients_clip.apply

class MLPblock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(MLPblock, self).__init__()
        self.linear = nn.Linear(input_channel, output_channel, bias=False)

        nn.init.kaiming_normal_(self.linear.weight.data, a=0, mode='fan_in', nonlinearity='relu')

    def forward(self, inputs):
        valid_inputs = (inputs > 0).float().clone().detach()
        weight_sum = self.linear(valid_inputs)

        linear_out = self.linear(inputs)
        outputs = grad_clip(weight_sum,linear_out)

        return outputs, linear_out, weight_sum

class CNNblock(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride=1, bias=False, padding=0):
        super(CNNblock, self).__init__()
        self.cnn = nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, bias=bias,
                             padding=padding)
        self.stride = stride
        self.padding = padding
        nn.init.kaiming_normal_(self.cnn.weight, a=0, mode='fan_in', nonlinearity='relu')

    def forward(self, inputs):
        valid_inputs = (inputs > 0).detach().float()
        weight_sum = self.cnn(valid_inputs)

        cnn_out = self.cnn(inputs)
        outputs = grad_clip(weight_sum,cnn_out)

        return outputs, cnn_out, weight_sum

class Finalblock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(Finalblock, self).__init__()
        self.linear = nn.Linear(input_channel, output_channel, bias=False)

        nn.init.kaiming_normal_(self.linear.weight.data, a=0, mode='fan_in', nonlinearity='relu')

        self.W_sum = None

    def forward(self, inputs):
        valid_inputs = (inputs > 0).detach().float()
        weight_sum = self.linear(valid_inputs)

        linear_out = self.linear(inputs)

        self.W_sum = weight_sum

        return linear_out

class SpikeVGG(nn.Module):
    def __init__(self, vggname):
        super(SpikeVGG, self).__init__()
        self.vggname = vggname

        self.num_class = cfg_fc[vggname][-1]
        self.pool = nn.MaxPool2d(2, 2)
        self.activation = SNNactivation()

        self.fc, self.save_sign_fc = self.construct_fc()
        self.conv, self.save_sign_conv = self.construct_conv()

        # cal sparsity
        self.spikes = 0.
        self.neurons = 0.

    def construct_fc(self):
        layers = []
        save_tensor = []
        input_dim = cfg_fc[self.vggname][0]
        for x in cfg_fc[self.vggname][1:]:
            if x != self.num_class:
                layers += [MLPblock(input_dim, x)]
                save_tensor.append('pre_activation')
                input_dim = x
                layers += [
                    self.activation
                           ]
                save_tensor.append('activation')
            else:
                layers += [Finalblock(input_dim, x)]
                save_tensor.append('last_preActi')
        return nn.Sequential(*layers), save_tensor

    def construct_conv(self):
        layers = []
        input_channel =3
        save_tensor = []
        for x in cfg_conv[self.vggname]:
            if x == 'M':
                layers += [self.pool]
                save_tensor.append('pool')
            else:
                layers += [
                    CNNblock(input_channel, x, kernel_size=3, padding=1, stride=1, bias=False), # 3, 1, 1
                    self.activation,
                ]
                save_tensor.append('pre_activation')
                save_tensor.append('activation')
                input_channel = x
        return nn.Sequential(*layers), save_tensor

    def forward(self, inputs):

        for i in range(len(self.save_sign_conv)):

            if self.save_sign_conv[i] == 'pre_activation':
                inputs, pre_acti, w_sum = self.conv[i](inputs)
            elif self.save_sign_conv[i] == 'activation':
                inputs = self.conv[i](inputs, pre_acti, w_sum)
                self.neurons += inputs.numel()
                self.spikes += (inputs>0).float().sum()
            else:
                inputs = self.conv[i](inputs)
                pass

        out = inputs.view(inputs.shape[0], -1)
        for i in range(len(self.save_sign_fc)):
            if self.save_sign_fc[i] == 'pre_activation':
                out, pre_acti, w_sum  = self.fc[i](out)
            elif self.save_sign_fc[i] == 'activation':
                out = self.fc[i](out,pre_acti,w_sum)
                self.neurons += out.numel()
                self.spikes += (out>0).float().sum()
            elif self.save_sign_fc[i] == 'last_preActi':
                out = self.fc[i](out)
                self.neurons += out.numel()
                self.spikes += out.numel()
            else:
                out = self.fc[i](out)
                pass

        return out