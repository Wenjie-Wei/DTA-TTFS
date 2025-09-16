import torch
import torch.nn as nn
import torch.nn.functional as F
import math

cfg_fc = {
    'cnn': [800,128,10],
    'VGG11': [512, 4096, 4096,10],
}

cfg_conv = {
    'cnn': [16, 'M', 32, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
}

def Setconstant(Arguments):
    global device, thresh, tensorOne, tensorZero, dt, TimeLine, nonspike, Timeline_reverse, Tmax, numclass
    device = Arguments['device']

    tensorOne = Arguments['tensorOne'].to(device)
    tensorZero = Arguments['tensorZero'].to(device)
    dt = Arguments['dt']
    nonspike = Arguments['nonspike']
    TimeLine = Arguments['TimeLine'].to(device)
    Timeline_reverse = Arguments['Timeline_reverse'].to(device)
    Tmax = nonspike
    numclass = Arguments['numclass']
    thresh = torch.zeros(2*int(1/dt))
    thresh[0:int(1/dt)] = math.inf
    thresh[int(1/dt):2*int(1/dt)] = torch.range(0, int(1/dt)-1, 1).flip(0) / (1.0/dt)
    thresh = thresh.to(device)

def voltage_relay(voltage_con, weight):
    voltage_transpose = voltage_con.permute(0, 2, 1)
    weight_transpose = weight.t()
    voltage_post = voltage_transpose.matmul(weight_transpose)
    voltage_post = voltage_post.permute(0, 2, 1)
    return voltage_post

def mlp_seek_spike(voltage_post):
    voltage_binary = torch.where(voltage_post > thresh, tensorOne , tensorZero )
    tmp = voltage_binary[:, :, -1].clone()
    voltage_binary[:, :, -1] = 1.
    voltage_binary *= Timeline_reverse
    spike_time = voltage_binary.argmax(2).type(torch.float) / (1.0 / dt) - 1.0
    spike_time[tmp == 0.0] =  1.0
        
    return spike_time.to(device)

def mlp_ops(ti, W):
    ti[ti == 1.] = math.inf
    num_timeline = TimeLine.shape[0]
    shape = ti.shape

    subtract_input = torch.repeat_interleave(ti.reshape(shape[0], shape[1], 1), num_timeline, dim=2)
    tmp = F.relu(TimeLine - subtract_input)
    mem = voltage_relay(tmp, W)

    out = mlp_seek_spike(mem)
    return out

def mlp_ops_last(ti, W):
    ti[ti == 1.] = math.inf
    t = 1.0
    weight_transpose = W.t()
    spike_time = torch.ones_like(ti.matmul(weight_transpose)) * math.inf
    while True:
        diff = t - ti
        diff[ti == math.inf] = 0.0
        voltage_post = diff.matmul(weight_transpose)
        voltage_binary = torch.where(voltage_post > 20.0, tensorOne, tensorZero)
        voltage_binary[spike_time != math.inf] = 0.0
        spike_time[voltage_binary == 1.0] = t
        t += dt
        if t >= 5.0:
            break
    return spike_time

def seek_conv_spike(voltage_post):
    voltage_binary = torch.where(voltage_post > thresh, tensorOne , tensorZero )
    tmp = voltage_binary[:, :, :, :, -1].clone()
    voltage_binary[:, :, :, :, -1] = 1.
    voltage_binary *= Timeline_reverse
    spike_time = voltage_binary.argmax(4).type(torch.float) / (1.0 / dt) - 1.0
    spike_time[tmp == 0.0] =  1.0
        
    return spike_time.to(device)

def conv_ops(ti_org, W, stride, padding):

    ti_org[ti_org == 1.0] = math.inf
    num_timeline = TimeLine.shape[0]
    ti = torch.ones(ti_org.shape[0], ti_org.shape[1], ti_org.shape[2]+padding*2, ti_org.shape[3]+padding*2).cuda() * math.inf
    ti[:, :, padding:ti.shape[2]-padding, padding:ti.shape[3]-padding] = ti_org
    shape = ti.shape
    subtract_input = torch.repeat_interleave(ti.reshape(shape[0], shape[1], shape[2], shape[3], 1), num_timeline, dim=4)
    tmp = F.relu(TimeLine - subtract_input)

    # size in b*time*channels*width*height
    tmp = tmp.permute(0, 4, 1, 2, 3)
    ts = tmp.shape
    tmp = tmp.reshape(ts[0] * ts[1], ts[2], ts[3], ts[4])
    mem = F.conv2d(tmp, W, stride=stride)
    mem = mem.reshape(shape[0], num_timeline, mem.shape[1], mem.shape[2], mem.shape[3]).permute(0, 2, 3, 4, 1)

    out = seek_conv_spike(mem)
    return out

class piecewise_function_layer1(torch.autograd.Function):
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

class CNNblock(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride=1, padding=0, bias=False):
        super(CNNblock, self).__init__()
        self.cnn = nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, bias=bias,
                             padding=padding)
        self.stride = stride
        self.padding = padding
    def forward(self, input):
        cnn_out = self.cnn(input)

        return cnn_out

class MLPblock(nn.Module):
    def __init__(self, input_channel, output_channel,bias=False):
        super(MLPblock, self).__init__()
        self.linear = nn.Linear(input_channel, output_channel, bias=bias)

    def forward(self, input):
        linear_out = self.linear(input)

        return linear_out


class SpikeMLP(nn.Module):
    def __init__(self, Arguments):
        super(SpikeMLP, self).__init__()
        self.mlpname = 'mnist_mlp'
        Setconstant(Arguments)
        self.num_class = cfg_fc[-1]
        self.activation = nn.ReLU()

        self.fc = self.construct_fc()
        self.save_sign_fc = [None, None]
        self.Tmax = 10

    def construct_fc(self):
        layers = []
        input_dim = cfg_fc[0]
        for x in cfg_fc[1:]:
            layers += [MLPblock(input_dim, x)]
            input_dim = x
            layers += [self.activation]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        x = mlp_ops(x, self.fc[0].linear.weight)
        output = mlp_ops_last(x, self.fc[2].linear.weight)

        return output

class SpikeCNN(nn.Module):
    def __init__(self, Arguments,cnnname = 'cnn'):
        super(SpikeCNN, self).__init__()
        self.cnnname = cnnname
        Setconstant(Arguments)

        self.num_class = cfg_fc[self.cnnname][-1]
        self.pool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(2, 2)
        self.activation = SNNactivation_layer1()

        self.fc, self.save_sign_fc = self.construct_fc()
        self.conv, self.save_sign_conv = self.construct_conv()

    def construct_fc(self):
        layers = []
        save_tensor = []
        input_dim = cfg_fc[self.cnnname][0]
        for x in cfg_fc[self.cnnname][1:]:
            if x != self.num_class:
                layers += [
                    MLPblock(input_dim, x)
                ]
                save_tensor.append('pre_activation')
                input_dim = x
                layers += [self.activation,
                           ]
                save_tensor.append('activation')
            else:
                layers += [MLPblock(input_dim, x)]
                save_tensor.append('last_preActi')
        return nn.Sequential(*layers), save_tensor

    def construct_conv(self):
        layers = []
        input_channel = 1
        save_tensor = []
        for x in cfg_conv[self.cnnname]:
            if x == 'M':
                layers += [self.pool]
                save_tensor.append('pool')
            elif x == 'A':
                layers += [self.avgpool]
                save_tensor.append('pool')
            else:
                layers += [
                    CNNblock(input_channel, x, kernel_size=3, bias=False),
                    self.activation,
                ]
                save_tensor.append('pre_activation')
                save_tensor.append('activation')
                input_channel = x
        return nn.Sequential(*layers), save_tensor

    def forward(self, inputs):

        for i in range(len(self.save_sign_conv)):

            if self.save_sign_conv[i] == 'pre_activation':
                inputs =  conv_ops(inputs, self.conv[i].cnn.weight, self.conv[i].stride, self.conv[i].padding)

            elif self.save_sign_conv[i] == 'activation':
                pass
            else:
                inputs = -self.conv[i](-inputs)

        out = inputs.view(inputs.shape[0], -1)
        for i in range(len(self.save_sign_fc)):
            if self.save_sign_fc[i] == 'pre_activation':
                out = mlp_ops(out, self.fc[i].linear.weight)

            elif self.save_sign_fc[i] == 'activation':
                pass
            elif self.save_sign_fc[i] == 'last_preActi':
                out = mlp_ops_last(out, self.fc[i].linear.weight)
            else:
                pass

        return out

class SpikeVGG(nn.Module):
    def __init__(self, Arguments, vggname):
        super(SpikeVGG, self).__init__()
        self.vggname = vggname
        Setconstant(Arguments)
        self.num_class = cfg_fc[vggname][-1]
        self.pool = nn.MaxPool2d(2, 2)
        self.activation = SNNactivation_layer1()

        self.fc, self.save_sign_fc = self.construct_fc()
        self.conv, self.save_sign_conv = self.construct_conv()

    def construct_fc(self):
        layers = []
        save_tensor = []
        input_dim = cfg_fc[self.vggname][0]
        for x in cfg_fc[self.vggname][1:]:
            if x != self.num_class:
                layers += [MLPblock(input_dim, x)]
                save_tensor.append('pre_activation')
                input_dim = x
                layers += [self.activation,]
                save_tensor.append('activation')
            else:
                layers += [MLPblock(input_dim, x)]
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
                layers += [
                    CNNblock(input_channel, x, kernel_size=3, padding=1, stride=1, bias=False),
                    self.activation,
                ]
                save_tensor.append('pre_activation')
                save_tensor.append('activation')
                input_channel = x
        return nn.Sequential(*layers), save_tensor

    def forward(self, inputs):

        for i in range(len(self.save_sign_conv)):

            if self.save_sign_conv[i] == 'pre_activation':
                if i == 0:
                    inputs = self.conv[i](inputs)
                else:
                    inputs = conv_ops(inputs, self.conv[i].cnn.weight, self.conv[i].stride, self.conv[i].padding)
            elif self.save_sign_conv[i] == 'activation':
                if i == 1:
                    inputs = self.conv[i](inputs)
                    inputs = 1 - inputs
                else:
                    pass
            else:
                inputs = -self.conv[i](-inputs)

        out = inputs.view(inputs.shape[0], -1)
        for i in range(len(self.save_sign_fc)):
            if self.save_sign_fc[i] == 'pre_activation':
                out = mlp_ops(out, self.fc[i].linear.weight)
            elif self.save_sign_fc[i] == 'activation':
                pass
            elif self.save_sign_fc[i] == 'last_preActi':
                out = mlp_ops_last(out, self.fc[i].linear.weight)
            else:
                pass

        return out
