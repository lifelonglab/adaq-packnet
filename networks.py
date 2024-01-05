"""Contains various network definitions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models


'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import numpy as np
import pdb

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn.functional import relu


## Define ResNet18 model
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
def conv7x7(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=1, bias=False)

class GPMBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(GPMBasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes, track_running_stats=False)
            )
        self.act = OrderedDict()
        self.count = 0

    def forward(self, x):
        self.count = self.count % 2 
        self.act['conv_{}'.format(self.count)] = x
        self.count +=1
        out = relu(self.bn1(self.conv1(x)))
        self.count = self.count % 2 
        self.act['conv_{}'.format(self.count)] = out
        self.count +=1
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class GPMResNet(nn.Module):
    def __init__(self, block, num_blocks, taskcla):
        super(GPMResNet, self).__init__()
        self.in_planes = nf
        self.conv1 = conv3x3(3, nf * 1, 1)
        self.bn1 = nn.BatchNorm2d(nf * 1, track_running_stats=False)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.layer5 = nn.Linear(nf * 8 * block.expansion * 4, 10, bias=False)

        #self.taskcla = taskcla
        #self.linear=torch.nn.ModuleList()
        #for t, n in self.taskcla:
        #    self.linear.append(nn.Linear(nf * 8 * block.expansion * 4, n, bias=False))
        self.act = OrderedDict()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        bsz = x.size(0)
        self.act['conv_in'] = x.view(bsz, 3, 32, 32)
        out = relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32)))) 
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        y = self.layer5(out)
        #y=[]
        #for t,i in self.taskcla:
        #    y.append(self.linear[t](out))
        return y


def GPMResNet18(nf=32):
    return GPMResNet(GPMBasicBlock, [2, 2, 2, 2])


class SubnetAlexNet_norm(nn.Module):
    def __init__(self):
        super(SubnetAlexNet_norm, self).__init__()

        self.use_track = False

        self.in_channel = []
        self.conv1 = nn.Conv2d(3, 64, 4, bias=False)

        if self.use_track:
            self.bn1 = nn.BatchNorm2d(64, momentum=0.1)
        else:
            self.bn1 = nn.BatchNorm2d(64, track_running_stats=False, affine=False)
        s = compute_conv_output_size(32, 4)
        s = s // 2
        self.in_channel.append(3)
        self.conv2 = nn.Conv2d(64, 128, 3, bias=False)
        if self.use_track:
            self.bn2 = nn.BatchNorm2d(128, momentum=0.1)
        else:
            self.bn2 = nn.BatchNorm2d(128, track_running_stats=False, affine=False)
        s = compute_conv_output_size(s, 3)
        s = s // 2
        self.in_channel.append(64)
        self.conv3 = nn.Conv2d(128, 256, 2, bias=False)
        if self.use_track:
            self.bn3 = nn.BatchNorm2d(256, momentum=0.1)
        else:
            self.bn3 = nn.BatchNorm2d(256, track_running_stats=False, affine=False)
        s = compute_conv_output_size(s, 2)
        s = s // 2
        self.smid = s
        self.in_channel.append(128)
        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(0.2)
        self.drop2 = torch.nn.Dropout(0.5)

        self.fc1 = nn.Linear(256 * self.smid * self.smid, 2048, bias=False)
        if self.use_track:
            self.bn4 = nn.BatchNorm1d(2048, momentum=0.1)
        else:
            self.bn4 = nn.BatchNorm1d(2048, track_running_stats=False, affine=False)
        self.fc2 = nn.Linear(2048, 2048, bias=False)

        if self.use_track:
            self.bn5 = nn.BatchNorm1d(2048, momentum=0.1)
        else:
            self.bn5 = nn.BatchNorm1d(2048, track_running_stats=False, affine=False)

        self.fc3 = nn.Linear(2048, 100, bias=False)
        

    def forward(self, x):

        bsz = deepcopy(x.size(0))
        x = self.conv1(x)
        x = self.maxpool(self.drop1(self.relu(self.bn1(x))))

        x = self.conv2(x)
        x = self.maxpool(self.drop1(self.relu(self.bn2(x))))

        x = self.conv3(x)
        x = self.maxpool(self.drop2(self.relu(self.bn3(x))))

        x = x.view(bsz, -1)
        x = self.fc1(x)
        x = self.drop2(self.relu(self.bn4(x)))

        x = self.fc2(x)
        x = self.drop2(self.relu(self.bn5(x)))

        y = self.fc3(x)
        return y


## Define AlexNet model
def compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1):
    return int(np.floor((Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1))



class AlexCifarNet(nn.Module):
    supported_dims = {32}

    def __init__(self):
        super(AlexCifarNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(4096, 384),
            nn.ReLU(inplace=True),
            nn.Linear(384, 192),
            nn.ReLU(inplace=True),
            nn.Linear(192, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 4096)
        x = self.classifier(x)
        return x


class AlexNet(nn.Module):
    supported_dims = {224}

    def __init__(self):
        super(AlexNet, self).__init__()
        #self.use_dropout = state.dropout
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=2),
            #nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            #nn.MaxPool2d(kernel_size=2),
            #nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            #nn.MaxPool2d(kernel_size=2),
        )
       
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 1 * 1, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 100),
        )

    def forward(self, x):
        
        for i in range(len(self.features)):
            x = self.features[i](x)
            #sz = y.size()
            #y = y.flatten()    

            #topk = torch.topk(torch.abs(y), int(0.06*y.flatten().size()[0]))
            #y = y.flatten()
            #y[torch.abs(y) < topk[0][-1]] = 0.0

            #y = y.view(sz[0], sz[1], sz[2], sz[3])

        x = x.view(x.size(0), 256 * 1 * 1)
        x = self.classifier(x)
        return x


class TinyNet(nn.Module):
    def __init__(self, outputs, tasks, use_bn):
        super(TinyNet, self).__init__()

        self.drop1 = torch.nn.Dropout(0.0)
        self.drop2 = torch.nn.Dropout(0.0)

        self.use_bn = use_bn
        self.last = []
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=160, kernel_size=3, stride=2, padding=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(160, eps=1e-05, momentum=0.1, track_running_stats=False, affine=False)
        self.conv2 = nn.Conv2d(in_channels=160, out_channels=160, kernel_size=3, stride=2, padding=1, bias=False)
        #self.bn2 = nn.BatchNorm2d(160, eps=1e-05, momentum=0.1, track_running_stats=False, affine=False)
        self.conv3 = nn.Conv2d(in_channels=160, out_channels=160, kernel_size=3, stride=2, padding=1, bias=False)
        #self.bn3 = nn.BatchNorm2d(160, eps=1e-05, momentum=0.1, track_running_stats=False, affine=False)
        self.conv4 = nn.Conv2d(in_channels=160, out_channels=160, kernel_size=3, stride=2, padding=1, bias=False)
        #self.bn4 = nn.BatchNorm2d(160, eps=1e-05, momentum=0.1, track_running_stats=False, affine=False)
        self.ln1 = nn.Linear(2560, 640)
        #self.bl1 = nn.BatchNorm1d(640, eps=1e-05, momentum=0.1, track_running_stats=False, affine=False)
        self.ln2 = nn.Linear(640, 640)
        #self.bl2 = nn.BatchNorm1d(640, eps=1e-05, momentum=0.1, track_running_stats=False, affine=False)
        self.ln3 = nn.Linear(640, outputs)

        #for i in range(tasks):
        #    self.last.append(nn.Linear(640, outputs))


    def forward(self, x, task):
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = relu(x)
        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = relu(x)

        x = self.conv3(x)
        if self.use_bn:
            x = self.bn3(x)
        x = relu(x)

        x = self.conv4(x)
        if self.use_bn:
            x = self.bn4(x)
        x = relu(x) 
        x = self.drop1(x)

        x = x.view(x.size(0), -1)

        x = self.ln1(x)
              

        if self.use_bn:
            x = self.bl1(x)
        x = relu(x) 
        x = self.drop2(x)

        x = self.ln2(x)
        if self.use_bn:
            x = self.bl2(x)
        x = relu(x) 
        return self.ln3(x) #self.last[task](x)      


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)
        self.log_layers = []
        self.error = None

    def forward(self, x, id=-1):
        errors = []
        #out = self.features(x)
        counter = 0
        for i in range(len(self.features)): 
                  
            x = self.features[i](x)
            sz = x.size()

            #if isinstance(self.features[i], nn.Conv2d):
            #    counter += 1
            #    if counter == 2:
            #        pdb.set_trace()
                               

            #x_ = x.clone().flatten()    

            #topk = torch.topk(torch.abs(x_), int(0.3*x_.size()[0]))
            #x_ = x.clone().flatten()
            #x_[torch.abs(x_) < topk[0][-1]] = 0.0
            #x_ = x_.view(sz[0], sz[1], sz[2], sz[3])

            #x = x_

            #if i in self.log_layers and id > -1 and self.error == 0:
            #    np.save(str(id) + '_' + str(i) + '_vgg', x.cpu().numpy())
            #if residual error - read file and compare in layers
            #if self.error and id > -1:
            #    if i in self.log_layers:
            #        x_ = np.load(str(id) + '_' + str(i) + '_vgg.npy')
            #        error = torch.from_numpy(x_).cuda() - x
            #        np.save(str(id) + '_' + str(i) + '_vgg_error', error.cpu().numpy())
            #        errors.append(error)

        out = x

        #pdb.set_trace()
        #layer by layer
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        #if id > -1 and self.error == 0:
        #    np.save(str(id) + '_out_vgg', x.cpu().numpy())

        #if self.error and id > -1: 
        #    x_ = np.load(str(id) + '_out_vgg.npy')
        #    error = torch.from_numpy(x_).cuda() - x
        #    np.save(str(id) + '_out_vgg_error', error.cpu().numpy())
        #    errors.append(error)
        return out

    def _make_layers(self, cfg):
            
        layers = []
        in_channels = 3
        counter = 0

        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if counter == 0:
                    layers += [nn.Conv2d(1, x, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                else:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                in_channels = x
                counter = 1

        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class MLP(nn.Module):
    def __init__(self, input_size, output_size,
                 hidden_size=100,
                 hidden_layer_num=2,
                 hidden_dropout_prob=.5,
                 input_dropout_prob=.2,
                 lamda=40):
        # Configurations.
        super().__init__()
        self.input_size = input_size*input_size
        self.input_dropout_prob = input_dropout_prob
        self.hidden_size = hidden_size
        self.hidden_layer_num = hidden_layer_num
        self.hidden_dropout_prob = hidden_dropout_prob
        self.output_size = output_size
        self.lamda = lamda

        #self.active_dendrite = []
       
        #self.active_dendrite.append(nn.Linear(28*28*1, 10).cuda())
        #self.active_dendrite.append(nn.Linear(28*28*1, 10).cuda())
        #self.active_dendrite.append(nn.Linear(28*28*1, 10).cuda())
        #self.active_dendrite.append(nn.Linear(28*28*1, 10).cuda())
        #self.active_dendrite.append(nn.Linear(28*28*1, 10).cuda())
        
        # Layers.
        #self.layers = nn.ModuleList([
        #    # input
        #    nn.Linear(self.input_size, self.hidden_size), nn.ReLU(),
        #    nn.Dropout(self.input_dropout_prob),
        #    # hidden
        #    *((nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(),
        #       nn.Dropout(self.hidden_dropout_prob)) * self.hidden_layer_num),
        #    # output
        #    nn.Linear(self.hidden_size, self.output_size)
        #])

        self.ln1 = nn.Linear(self.input_size, self.hidden_size)
        self.r1 = nn.ReLU()
        self.d1 = nn.Dropout(self.input_dropout_prob)
        #self.ln2 = nn.Linear(self.hidden_size, self.hidden_size)
        #self.r2 = nn.ReLU()
        #self.d2 = nn.Dropout(self.hidden_dropout_prob)
        self.ln3 = nn.Linear(self.hidden_size, self.output_size)

        #self.conv1 = nn.Conv2d(1, 6, 5)
        #self.relu1 = nn.ReLU()
        
        #self.pool1 = nn.MaxPool2d(2)
        #self.conv2 = nn.Conv2d(6, 16, 5)
        #self.relu2 = nn.ReLU()
        
        #self.pool2 = nn.MaxPool2d(2)
        #self.fc1 = nn.Linear(256, 120)
        #self.relu3 = nn.ReLU()
        #self.fc2 = nn.Linear(120, 84)
        #self.relu4 = nn.ReLU()
        #self.fc3 = nn.Linear(84, 10)
        #self.relu5 = nn.ReLU()


    def forward(self, x):

        #pdb.set_trace()
        x = x.view(x.size(0), x.size(1)*x.size(2))
        #sigma = torch.sigmoid(torch.max(self.active_dendrite[0](x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])), dim=1)[0])
        y = self.ln1(x)
        #for j in range(sigma.size(0)):
        #    y[j,:] = y[j,:].clone()*sigma[j] 
          
        y = self.r1(y)

        sz = y.size()
        #y = y.flatten()    

        #topk = torch.topk(torch.abs(y), int(0.2*y.flatten().size()[0]))
        #y = y.flatten()
        #y[torch.abs(y) < topk[0][-1]] = 0.0

        #y = y.view(sz[0], sz[1])
        #y = self.d1(y)

        #sigma = torch.sigmoid(torch.max(self.active_dendrite[1](x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])), dim=1)[0])
        #y = self.ln2(y)
        #for j in range(sigma.size(0)):
        #    y[j,:] = y[j,:].clone()*sigma[j]

        #y = self.r2(y)
        #y = self.d2(y)

        #pdb.set_trace()
        #y = y.view(y.shape[0], -1)

        #sigma = torch.sigmoid(torch.max(self.active_dendrite[2](x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])), dim=1)[0])
        #y = self.fc1(y)

        #for j in range(sigma.size(0)):
        #    y[j,:] = y[j,:].clone()*sigma[j]

        #y = self.relu3(y)
  
        #sigma = torch.sigmoid(torch.max(self.active_dendrite[3](x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])), dim=1)[0])
        #y = self.fc2(y)

        #for j in range(sigma.size(0)):
        #    y[j,:] = y[j,:].clone()*sigma[j]

        #y = self.relu4(y)

        #sigma = torch.sigmoid(torch.max(self.active_dendrite[4](x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])), dim=1)[0])
        #y = self.fc3(y)

        #for j in range(sigma.size(0)):
        #    y[j,:] = y[j,:].clone()*sigma[j]

        #y = self.relu5(y)
        y = self.ln3(y)
        sz = y.size()
        #y = y.flatten()    

        #topk = torch.topk(torch.abs(y), int(0.7*y.flatten().size()[0]))
        #y = y.flatten()
        #y[torch.abs(y) < topk[0][-1]] = 0.0

        #y = y.view(sz[0], sz[1])
    
        return y


class MLP_(nn.Module):
    def __init__(self, input_size, output_size,
                 hidden_size=100,
                 hidden_layer_num=2,
                 hidden_dropout_prob=.5,
                 input_dropout_prob=.2,
                 lamda=40):
        # Configurations.
        super().__init__()
        self.input_size = input_size*input_size
        self.input_dropout_prob = input_dropout_prob
        self.hidden_size = hidden_size
        self.hidden_layer_num = hidden_layer_num
        self.hidden_dropout_prob = hidden_dropout_prob
        self.output_size = output_size
        self.lamda = lamda

        #self.active_dendrite = []
        #self.ln0 = nn.Linear(32*32*1, 10)
        #self.active_dendrite.append(nn.Linear(32*32*1, 10).cuda())
        #self.active_dendrite.append(nn.Linear(28*28*1, 10).cuda())
        #self.active_dendrite.append(nn.Linear(28*28*1, 10).cuda())
        #self.active_dendrite.append(nn.Linear(28*28*1, 10).cuda())
        #self.active_dendrite.append(nn.Linear(28*28*1, 10).cuda())
        
        # Layers.
        #self.layers = nn.ModuleList([
        #    # input
        #    nn.Linear(self.input_size, self.hidden_size), nn.ReLU(),
        #    nn.Dropout(self.input_dropout_prob),
        #    # hidden
        #    *((nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(),
        #       nn.Dropout(self.hidden_dropout_prob)) * self.hidden_layer_num),
        #    # output
        #    nn.Linear(self.hidden_size, self.output_size)
        #])

        self.ln1 = nn.Linear(self.input_size, self.hidden_size)
        self.r1 = nn.ReLU()
        self.d1 = nn.Dropout(self.input_dropout_prob)
        #self.ln2 = nn.Linear(self.hidden_size, self.hidden_size)
        #self.r2 = nn.ReLU()
        #self.d2 = nn.Dropout(self.hidden_dropout_prob)
        self.ln3 = nn.Linear(self.hidden_size, self.output_size)

        #self.conv1 = nn.Conv2d(1, 6, 5)
        #self.relu1 = nn.ReLU()
        
        #self.pool1 = nn.MaxPool2d(2)
        #self.conv2 = nn.Conv2d(6, 16, 5)
        #self.relu2 = nn.ReLU()
        
        #self.pool2 = nn.MaxPool2d(2)
        #self.fc1 = nn.Linear(256, 120)
        #self.relu3 = nn.ReLU()
        #self.fc2 = nn.Linear(120, 84)
        #self.relu4 = nn.ReLU()
        #self.fc3 = nn.Linear(84, 10)
        #self.relu5 = nn.ReLU()
        self.ln4 = nn.Linear(32*32*1, 10)


    def forward(self, x, c=None):

        #pdb.set_trace()
        if c is None:
            sigma = torch.sigmoid(torch.max(self.ln4(x.view(x.shape[0], x.shape[1]*x.shape[2])), dim=1)[0])
        else:
            #c = np.reshape(c, (1, c.shape[0], c.shape[1]))
            c = c.cuda()
            sigma = torch.sigmoid(torch.max(self.ln4(c.view(c.shape[0], c.shape[1]*c.shape[2])), dim=1)[0])
        #sigma = torch.sigmoid(torch.max(self.active_dendrite[0](x.view(x.shape[0], x.shape[1]*x.shape[2])), dim=1)[0])
        x = x.view(x.size(0), x.size(1)*x.size(2))
        
        y = self.ln1(x)
        for j in range(sigma.size(0)):
            y[j,:] = y[j,:].clone()*sigma[j] 
          
        y = self.r1(y)

        #sz = y.size()
        #y = y.flatten()    

        #topk = torch.topk(torch.abs(y), int(0.7*y.flatten().size()[0]))
        #y = y.flatten()
        #y[torch.abs(y) < topk[0][-1]] = 0.0

        #y = y.view(sz[0], sz[1])
        y = self.d1(y)

        #sigma = torch.sigmoid(torch.max(self.active_dendrite[1](x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])), dim=1)[0])
        #y = self.ln2(y)
        #for j in range(sigma.size(0)):
        #    y[j,:] = y[j,:].clone()*sigma[j]

        #y = self.r2(y)
        #y = self.d2(y)

        #pdb.set_trace()
        #y = y.view(y.shape[0], -1)

        #sigma = torch.sigmoid(torch.max(self.active_dendrite[2](x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])), dim=1)[0])
        #y = self.fc1(y)

        #for j in range(sigma.size(0)):
        #    y[j,:] = y[j,:].clone()*sigma[j]

        #y = self.relu3(y)
  
        #sigma = torch.sigmoid(torch.max(self.active_dendrite[3](x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])), dim=1)[0])
        #y = self.fc2(y)

        #for j in range(sigma.size(0)):
        #    y[j,:] = y[j,:].clone()*sigma[j]

        #y = self.relu4(y)

        #sigma = torch.sigmoid(torch.max(self.active_dendrite[4](x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])), dim=1)[0])
        #y = self.fc3(y)

        #for j in range(sigma.size(0)):
        #    y[j,:] = y[j,:].clone()*sigma[j]

        #y = self.relu5(y)
        y = self.ln3(y)
        sz = y.size()
        #y = y.flatten()    

        #topk = torch.topk(torch.abs(y), int(0.7*y.flatten().size()[0]))
        #y = y.flatten()
        #y[torch.abs(y) < topk[0][-1]] = 0.0

        #y = y.view(sz[0], sz[1])
    
        return y


class View(nn.Module):
    """Changes view using a nn.Module."""

    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


class ModifiedVGG16(nn.Module):
    """VGG16 with different classifiers."""

    def __init__(self, make_model=True):
        super(ModifiedVGG16, self).__init__()
        if make_model:
            self.make_model()

    def make_model(self):
        """Creates the model."""
        # Get the pretrained model.
        vgg16 = models.vgg16(pretrained=True)
        self.datasets, self.classifiers = [], nn.ModuleList()

        idx = 6
        for module in vgg16.classifier.children():
            if isinstance(module, nn.Linear):
                if idx == 6:
                    fc6 = module
                elif idx == 7:
                    fc7 = module
                elif idx == 8:
                    self.datasets.append('imagenet')
                    self.classifiers.append(module)
                idx += 1
        features = list(vgg16.features.children())
        features.extend([
            View(-1, 25088),
            fc6,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            fc7,
            nn.ReLU(inplace=True),
            nn.Dropout(),
        ])

        # Shared params are those which are to be pruned.
        self.shared = nn.Sequential(*features)

        # model.set_dataset() has to be called explicity, else model won't work.
        self.classifier = nn.Linear(4096, 10) #None

        # Make sure conv transform is correct.
        # self.check_correctness(vgg16)

    def add_dataset(self, dataset, num_outputs):
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            self.classifiers.append(nn.Linear(4096, num_outputs))

    def set_dataset(self, dataset):
        """Change the active classifier."""
        assert dataset in self.datasets
        self.classifier = self.classifiers[self.datasets.index(dataset)]

    def forward(self, x):
        x = self.shared(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_nobn(self, mode=True):
        """Override the default module train."""
        super(ModifiedVGG16, self).train(mode)

    def check_correctness(self, vgg16):
        """Ensures that conversion of fc layers to conv is correct."""
        # Test to make sure outputs match.
        vgg16.eval()
        self.shared.eval()
        self.classifier.eval()

        rand_input = Variable(torch.rand(1, 3, 224, 224))
        fc_output = vgg16(rand_input)
        print(fc_output)

        x = self.shared(rand_input)
        x = x.view(x.size(0), -1)
        conv_output = self.classifier[-1](x)
        print(conv_output)

        print(torch.sum(torch.abs(fc_output - conv_output)))
        assert torch.sum(torch.abs(fc_output - conv_output)).data[0] < 1e-8
        print('Check passed')
        raw_input()


class ModifiedVGG16BN(ModifiedVGG16):
    """VGG16 with batch norm."""

    def __init__(self, make_model=True):
        super(ModifiedVGG16BN, self).__init__(make_model=False)
        if make_model:
            self.make_model()

    def train_nobn(self, mode=True):
        """Override the default module train."""
        super(ModifiedVGG16BN, self).train(mode)

        # Set the BNs to eval mode so that the running means and averages
        # do not update.
        for module in self.shared.children():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def make_model(self):
        """Creates the model."""
        # Get the pretrained model.
        vgg16_bn = models.vgg16_bn(pretrained=True)
        self.datasets, self.classifiers = [], nn.ModuleList()

        # Get classifiers.
        idx = 6
        for module in vgg16_bn.classifier.children():
            if isinstance(module, nn.Linear):
                if idx == 6:
                    fc6 = module
                elif idx == 7:
                    fc7 = module
                elif idx == 8:
                    self.datasets.append('imagenet')
                    self.classifiers.append(module)
                idx += 1

        features = list(vgg16_bn.features.children())
        features.extend([
            View(-1, 25088),
            fc6,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            fc7,
            nn.ReLU(inplace=True),
            nn.Dropout(),
        ])

        # Shared params are those which are to be pruned.
        self.shared = nn.Sequential(*features)

        # model.set_dataset() has to be called explicity, else model won't work.
        self.classifier = None

        # Make sure conv transform is correct.
        # self.check_correctness(vgg16)


class ModifiedResNet(ModifiedVGG16):
    """ResNet-50."""

    def __init__(self, make_model=True):
        super(ModifiedResNet, self).__init__(make_model=False)
        if make_model:
            self.make_model()

    def train_nobn(self, mode=True):
        """Override the default module train."""
        super(ModifiedResNet, self).train(mode)

        # Set the BNs to eval mode so that the running means and averages
        # do not update.
        for module in self.shared.modules():
            if 'BatchNorm' in str(type(module)):
                module.eval()

    def make_model(self):
        """Creates the model."""
        # Get the pretrained model.
        resnet = models.resnet50(pretrained=True)
        self.datasets, self.classifiers = [], nn.ModuleList()

        # Create the shared feature generator.
        self.shared = nn.Sequential()
        for name, module in resnet.named_children():
            if name != 'fc':
                self.shared.add_module(name, module)

        # Add the default imagenet classifier.
        self.datasets.append('imagenet')
        self.classifiers.append(resnet.fc)

        # model.set_dataset() has to be called explicity, else model won't work.
        self.classifier = None

    def add_dataset(self, dataset, num_outputs):
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            self.classifiers.append(nn.Linear(2048, num_outputs))


class ModifiedDenseNet(ModifiedVGG16):
    """DenseNet-121."""

    def __init__(self, make_model=True):
        super(ModifiedDenseNet, self).__init__(make_model=False)
        if make_model:
            self.make_model()

    def train_nobn(self, mode=True):
        """Override the default module train."""
        super(ModifiedDenseNet, self).train(mode)

        # Set the BNs to eval mode so that the running means and averages
        # do not update.
        for module in self.shared.modules():
            if 'BatchNorm' in str(type(module)):
                module.eval()

    def make_model(self):
        """Creates the model."""
        # Get the pretrained model.
        densenet = models.densenet121(pretrained=True)
        self.datasets, self.classifiers = [], nn.ModuleList()

        # Create the shared feature generator.
        self.shared = densenet.features

        # Add the default imagenet classifier.
        self.datasets.append('imagenet')
        self.classifiers.append(densenet.classifier)

        # model.set_dataset() has to be called explicity, else model won't work.
        self.classifier = None

    def forward(self, x):
        features = self.shared(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7).view(features.size(0), -1)
        out = self.classifier(out)
        return out

    def add_dataset(self, dataset, num_outputs):
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            self.classifiers.append(nn.Linear(1024, num_outputs))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

number_of_classes = 100

class BasicBlock(nn.Module):
    expansion = 1
    statistics = 0

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        
        #self.conv1.weight = torch.nn.Parameter(self.conv1.weight * self.mask1) 
        out = F.relu(self.bn1(self.conv1(x)))
        #self.conv2.weight = torch.nn.Parameter(self.conv2.weight * self.mask2)
        out = F.relu(self.bn2(self.conv2(out)))
        
        #self.conv3.weight = torch.nn.Parameter(self.conv3.weight * self.mask3)
        out = self.bn3(self.conv3(out))

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    
    heatmaps = {}    

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 32

        self.conv1 = nn.Conv2d(3, self.in_planes*1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes*1)
        self.layer1 = self._make_layer(block, self.in_planes*1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, self.in_planes*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, self.in_planes*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, self.in_planes*8, num_blocks[3], stride=2)
        self.linear = nn.Linear(100352, num_classes) #(self.in_planes*8*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

