import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
from torch.nn import functional, Parameter
import os
import torch.nn.functional as F
import numpy as np
import pdb
import copy


###################
# LOADING DATASET #
###################
train_dataset = dsets.CIFAR10(root='./data',
                              train=True,
                              transform=transforms.ToTensor(),
                              download=True)

test_dataset = dsets.CIFAR10(root='./data',
                             train=False,
                             transform=transforms.ToTensor())

###########################
# MAKING DATASET ITERABLE #
###########################
# print(train_dataset.train_data.size())   #torch.Size([60000, 28, 28])   # These are old parameters for MNIST, not for CIFAR 10
# print(train_dataset.train_labels.size())   #torch.Size([60000])
# print(test_dataset.test_data.size())   #torch.Size([10000, 28, 28])
# print(test_dataset.test_labels.size())   #torch.Size([10000])

batch_size = 100
num_iter = 10 ** 5
num_epoches = num_iter / (len(train_dataset) / 100)  # 5 epoches
num_epoches = int(num_epoches)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
# print(help(torch.utils.data))
# print(help(nn.Conv2d))

####################
# CREATE CNN MODEL #
####################
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Convolution Layer 1 + VALID PADDING
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        # Max Pooling 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # Convolution Layer 2 + VALID PADDING
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        # Max Pooling 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # Fully Connected Layer (readout)
        self.fc1 = nn.Linear(32 * 5 * 5, 10)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)

        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)

        # flatten
        # Original size: (100,32,5,5)
        # Out.size(0): 100
        # New out size: (100,32*5*5)
        out = out.view(out.size(0), -1)
        # Linear function for readout
        out = self.fc1(out)

        return out

##################################################
# LINEAR COMBINADTION OF WEIGHTS AND OLD WEIGHTS #
##################################################
class LinearSimple(nn.Module):
    def __init__(self, s):
        super(LinearSimple, self).__init__()
        self.P = Parameter(torch.eye(s[0]) + torch.eye(s[0]) * torch.randn(s[0], s[0]) / s[0]).cuda()
        self.register_parameter('P', self.P)

    def forward(self, w):
        return torch.matmul(self.P, w)


class controlledConv2(nn.Module):
    def __init__(self, conv, ControlType='linear', bias=None, rnk_ratio=.5, *args, **kwargs):
        super(controlledConv2, self).__init__(*args, **kwargs)
        # print('controlledConv2')
        self.conv = conv
        s = conv.weight.size()  # Copy the weights as a constant from the original convolution
        # print('list s',list(s))
        self.s = list(s)
        w = Variable(
            torch.Tensor(s).copy_(conv.weight.data))  # Copy the elements from original weights and save them to w
        w = w.view(s[0], -1)  # flatten the weights
        # print('w1',w)
        self.w = w.detach().cuda()  # Returns a new Variable, detached from the current graph
        # print('w2',w)
        # utilize GPUs for computation
        self.my_bn = None  # batch normalization layer
        s = conv.weight.size()
        # if ControlType == 'linear':
        #     ctrl = LinearSimple(s)
        # self.ctrl = ctrl
        self.P = Parameter(torch.eye(s[0]) + torch.eye(s[0]) * torch.randn(s[0], s[0]) / s[0])
        self.register_parameter('P', self.P)
        # print(self.P)

        # bias
        # s_bias = self.s[0]
        # self.conv_bias = Variable(torch.Tensor(conv.bias.data.size()).copy_(conv.bias.data))
        # self.conv_bias = self.conv_bias.detach().cuda()
        # self.bias.data.copy_(conv.bias.data[:s_bias])

    def setConvLearnable(self, T):
        for p in self.conv.parameters():
            p.requires_grad = T

    def set_bn(self, bn):
        my_bn = nn.BatchNorm2d(bn.num_features, affine=bn.affine)
        bn.eval()
        my_bn.load_state_dict(bn.state_dict())
        my_bn.train()
        self.my_bn = my_bn
        self.old_bn = bn

    def forward(self, input, alpha=None):  # TODO: take bias into consideration
        # Modify the weights
        # print('forward')
        s = self.s
        w = self.w
        # print('w', w)
        newW = torch.matmul(self.P, w)  # .cuda()
        # newW = w
        # newW = self.ctrl(w)
        # print('newW2', newW)

        if alpha is not None:
            # print 'got alpha'
            alpha1 = alpha.expand_as(w)
            newWeights = alpha1 * newW + (1 - alpha1) * w
            # print(alpha1)
            alpha2 = alpha.squeeze().expand_as(self.bias)
            # bias = alpha2 * self.bias + (1 - alpha2) * self.conv_bias
        else:
            # print('no alpha')
            newWeights = newW
            # bias = self.bias
        newWeights = newWeights.contiguous()  # unnecessary
        newWeights = newWeights.view(s)

        # print(self.conv.stride, self.conv.dilation, self.conv.padding)
        x = F.conv2d(input, newWeights, bias=None, stride=self.conv.stride,
                     padding=self.conv.padding, dilation=self.conv.dilation)
        # print('x \n', x)
        # apply the batch normalization...
        if self.my_bn is not None:
            x_bn = self.my_bn(x)
            if alpha is not None:
                alpha3 = alpha.expand_as(x)
                x = alpha3 * x_bn + (1 - alpha3) * self.old_bn(x)
            else:
                x = x_bn
        # print(x)
        return x

    def make_layer(self):
        print("make layer")
        conv_layer = self.forward()
        self.cnn1 = conv_layer(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # Convolution 2
        self.cnn2 = conv_layer(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(32 * 5 * 5, 10)

    def forward_2(self, x):
        # Convolution 1
        print('here')
        out = self.cnn1(x)
        out = self.relu1(out)

        # Max pool 1
        out = self.maxpool1(out)

        # Convolution 2
        out = self.cnn2(out)
        out = self.relu2(out)

        # Max pool 2
        out = self.maxpool2(out)

        # Resize
        # Original size: (100, 32, 5, 5)
        # out.size(0): 100
        # New out size: (100, 32*5*5)
        out = out.view(out.size(0), -1)

        # Linear function (readout)
        out = self.fc1(out)

        return out


def makeItControlled(origModule, newModule, controlAnyway=True, ControlType='linear', rnk_ratio=.5, verbose=False):
    for orig, new in zip(origModule.named_children(), newModule.named_children()):
        # print '.'
        # print('makeItControlled')
        name1, module1 = orig
        name2, module2 = new
        # if a convolution - make a controlled copy. otherwise, do nothing! everything is as it should be.
        if type(module1) is nn.Conv2d:

            # print 'setting',name2,'of new module to a controlled conv.'
            O = module1.out_channels
            I = module1.in_channels
            K = np.prod(module1.kernel_size)
            params_before = O * I * K
            if ControlType == 'diagonal':
                params_after = O
            elif ControlType == 'linear':
                params_after = O ** 2
            else:
                params_after = 2 * (O * rnk_ratio) ** 2

            if params_after < params_before or controlAnyway:
                m = controlledConv2(module1, ControlType, bias=None, rnk_ratio=0.5)  # only this line is important
                # TODO: should combine module2.bias to new model
                setattr(newModule, name1, m)

        makeItControlled(module1, module2, controlAnyway=controlAnyway,  # TODO: check if should call the class here
                         ControlType=ControlType, verbose=False, rnk_ratio=rnk_ratio)
    return newModule


#######################################
# INSTANTIATE NEWMODEL BASED ON MODEL #
#######################################
model = CNNModel()  # Regular 2-layer CNN model
# newmodel = copy.deepcopy(model)
newmodel = model
if torch.cuda.is_available():
    model.cuda()
print('model', model)
# Establish new model by going through makeItControlled
newmodel = makeItControlled(model, newmodel, controlAnyway=True, ControlType='linear', rnk_ratio=.5, verbose=False)
if torch.cuda.is_available():
    newmodel.cuda()
print('newmodel', newmodel)

# class CNNModel2(nn.Module):
#     def __init__(self):
#         super(CNNModel2, self).__init__()
#         conv_layer = controlledConv2(model)
#         # Convolution Layer 1 + VALID PADDING
#         self.cnn1 = conv_layer(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
#         self.relu1 = nn.ReLU()
#         # Max Pooling 1
#         self.maxpool1 = nn.MaxPool2d(kernel_size=2)
#         # Convolution Layer 2 + VALID PADDING
#         self.cnn2 = conv_layer(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
#         self.relu2 = nn.ReLU()
#         # Max Pooling 2
#         self.maxpool2 = nn.MaxPool2d(kernel_size=2)
#         # Fully Connected Layer (readout)
#         self.fc1 = nn.Linear(32 * 4 * 4, 10)
#
#     def forward(self, x):
#         out = self.cnn1(x)
#         out = self.relu1(out)
#         out = self.maxpool1(out)
#
#         out = self.cnn2(out)
#         out = self.relu2(out)
#         out = self.maxpool2(out)
#
#         # flatten
#         # Original size: (100,32,4,4)
#         # Out.size(0): 100
#         # New out size: (100,32*4*4)
#         out = out.view(out.size(0), -1)
#         # Linear function for readout
#         out = self.fc1(out)
#
#         return out

# newmodel = CNNModel2()
# newmodel.cuda()

#######################################
# INSTANTIATE LOSS AND OPTIMIZER CLASS#
#######################################
criterion = nn.CrossEntropyLoss()
learning_rate = 0.1  # TODO: here learning rate is fixed, so need to find out some methods, maybe not fixed?
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters())

#########################
# TRAINING WITH NEWMODEL#
#########################
iter = 0
for epoch in range(num_epoches):
    for i, (images, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(images)
            labels = Variable(labels)

        optimizer.zero_grad()  # Clear gradients first
        outputs = newmodel(images)  # Forward to get outputs
        # print(outputs)
        loss = criterion(outputs, labels)  # Cross-entropy loss function
        loss.backward()
        optimizer.step()

        iter += 1

        if iter % 500 == 0:  # Calculate Accuracy
            correct = 0
            total = 0
            for images, labels in test_loader:
                if torch.cuda.is_available():
                    images = Variable(images.cuda())
                else:
                    images = Variable(images)

                outputs = newmodel(images)
                predicted = torch.max(outputs.data, 1)[1]  # Get predictions from  the maximum value.
                # function:: max(input, dim, keepdim=False, out=None) -> (Tensor, LongTensor)
                # Returns the maximum value of each row of the :attr:`input` Tensor in the given
                # dimension :attr:`dim`. The second return value is the index location of each
                # maximum value found (argmax).
                total += labels.size(0)
                if torch.cuda.is_available():
                    correct += (predicted.cpu() == labels.cpu()).sum()
                else:
                    correct += (predicted == labels).sum()

            accuracy = 100 * correct / total

            # Print Consequences
            print('Iteration: {}. Loss: {}. Accuracy: {}.'.format(iter, loss.data[0], accuracy))
