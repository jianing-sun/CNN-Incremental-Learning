import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
from torch.nn import functional, Parameter
import os
import torch.nn.functional as F
import numpy as np
from my_dataset import MYCIFAR10

'''
Hyper-parameters
'''
torch.manual_seed(1234)
dataset_identifier = {'MNIST': 0, 'CIFAR10': 1}
batch_size = 100
n_iters = 10 ** 5
validation_ratio = 0.1



def get_data(dataset_name):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = eval('MY' + dataset_name)(ratio=validation_ratio,
                                              root='./data',
                                              train=True,
                                              transform=transform,
                                              download=True)

    validat_dataset = train_dataset.get_validat_set()

    test_dataset = eval('dsets.' + dataset_name)(root='./data',
                                                 train=False,
                                                 transform=transforms,
                                                 download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    validate_loader = torch.utils.data.DataLoader(dataset=validat_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    return train_loader, validate_loader, test_loader, len(train_dataset), len(test_dataset)



class ContConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(ContConv2d, self).__init__(*args, **kwargs)

    def forward(self, input):
        comb_weight = self.weight.mm(self.old_filters.view(self.out_channels, -1)).view(self.out_channels,
                                                                                        self.in_channels,
                                                                                        *self.kernel_size)
        return functional.conv2d(input, comb_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def add_controller(self):
        # save old filters and bias, place controller module parameters
        self.old_filters = self.weight
        self.old_bias = self.bias
        if torch.cuda.is_available():
            self.weight = nn.Parameter(torch.eye(self.out_channels).cuda())
            self.bias = nn.Parameter(torch.randn(self.out_channels).cuda())
        else:
            self.weight = nn.Parameter(torch.eye(self.out_channels))
            self.bias = nn.Parameter(torch.randn(self.out_channels))


class CNNModel(nn.Module):
    def __init__(self, dataset_id, load_state):
        super(CNNModel, self).__init__()
        self.dataset_id = dataset_id
        self.load_state = load_state
        self.make_layer()

    def make_layer(self):
        # load_state = True means we are going to experiment on transfer learning;
        # otherwise, it's only an initial learning
        if self.load_state:
            conv_layer = ContConv2d
            print("True")
        else:
            conv_layer = nn.Conv2d

        # Convolution 1
        if self.dataset_id == 0:
            # MNIST dataset, in_channel is 1
            self.cnn1 = conv_layer(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        else:
            # CIFAR dataset, in_channel is 3
            self.cnn1 = conv_layer(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()

        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Convolution 2
        self.cnn2 = conv_layer(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()

        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Fully connected 1 (readout)
        if self.dataset_id == 0:
            self.fc1 = nn.Linear(32 * 4 * 4, 10)
        else:
            self.fc1 = nn.Linear(32 * 5 * 5, 10)

    def forward(self, x):
        # Convolution 1
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
        # Original size: (100, 32, 7, 7)
        # out.size(0): 100
        # New out size: (100, 32*7*7)
        out = out.view(out.size(0), -1)

        # Linear function (readout)
        out = self.fc1(out)

        return out

    def load_state_dict(self, state_dict):
        # super(CNNModel, self).load_state_dict(state_dict)

        own_state = self.state_dict()
        for name, param in state_dict.items():

            # only copy the filters in convolutional layers
            if not name.lower().startswith('cnn'):
                continue

            if name not in own_state:
                raise KeyError('unexpected key "{}" in state_dict'.format(name))

            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data

            try:
                own_state[name].copy_(param)
            except:
                print('While copying the parameter named {}, whose dimensions in the model are'
                      ' {} and whose dimensions in the checkpoint are {}, ...'.format(
                    name, own_state[name].size(), param.size()))
                raise

        # for stationary model structure, there shouldn't be missing states
        missing = set(own_state.keys()) - set(state_dict.keys())
        if len(missing) > 0:
            raise KeyError('missing keys in state_dict: "{}"'.format(missing))

        for module in self.modules():
            if isinstance(module, ContConv2d):
                module.add_controller()

    def add_controller(self):
        cont_params = []
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                for name, param in module.named_parameters():
                    if name == 'weight':
                        param.require_grad = False
                        cont_param = nn.Parameter(torch.randn(param.size()[0], param.size()[0]))
                        comb_param = cont_param.mm(param.view(param.size()[0], -1)).view(*param.size())
                        module._parameters['weight'] = comb_param
                        cont_params.append(cont_param)
        return cont_params

    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['state_dict'])

    def save(self, best_path):
        torch.save({'state_dict': best_path}, './model_params/param.pth.tar')

import copy

def get_model(dataset_id, load_state=True):
    # model = CNNModel(dataset_id, load_state)
    model = CNNModel2()
    newModel = copy.deepcopy(model)

    # if load_state:
    #     if os.path.exists('./model_params/param.pth.tar'):
    #         model.load('./model_params/param.pth.tar')
    #     else:
    #         raise RuntimeError('./model_params/param.pth.tar' + ' does not exist')

    if torch.cuda.is_available():
        model.cuda()
    return model, newModel


def train(model, train_loader, validate_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    # num_epochs is the upper bound
    num_epochs = int(n_iters / (len_train / batch_size))
    iter = 0
    best_acc = 0
    best_path = None
    failure = 0

    out_file = open('outcome.txt', 'a')
    out_file.write('TRAINING PHASE\n')
    for epoch in range(num_epochs):

        for i, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
            else:
                images = Variable(images)
                labels = Variable(labels)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            iter += 1

            if iter % 500 == 0:
                # Calculate Accuracy
                correct = 0
                total = 0
                # Iterate through validate dataset
                for images, labels in validate_loader:
                    if torch.cuda.is_available():
                        images = Variable(images.cuda())
                    else:
                        images = Variable(images)
                    outputs = model(images)
                    # Get predictions from the maximum value
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    # Total correct predictions
                    if torch.cuda.is_available():
                        correct += (predicted.cpu() == labels.cpu()).sum()
                    else:
                        correct += (predicted == labels).sum()

                accuracy = 100 * correct / total
                # Print Loss
                print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.data[0], accuracy))
                # out_file.write('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.data[0], accuracy) + '\n')

    #             if accuracy >= best_acc:
    #                 prev_acc = accuracy
    #                 best_path = model.state_dict()
    #                 failure = 0
    #             else:
    #                 failure += 1
    #
    #     if failure >= 10:
    #         break
    # out_file.close()
    # if not os.path.exists('./model_params/param.pth.tar'):
    #     model.save(best_path)
    # return model.load_state_dict(best_path)


def _test(model, test_loader):
    correct = 0
    total = 0
    out_file = open('outcome.txt', 'a')
    out_file.write('TESTING PHASE\n')
    for images, labels in test_loader:
        if torch.cuda.is_available():
            images = Variable(images.cuda())
        else:
            images = Variable(images)
        outputs = model(images)
        # Get predictions from the maximum value
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        # Total correct predictions
        if torch.cuda.is_available():
            correct += (predicted.cpu() == labels.cpu()).sum()
        else:
            correct += (predicted == labels).sum()

    accuracy = 100 * correct / total
    # Print Loss
    print('Test Data Accuracy: {}'.format(accuracy))
    out_file.write('Test Data Accuracy: {}'.format(accuracy) + '\n')
    out_file.close()


class LinearSimple(nn.Module):
    def __init__(self, s):
        super(LinearSimple, self).__init__()
        self.P = Parameter(torch.eye(s[0]) + torch.eye(s[0]) * torch.randn(s[0], s[0]) / s[0])
        self.register_parameter('P', self.P)

    def forward(self, w):
        return torch.matmul(self.P, w)


class controlledConv2(nn.Module):
    def __init__(self, conv, ControlType = 'linear', bias=None, rnk_ratio=.5):
        super(controlledConv2, self).__init__()
        self.conv = conv
        s = conv.weight.size()  # Copy the weights as a constant from the original convolution
        self.s = list(s)
        w = Variable(
            torch.Tensor(s).copy_(conv.weight.data))  # Copy the elements from original weights and save them to w
        w = w.view(s[0], -1)  # flatten the weights
        self.w = w.detach().cuda()  # Returns a new Variable, detached from the current graph
        # utilize GPUs for computation
        self.my_bn = None  # batch normalization layer
        s = conv.weight.size()
        if ControlType == 'linear':
            ctrl = LinearSimple(s)
        self.ctrl = ctrl
        # bias
        # s_bias = self.s[0]
        # self.conv_bias = Variable(torch.Tensor(conv.bias.data.size()).copy_(conv.bias.data))
        # self.conv_bias = self.conv_bias.detach().cuda()
        # self.bias.data.copy_(conv.bias.data[:s_bias])

    def set_bn(self, bn):
        my_bn = nn.BatchNorm2d(bn.num_features, affine=bn.affine)
        bn.eval()
        my_bn.load_state_dict(bn.state_dict())
        my_bn.train()
        self.my_bn = my_bn
        self.old_bn = bn

    def forward(self, x, alpha=None):
        # Modify the weights
        s = self.s
        w = self.w

        newW = self.ctrl(w)

        if alpha is not None:
            # print 'got alpha'
            alpha1 = alpha.expand_as(w)
            newWeights = alpha1 * newW + (1 - alpha1) * w

            alpha2 = alpha.squeeze().expand_as(self.bias)
            bias = alpha2 * self.bias + (1 - alpha2) * self.conv_bias
        else:
            # print 'no alpha'
            newWeights = newW
            bias = self.bias
        newWeights = newWeights.contiguous()  # unnecessary
        newWeights = newWeights.view(s)

        x = F.conv2d(x, newWeights, bias, stride=self.conv.stride,
                     padding=self.conv.padding, dilation=self.conv.dilation)

        # apply the batch normalization...
        if self.my_bn is not None:
            x_bn = self.my_bn(x)
            if alpha is not None:
                alpha3 = alpha.expand_as(x)
                x = alpha3 * x_bn + (1 - alpha3) * self.old_bn(x)
            else:
                x = x_bn
        return x


def makeItControlled(origModule, newModule, controlAnyway=True, ControlType='linear', rnk_ratio=.5, verbose=False):
    for orig, new in zip(origModule.named_children(), newModule.named_children()):
        # print '.'
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
                # print 'creating controlled conv'
                m = controlledConv2(module1, ControlType, module2, rnk_ratio)
                setattr(newModule, name1, m)   # newModule.name1 = m
                # else:
                #    pass
                # print '...meh'
                # leave it as it is.
        # print 'going into', name1
        makeItControlled(module1, module2, controlAnyway=controlAnyway,
                         ControlType=ControlType, verbose=verbose, rnk_ratio=rnk_ratio)


class CNNModel2(nn.Module):
    def __init__(self):
        super(CNNModel2, self).__init__()
        # Convolution Layer 1 + VALID PADDING
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        # Max Pooling 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # Convolution Layer 2 + VALID PADDING
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        # Max Pooling 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # Fully Connected Layer (readout)
        self.fc1 = nn.Linear(32 * 4 * 4, 10)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)

        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)

        # flatten
        # Original size: (100,32,4,4)

        # Out.size(0): 100
        # New out size: (100,32*4*4)
        out = out.view(out.size(0), -1)
        # Linear function for readout
        out = self.fc1(out)

        return out



if __name__ == '__main__':
    # should get dataset data from argparse
    # dataset_name = 'CIFAR10'
    # train_loader, validate_loader, test_loader, len_train, len_test = get_data(dataset_name)
    train_dataset = dsets.MNIST(root='./data',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

    test_dataset = dsets.MNIST(root='./data',
                               train=False,
                               transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    # init_model = \
    # model, newmodel = get_model(dataset_identifier[dataset_name], load_state=True)
    model = CNNModel2()
    newmodel = copy.deepcopy(model)

    if torch.cuda.is_available():
        model.cuda()
    if torch.cuda.is_available():
        newmodel.cuda()

    makeItControlled(model, newmodel, controlAnyway=True, ControlType='linear', rnk_ratio=.5, verbose=False)

    # train
    iter = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    # num_epochs is the upper bound
    num_epoches = int(n_iters / (len(train_dataset) / batch_size))
    for epoch in range(num_epoches):
        for i, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
            else:
                images = Variable(images)
                labels = Variable(labels)

            optimizer.zero_grad()  # Clear gradients first
            outputs = model(images)  # Forward to get outputs
            loss = criterion(outputs, labels)  # Cross-entry loss function
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

                    outputs = model(images)
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

                # model = train(init_model, train_loader, validate_loader)
    # _test(model, test_loader)
