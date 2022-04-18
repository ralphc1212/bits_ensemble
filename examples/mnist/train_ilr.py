import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

sys.path.append('../')
from models import ECELoss, set_temp_tcp_baens
from models import ILR

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataset = torchvision.datasets.mnist.MNIST(root='~/data', train=True, download=True, transform=transforms.Compose([
                   transforms.RandomCrop(28, padding=4),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
               ]))

test_dataset = torchvision.datasets.mnist.MNIST(root='~/data', train=False, download=True, transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
               ]))

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True, drop_last=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False)

criterion = nn.CrossEntropyLoss()

def calculate_negative_correlation(ens_outputs):
    # ens_outputs of size [sub-batch-size, ensemble size, output dimension]
    avg = ens_outputs.mean(dim=1, keepdim=True)
    dif = ens_outputs - avg
    divs = []

    for i in range(ens_outputs.shape[1]):
        temp_dif = torch.clone(dif)
        temp_dif[:,i,:] = 0
        temp_dif = temp_dif.sum(dim=1)
        divs.append(torch.einsum('xz, xz -> ', dif[:,i,:], temp_dif).item())
    return np.array(divs)

def train(epoch, member_index):
    global nmembers
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = []
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.view(inputs.shape[0], -1).to(device), targets.to(device)

        it_loss = 0

        optim_u.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        it_loss += loss.item()

        optim_u.step()

        train_loss.append(it_loss)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print('(member index: {}/ nmembers: {}) Train loss: {}'.format(member_index, nmembers, np.mean(train_loss)))
    print('(member index: {}/ nmembers: {}) Train accuracy: {}'.format(member_index, nmembers, (correct * 100.0/total)))

def test_member(epoch, member_index):

    global nmembers

    model.eval()
    test_loss = []
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.view(inputs.shape[0], -1).to(device), targets.to(device)

        it_loss = 0

        outputs = model(inputs)

        loss = criterion(outputs, targets)
        it_loss += loss.item()

        test_loss.append(it_loss)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print('(member index: {}/ nmembers: {}) Test loss: {}'.format(member_index, nmembers, np.mean(test_loss)))
    print('(member index: {}/ nmembers: {}) Test accuracy: {}'.format(member_index, nmembers, (correct * 100.0/total)))
    return correct * 100.0/total

setups = [[16]]

ckpt_file = 'ckpt.pytorch'

wherami_path = './checkpoint/logistic-regression/{}/performance_dict.pytorch'

if os.path.exists(wherami_path):
    performance_dict = torch.load(wherami_path)
else:
    performance_dict = {}

for setup in setups:
    nmembers = setup[0]
    suffix = '-'.join(['nmember', str(nmembers)])

    result_path = './checkpoint/logistic-regression/' + suffix

    if not os.path.exists(result_path):
        os.mkdir(result_path)

    lr = 0.001
    wd = 1e-5

    # scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optim_u, milestones=[30, 60, 75, 90], gamma=0.1)

    start_epoch = 0

    temp = 0.005
    ece = ECELoss()

    for model_index in range(nmembers):
        performance_dict[model_index] = 0.
        model = ILR(input_dim=784, output_dim=10).to(device)
        optim_u = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        for epoch in range(start_epoch, start_epoch+20):

            train(epoch, model_index)
            te_acc = test_member(epoch, model_index)

        torch.save(model, './checkpoint/logistic-regression/' + suffix + '/' + str(model_index) + '-' + ckpt_file)
