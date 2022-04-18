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

ece = ECELoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataset = torchvision.datasets.mnist.MNIST(root='~/data', train=True, transform=transforms.Compose([
                   transforms.RandomCrop(28, padding=4),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
               ]))

test_dataset = torchvision.datasets.mnist.MNIST(root='~/data', train=False, transform=transforms.Compose([
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

def test(ensemble):
    global perf_dict
    global suffix
    global ece
    global result_path
    test_loss = []
    correct = 0
    total = 0
    member_corrects = np.zeros(len(ensemble))

    ece_losses = []
    neg_corrs = np.array([0.] * len(ensemble))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.view(inputs.shape[0], -1).to(device), targets.to(device)

            outputs = []

            for model in ensemble:
                outputs.append(model(inputs))

            outputs = torch.stack(outputs, dim = 1)

            neg_corrs += calculate_negative_correlation(outputs)
            output = outputs.mean(dim=1)

            loss = criterion(output, targets)
            ece_losses.append(ece(output, targets).item())

            test_loss.append(loss.item())
            _, predicted = output.max(1)
            _, member_predicted = outputs.max(2)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            member_corrects += member_predicted.eq(targets.unsqueeze(1).expand_as(member_predicted)).sum(0).cpu().numpy()

    neg_corrs = neg_corrs / total
    ece_loss = np.mean(ece_losses)
    te_loss = np.mean(test_loss)
    acc = 100.*correct / total
    member_acc = 100.*member_corrects / total

    # print('te_loss: %.3f' % np.mean(test_loss))
    # print('ECE: %.3f' % np.mean(ece_loss))
    # print('test_acc: %.3f%%' % acc)
    # print('Avg negative correlation: {}'.format(np.mean(neg_corrs)))
    print('te_loss: {}, ECE: {}, test_acc: {}, ENC: {}'.format(np.mean(test_loss), np.mean(ece_loss), acc, np.mean(neg_corrs)))
    perf_dict['te_loss'].append(np.mean(test_loss))
    perf_dict['ece'].append(np.mean(np.mean(ece_loss)))
    perf_dict['te_acc'].append(acc)
    perf_dict['enc'].append(np.mean(neg_corrs))

    # print('Member accuracy: {}'.format(member_acc))
    # print('Negative correlation: {}'.format(neg_corrs))


def load_model(member_index):
    ckpt_path_name = './checkpoint/logistic-regression/' + suffix + '/test6' + str(member_index) + '-' + ckpt_file

    model = ILR(input_dim=784, output_dim=10)

    model.load_state_dict(torch.load(ckpt_path_name).state_dict())
    # model.load_state_dict(torch.load(ckpt_path_name))

    model = model.to(device)

    model.eval()

    return model

mode = 'fp'
nmembers = 16
suffix = '-'.join(['nmember', str(nmembers)])

ckpt_file = 'ckpt.pytorch'

ensemble = []

ensemble.append(load_model(0))

perf_dict = {'te_loss': [], 'ece': [], 'te_acc': [], 'enc': []}

for i in range(nmembers - 1):
    ensemble.append(load_model(i + 1))
    test(ensemble)

torch.save(perf_dict, './checkpoint/logistic-regression/' + suffix + '/test6.results')




