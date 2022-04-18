import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

sys.path.append('../')

from tricks import GradualWarmupScheduler, CrossEntropyLossMaybeSmooth, mixup_data, mixup_criterion

from models import modules
from models import VGG_baens_quant, ECELoss

nmembers = 4
batch_size = 512
''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = torchvision.datasets.CIFAR10(root='~/data', download=True, train=True, transform=transforms.Compose([
                   transforms.RandomCrop(32, padding=4),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
               ]))

test_dataset = torchvision.datasets.CIFAR10(root='~/data', download=True, train=False, transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
               ]))

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=int(nmembers * batch_size), shuffle=True, drop_last=True, num_workers=16)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=16)

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

# Training
def train(epoch):
    global parameters_v
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = []
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()

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

    print('Train loss: %.3f' % np.mean(train_loss))
    print('Train accuracy: %.3f%%' % (correct * 100.0/total))

def test(epoch):
    global performance_dict
    global suffix
    global ece
    global result_path
    model.eval()
    test_loss = []
    correct = 0
    total = 0
    member_corrects = np.zeros(nmembers)

    ece_losses = []
    neg_corrs = np.array([0.] * nmembers)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()

            outputs = model(inputs)
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

    neg_corrs = neg_corrs/total
    ece_loss = np.mean(ece_losses)
    te_loss = np.mean(test_loss)
    acc = 100.*correct/total
    member_acc = 100.*member_corrects/total

    performance_dict[suffix]['acc'].append(acc)
    performance_dict[suffix]['member_acc'].append(member_acc)
    performance_dict[suffix]['loss'].append(te_loss)
    performance_dict[suffix]['ece'].append(ece_loss)
    performance_dict[suffix]['nc'].append(neg_corrs)

    if acc > performance_dict[suffix]['best_acc']:
        performance_dict[suffix]['best_acc'] = acc
        print('### Saving {} ###'.format(result_path+'/'+ckpt_file))
        torch.save(model.state_dict(), result_path+'/'+ckpt_file)

    if te_loss < performance_dict[suffix]['best_loss']:
        performance_dict[suffix]['best_loss'] = te_loss

    if ece_loss < performance_dict[suffix]['best_ece']:
        performance_dict[suffix]['best_ece'] = ece_loss

    print('Test loss: %.3f' % np.mean(test_loss))
    print('Test accuracy: %.3f%%' % acc)
    print('Member accuracy: {}'.format(member_acc))
    print('Negative correlation: {}'.format(neg_corrs))
    print('Best accuracy: %.3f%%' % performance_dict[suffix]['best_acc'])

# 13008. submission/checkpoint/data-cifar10.vgg11.baens/nmember-4.280epoch/ckpt.pytorch.1
# 13010.


# 14786. 

setups = []

times = 10

nmembers = 4

setups = [[nmembers]]

special_suffix = '320epoch'
MODEL = 'data-cifar10.vgg11.baens_quant'
ckpt_file = 'ckpt.pytorch'

wherami_path = './checkpoint/' + MODEL + '/' + 'wherami.pytorch'

if os.path.exists(wherami_path):
    performance_dict = torch.load(wherami_path)
else:
    performance_dict = {}

for setup in setups:
    suffix = '-'.join(['nmember', str(setup[0])])

    result_path = './checkpoint/' + MODEL + '/' + suffix + '.' + special_suffix

    if not os.path.exists(result_path):
        os.mkdir(result_path)

    model = VGG_baens_quant(10, True, '11', *setup)
    model.cuda()

    lr = 0.01
    wd = 1e-5
    optim_u = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)

    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optim_u, milestones=[60, 100, 140, 180], gamma=0.1)
    scheduler11 = GradualWarmupScheduler(optim_u, multiplier=1., total_epoch=20, after_scheduler=scheduler1)

    start_epoch = 0

    performance_dict[suffix] = {'acc': [], 'member_acc': [], 'loss': [], 'nc':[], 'ece': [], 'best_acc': 0, 'best_loss': 1e5, 'best_ece': 1e5}
    ece = ECELoss()

    for epoch in range(start_epoch, start_epoch+220):

        torch.save(performance_dict, wherami_path)
        train(epoch)
        test(epoch)

        scheduler11.step()
