import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

sys.path.append('../')

from models import modules
from models import ResNet18_kv, ECELoss
from tricks import GradualWarmupScheduler, CrossEntropyLossMaybeSmooth, mixup_data, mixup_criterion
from torch.autograd import Variable
import time

# cutout=16
# class CutoutDefault(object):
#     """
#     Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
#     """
#     def __init__(self, length):
#         self.length = length

#     def __call__(self, img):
#         if self.length <= 0:
#             return img
#         h, w = img.size(1), img.size(2)
#         mask = np.ones((h, w), np.float32)
#         y = np.random.randint(h)
#         x = np.random.randint(w)

#         y1 = np.clip(y - self.length // 2, 0, h)
#         y2 = np.clip(y + self.length // 2, 0, h)
#         x1 = np.clip(x - self.length // 2, 0, w)
#         x2 = np.clip(x + self.length // 2, 0, w)

#         mask[y1: y2, x1: x2] = 0.
#         mask = torch.from_numpy(mask)
#         mask = mask.expand_as(img)
#         img *= mask
#         return img

nmembers = 4
batch_size = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = torchvision.datasets.CIFAR10(root='~/data', download=True, train=True, transform=transforms.Compose([
                   transforms.RandomCrop(32, padding=4),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                   # CutoutDefault(cutout)
               ]))

test_dataset = torchvision.datasets.CIFAR10(root='~/data', download=True, train=False, transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
               ]))

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=int(nmembers * batch_size), shuffle=True, drop_last=True, num_workers=12)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=12)

# criterion = CrossEntropyLossMaybeSmooth(0.2)
criterion = nn.CrossEntropyLoss()

def calculate_negative_correlation(ens_outputs):
    # ens_outputs of size [sub-batch-size, ensemble size, output dimension]
    avg = ens_outputs.mean(dim=1, keepdim=True).detach()
    dif = ens_outputs - avg
    divs = []
    for i in range(ens_outputs.shape[1]):
        temp_dif = torch.clone(dif)
        temp_dif[:,i,:] = 0
        temp_dif = temp_dif.sum(dim=1)
        divs.append(torch.einsum('xz, xz -> ', dif[:,i,:], temp_dif).unsqueeze(0))
    return torch.cat(divs)

# def get_thres_n_partis(module):
#     thres = []
#     partis = []
#     for k, v in module.named_modules():
#         if isinstance(v, modules.dense_res_bit_tree_baens) or isinstance(v, modules.Conv2d_res_bit_tree_baens):
#             thres.append(v.thres.item())
#             partis.append(v.npartitions)
#     return thres, partis

def get_thres_n_partis(module):
    # thres = []
    partis = []
    for k, v in module.named_modules():
        if isinstance(v, modules.dense_res_bit_tree_meanvar_freeze_fine_partition_wkernel_kv_gn_baens) \
        or isinstance(v, modules.Conv2d_res_bit_tree_meanvar_freeze_fine_partition_wkernel_kv_gn_baens):
            # thres.append([v.thres_mean.detach().cpu().numpy(), v.thres_var.detach().cpu().numpy()])
            partis.append(v.npartitions)
    return partis

def set_update_partition(module, BOOL):
    if isinstance(module, modules.dense_res_bit_tree_meanvar_freeze_fine_partition_wkernel_kv_gn_baens) \
    or isinstance(module, modules.Conv2d_res_bit_tree_meanvar_freeze_fine_partition_wkernel_kv_gn_baens):
        module.update_partition = BOOL
    if hasattr(module, 'children'):
        for submodule in module.children():
            set_update_partition(submodule, BOOL)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = []
    correct = 0
    total = 0

    # thres = []
    partis = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # inputs, targets = inputs.view(inputs.shape[0], -1).to(device), targets.to(device)
        # inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = inputs.cuda(), targets.cuda()
        # inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,
        #                                                1.0)
        # inputs, targets_a, targets_b = map(Variable, (inputs,
        #                                               targets_a, targets_b))

        # inputs = inputs.view(inputs.shape[0], -1)
        it_loss = 0

        optim_u.zero_grad()

        outputs = model(inputs)

        # loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam, False)
        loss = criterion(outputs, targets)

        loss.backward()
        it_loss += loss.item()

        optim_u.step()

        p_ = get_thres_n_partis(model)

        # thres.append(np.array(t_))
        partis.append(np.array(p_))

        train_loss.append(it_loss)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    # print('Thres: {}'.format(np.mean(thres, axis=0)))
    print('nPartitions: {}'.format(np.mean(partis, axis=0)))
    print('Train loss: %.3f' % np.mean(train_loss))
    print('Train accuracy: %.3f%%' % (correct * 100.0/total))

# Tuning
def tune(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = []
    correct = 0
    total = 0

    # thres = []
    partis = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # inputs, targets = inputs.view(inputs.shape[0], -1).to(device), targets.to(device)
        # inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = inputs.cuda(), targets.cuda()
        # inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,
        #                                                1.0)
        # inputs, targets_a, targets_b = map(Variable, (inputs,
        #                                               targets_a, targets_b))

        # inputs = inputs.view(inputs.shape[0], -1)
        it_loss = 0
        
        start = time.time()

        optim_v.zero_grad()
        optim_u.zero_grad()

        outputs = model(inputs)
        # nc = calculate_negative_correlation(outputs)

        # loss = criterion(outputs, targets) - nc.mean()
        loss = criterion(outputs, targets)
        # loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam, False)

        loss.backward()
        it_loss += loss.item()

        optim_v.step()
        optim_u.step()

        print(time.time()-start)
        exit()

        p_ = get_thres_n_partis(model)

        partis.append(np.array(p_))

        train_loss.append(it_loss)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    # print('Thres: {}'.format(np.mean(thres, axis=0)))
    print('UPDATEING THRESHOLD WITH NC')
    print('nPartitions: {}'.format(np.mean(partis, axis=0)))
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
            neg_corrs += calculate_negative_correlation(outputs).detach().cpu().numpy()

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


setups = []

times = 1e-4

split_learning_period = 1

# add trick
# card 1: split every round
# card 2: split every 120
# card 3: split just once

nmembers = 4
nbits = 3

setups = [[nmembers, nbits]]

special_suffix = '300epoch.sgd.bs2048.labelsmooth'
MODEL = 'data-cifar10.res18.kv-baens'
ckpt_file = 'ckpt.pytorch'

wherami_path = './checkpoint/' + MODEL + '/' + 'wherami.pytorch'

if os.path.exists(wherami_path):
    performance_dict = torch.load(wherami_path)
else:
    performance_dict = {}

for setup in setups:
    suffix = '-'.join(['nmember', str(setup[0]), 'bitwidth', str(setup[1])])

    result_path = './checkpoint/' + MODEL + '/' + suffix + '.' + special_suffix

    if not os.path.exists(result_path):
        os.mkdir(result_path)

    model = ResNet18_kv(*setup)
    model.cuda()

    parameters_u = []
    parameters_v = []
    parameters_d = []
    for k, v in model.named_parameters():
        if 'thres' in k:
            parameters_v.append(v)
        else:
            parameters_u.append(v)

    lr = 0.01
    wd = 1e-5
    momentum = 0.9

    # optim_u = torch.optim.Adam(parameters_u, lr=lr, weight_decay=wd)
    # optim_v = torch.optim.Adam(parameters_v, lr=times * lr, weight_decay=wd)

    optim_u = torch.optim.SGD(parameters_u, lr=lr, weight_decay=wd, momentum=momentum)
    optim_v = torch.optim.SGD(parameters_v, lr=times * lr, weight_decay=wd, momentum=momentum)

    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optim_u, milestones=[60, 100, 140, 180, 220, 260], gamma=0.1)
    # scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optim_v, milestones=[60, 100, 140, 180, 240, 280], gamma=0.1)

    scheduler11 = GradualWarmupScheduler(optim_u, multiplier=1., total_epoch=20, after_scheduler=scheduler1)
    # scheduler22 = GradualWarmupScheduler(optim_v, multiplier=1., total_epoch=20, after_scheduler=scheduler1)

    start_epoch = 0

    performance_dict[suffix] = {'acc': [], 'member_acc': [], 'loss': [], 'nc':[], 'ece': [], 'best_acc': 0, 'best_loss': 1e5, 'best_ece': 1e5}
    ece = ECELoss()

    # print('### Set temperature {} ###'.format(temp))
    # set_temp_res_bit_baens(model, temp)

    for epoch in range(start_epoch, start_epoch+300):

        torch.save(performance_dict, wherami_path)

        if epoch % split_learning_period == 0:
        # if epoch == 0:
            set_update_partition(model, True)
            model.tune = True
            tune(epoch)
            model.tune = False
            test(epoch)
        else:
            set_update_partition(model, False)
            model.tune = False
            train(epoch)
            test(epoch)

        scheduler11.step()
        # scheduler22.step()