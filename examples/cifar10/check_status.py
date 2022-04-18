import torch

MODEL = 'data-cifar10.vgg11.res-bit-tree-ensemble-baens'

suffix = 'nmember-6-bitwidth-3'

a = torch.load('./checkpoint/' + MODEL + '/' + 'wherami.pytorch')
print('check ... accuracy ...')
print(a[suffix]['acc'])
print('check ... best ...')
print(a[suffix]['best_acc'])