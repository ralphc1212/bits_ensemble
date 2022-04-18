import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class dense_binary_pca(nn.Module):
    def __init__(self, N=5, D1=3, D2=2, K=2, bitwidth=2):
        super(dense_binary_pca, self).__init__()

        self.N = N
        self.D1 = D1
        self.D2 = D2
        D = int(D1 * D2)
        self.bitwidth = bitwidth
        self.W_list = nn.ParameterList()
        self.Delta_list = nn.ParameterList()
        # self.scale = nn.Parameter(torch.normal(0, 1, (N, 1)), requires_grad=True)
        # self.bias = nn.Parameter(torch.normal(0, 1, (N, 1)), requires_grad=True)
        self.temp = 0.01
        self.TWO = nn.Parameter(torch.tensor(2.), requires_grad=False)

        for i in range(bitwidth):
            self.W_list.append(torch.nn.Parameter(torch.normal(0, 1, (N, D1, D2)), requires_grad=True))
            self.Delta_list.append(torch.nn.Parameter(torch.normal(0, 1, (D,)), requires_grad=True))

    def _relaxed_bern(self, p_logit):
        eps = 1e-7

        p = torch.sigmoid(p_logit)
        u_noise = torch.rand_like(p_logit)

        drop_prob = (torch.log(p + eps) -
                     torch.log(1 - p + eps) +
                     torch.log(u_noise + eps) -
                     torch.log(1 - u_noise + eps))

        drop_prob = torch.sigmoid(drop_prob / self.temp)

        random_tensor = 1 - drop_prob

        return random_tensor

    def forward(self, x):

        # w = 0
        #
        # for i in range(self.bitwidth):
        #     # Theta = self.W_list[i] + self.Delta_list[i]
        #     Theta = self.W_list[i]
        #
        #     # sample = self._relaxed_bern(Theta)
        #
        #     # sample = torch.sigmoid(Theta / self.temp)
        #     #
        #     # w += sample * (2. ** i)
        #
        #     w = Theta
        #     if torch.sum(torch.isnan(Theta)) != 0:
        #         print('1 Theta nan')
        #
        #     if torch.sum(torch.isnan(w)) != 0:
        #         print('2 w nan')
        #
        # # print(w)
        #
        # # weight = self.scale * (w - self.bias - torch.pow(self.TWO, self.bitwidth)/2.)
        # # weight = w - torch.pow(self.TWO, self.bitwidth)/2.
        #
        # weight = w
        # # print(self.scale)
        # # print(self.bias)
        # # print(torch.pow(self.TWO, self.bitwidth)/2.)
        # # print((weight < 0).sum())
        # # print((weight > 0).sum())
        # # print('---')
        #
        # w = weight.view(self.N, self.D1, self.D2)
        #
        # # act = torch.einsum('ij,bjk->bik', x, w).mean(dim=0)

        # act = torch.mm(x, self.W_list[0].view(self.D1, self.D2))
        act = torch.mm(x, self.W_list[0][0])

        if torch.sum(torch.isnan(act)) != 0:

            print('3 act nan')
            print(act)
            exit()

        return act


class mnist_binary_pca(nn.Module):
    def __init__(self):
        super(mnist_binary_pca, self).__init__()
        self.l1 = dense_binary_pca(N = 1, D1=784, D2=8192, K=5, bitwidth=1)
        self.l2 = dense_binary_pca(N = 1, D1=8192, D2=2048, K=5, bitwidth=1)
        self.l3 = dense_binary_pca(N = 1, D1=2048, D2=10, K=5, bitwidth=1)

        self.bn1 = nn.BatchNorm1d(8192)
        self.bn2 = nn.BatchNorm1d(2048)

    def forward(self, x):
        # x = F.relu(self.bn(self.l1(x)))
        x = F.relu(self.bn1(self.l1(x)))
        x = F.relu(self.bn2(self.l2(x)))
        # x = F.relu(self.l1(x))
        # x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


model = mnist_binary_pca().to(device)

stdict = model.state_dict()

print(stdict.keys())

parameters_w = []
# parameters_v = []
parameters_d = []
# for k in stdict.keys():
#     if 'W_list' in k or 'Delta_list' in k:
#         parameters_w.append(stdict[k])
#     # elif 'TWO' not in k:
#     #     parameters_d.append(stdict[k])

for k, v in model.named_parameters():
    if 'W_list' in k or 'Delta_list' in k:
        parameters_w.append(v)

optim_u = torch.optim.Adam(parameters_w, lr=1e-1)
# optim_d = torch.optim.Adam(parameters_d, lr=1e-1)

train_dataset = torchvision.datasets.mnist.MNIST(root='~/data', train=True, transform=transforms.Compose([
                   # transforms.RandomCrop(28, padding=4),
                   # transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
               ]))
test_dataset = torchvision.datasets.mnist.MNIST(root='~/data', train=False, transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
               ]))

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_u, 'min')
# scheduler3 = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_d, 'min')
criterion = nn.CrossEntropyLoss()

start_epoch = 0

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = []
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # inputs, targets = inputs.view(inputs.shape[0], -1).to(device), targets.to(device)
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.view(inputs.shape[0],-1)
        it_loss = 0

        optim_u.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        it_loss += loss.item()
        optim_u.step()


        # optim_d.zero_grad()
        # outputs = model(inputs)
        # loss = criterion(outputs, targets)
        # loss.backward()
        # it_loss += loss.item()
        #
        # optim_d.step()

        train_loss.append(it_loss)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print('Train loss: %.3f' % np.mean(train_loss))
    print('Train accuracy: %.3f%%' % (correct * 100.0/total))

def test(epoch):
    global best_acc
    global best_compression
    model.eval()
    test_loss = []
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.view(inputs.shape[0], -1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss.append(loss.item())
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Save checkpoint.
    acc = 100.*correct/total
    print('Test loss: %.3f' % np.mean(test_loss))
    print('Test accuracy: %.3f%%' % acc)

    scheduler1.step(loss)
    # scheduler3.step(loss)


for epoch in range(start_epoch, start_epoch+100):
    test(epoch)
    train(epoch)
