import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

def set_temp(module, temp):
    if isinstance(module, dense_binary_pca):
        module.temp = temp
    if hasattr(module, 'children'):
        for submodule in module.children():
            set_temp(submodule, temp)

class dense_binary_pca(nn.Module):
    def __init__(self, N=5, D1=3, D2=2, K=2, bitwidth=2):
        super(dense_binary_pca, self).__init__()

        self.N = N
        self.D1 = D1
        self.D2 = D2
        D = int(D1*D2)
        self.bitwidth = bitwidth
        self.U_list = nn.ParameterList()
        self.V_list = nn.ParameterList()
        self.Delta_list = nn.ParameterList()
        self.scale = nn.Parameter(torch.normal(0, 1, (N, 1)), requires_grad=True)
        self.bias = nn.Parameter(torch.normal(0, 1, (N, 1)), requires_grad=True)
        self.temp = 0.5
        self.TWO = nn.Parameter(torch.tensor([2.]), requires_grad=False)

        for i in range(bitwidth):
            self.U_list.append(torch.nn.Parameter(torch.normal(0, 1, (N, K)), requires_grad=True))
            self.V_list.append(torch.nn.Parameter(torch.normal(0, 1, (K, D)), requires_grad=True))
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

        w = 0

        for i in range(self.bitwidth):
            Theta = torch.mm(self.U_list[i], self.V_list[i]) + self.Delta_list[i]
            sample = self._relaxed_bern(Theta)
            # print(self.temp)

            w += sample * (2. ** i)

        w = self.scale * (w - self.bias - torch.pow(self.TWO, self.bitwidth))

        w = w.view(self.N, self.D1, self.D2)

        act = torch.einsum('ij,bjk->bik', x, w).mean(dim=0)

        if torch.sum(torch.isnan(act)) != 0:

            print('act nan')
            print(act)
            exit()

        return act


class mnist_binary_pca(nn.Module):
    def __init__(self, N=1, K=1, bitwidth=1):
        super(mnist_binary_pca, self).__init__()
        self.l1 = dense_binary_pca(N = N, D1=784, D2=400, K=K, bitwidth=bitwidth)
        self.l2 = dense_binary_pca(N = N, D1=400, D2=10, K=K, bitwidth=bitwidth)
        # self.bn = nn.BatchNorm1d(256)

    def forward(self, x):
        # x = F.relu(self.bn(self.l1(x)))
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

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

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

criterion = nn.CrossEntropyLoss()

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

        optim_v.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        it_loss += loss.item()

        optim_v.step()

        optim_d.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        it_loss += loss.item()

        optim_d.step()

        train_loss.append(it_loss/3)
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

    ece_losses = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.view(inputs.shape[0], -1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            ece_losses.append(ece(outputs, targets).item())

            test_loss.append(loss.item())
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    ece_loss = np.mean(ece_losses)
    te_loss = np.mean(test_loss)
    acc = 100.*correct/total

    performance_dict[suffix]['acc'].append(acc)
    performance_dict[suffix]['loss'].append(te_loss)
    performance_dict[suffix]['ece'].append(ece_loss)

    if acc > performance_dict[suffix]['best_acc']:
        performance_dict[suffix]['best_acc'] = acc
        print('### Saving ###')
        torch.save(model.state_dict(), result_path+'/'+ckpt_file)

    if te_loss < performance_dict[suffix]['best_loss']:
        performance_dict[suffix]['best_loss'] = te_loss

    if ece_loss < performance_dict[suffix]['best_ece']:
        performance_dict[suffix]['best_ece'] = ece_loss

    print('Test loss: %.3f' % np.mean(test_loss))
    print('Test accuracy: %.3f%%' % acc)
    print('Best accuracy: %.3f%%' % performance_dict[suffix]['best_acc'])

    scheduler1.step(loss)
    scheduler2.step(loss)
    scheduler3.step(loss)


setups = []

# write the setups

# for iter_ in range(10):
#     nmembers = iter_ + 1.
#     if nmembers != 10:
#         setups.append([int(nmembers), math.ceil(nmembers/2.), 1])
#         setups.append([1, 1, int(nmembers)])
#     else:
#         for i in range(8):
#             setups.append([int(nmembers), i+1, 1])
#         setups.append([1, 1, int(nmembers)])

for i in range(5):
    setups.append([int(5), i+1, 1])

setups.remove([5,3,1])

MODEL = 'data-mnist.model-700-400'
ckpt_file = 'ckpt.pytorch'

wherami_path = '../checkpoint/' + MODEL + '/' + 'wherami.pytorch'

if os.path.exists(wherami_path):
    performance_dict = torch.load(wherami_path)
else:
    performance_dict = {}

for setup in setups:
    suffix = '-'.join(['nmember', str(setup[0]), 'hidden', str(setup[1]), 'bitwidth', str(setup[2])])

    result_path = '../checkpoint/' + MODEL + '/' + suffix

    if not os.path.exists(result_path):
        os.mkdir(result_path)

    model = mnist_binary_pca(*setup).to(device)

    parameters_u = []
    parameters_v = []
    parameters_d = []
    for k, v in model.named_parameters():
        if 'U_list' in k:
            parameters_u.append(v)
        elif 'V_list' in k:
            parameters_v.append(v)
        else:
            parameters_d.append(v)

    optim_u = torch.optim.Adam(parameters_u, lr=0.1)
    optim_v = torch.optim.Adam(parameters_v, lr=0.1)
    optim_d = torch.optim.Adam(parameters_d, lr=0.1)

    scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_u, 'min')
    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_v, 'min')
    scheduler3 = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_d, 'min')

    start_epoch = 0

    temp = 0.5
    performance_dict[suffix] = {'acc': [], 'loss': [], 'ece': [], 'best_acc': 0, 'best_loss': 1e5, 'best_ece': 1e5}
    ece = ECELoss()

    for epoch in range(start_epoch, start_epoch+100):
        if (epoch + 1) % 30 == 0 and epoch < 80:
            temp = temp/10.
            print('### Set temperature {} ###'.format(temp))
            set_temp(model, temp)
            performance_dict[suffix]['best_acc'] = 0
            performance_dict[suffix]['best_loss'] = 1e5
            performance_dict[suffix]['best_ece'] = 1e5

        torch.save(performance_dict, wherami_path)
        test(epoch)
        train(epoch)

