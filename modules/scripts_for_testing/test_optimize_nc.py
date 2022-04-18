import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

TEMPERATURE = 0.01
THRES = 0.9

def get_round():
    class round(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.round(input)
            
        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            return grad_input
    return round().apply

def df_lt(a, b, temp):
    # if a > b
    return torch.sigmoid((a - b) / temp)

def rbf_wmeans(x, means, sigma, eps):
    return torch.exp(- torch.abs(x - means + eps) / (2 * sigma ** 2))


train_dataset = torchvision.datasets.mnist.MNIST(root='./data', download=True, train=True, transform=transforms.Compose([
                   transforms.RandomCrop(28, padding=4),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
               ]))
test_dataset = torchvision.datasets.mnist.MNIST(root='./data', download=True, train=False, transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
               ]))

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True, num_workers=0)
tuningloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True, num_workers=0)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=True, num_workers=0)


class dense_res_bit_tree_meanvar_freeze_fine_partition_wkernel_baens(nn.Module):
    def __init__(self, N=5, D1=3, D2=2, bw=3, quantize=True):
        super(dense_res_bit_tree_meanvar_freeze_fine_partition_wkernel_baens, self).__init__()

        self.N = N
        self.D1 = D1
        self.D2 = D2
        D = int(D1 * D2)
        self.D = D
        self.U = nn.Parameter(torch.normal(0, 1, (N, D)), requires_grad=True)

        self.res_denos = nn.Parameter(torch.tensor([2**2-1, 2**2+1, 2**4+1, 2**8+1, 2**16+1]), requires_grad=False)

        self.round = get_round()
        self.maxBWPow = bw

        self.zeroD = nn.Parameter(torch.zeros(1, self.D), requires_grad=False)
        self.thres_mean = nn.Parameter(torch.tensor([-1.3] * D2), requires_grad=True)
        self.thres_var  = nn.Parameter(torch.tensor([-2.197]), requires_grad=True)
        self.transfer_matrix = None
        self.split_point = None
        self.update_partition = True
        self.eps = nn.Parameter(torch.tensor([10 ** (-16)]), requires_grad=False)

        self.rbf_mu = nn.Parameter(torch.tensor([0., 1., 2., 3.]).unsqueeze(dim=0), requires_grad=False)
        self.rbf_sigma = nn.Parameter(torch.tensor([0.1]), requires_grad=False)

        self.temperature = TEMPERATURE
        torch.nn.init.kaiming_normal_(self.U)

    def forward(self, x):
        # wQ = self.quant(self.U)
        # betas, _ = torch.max(self.U, dim=1)
        # alphas, _ = torch.min(self.U, dim=1)
        self.npartitions = [0] * (self.maxBWPow - 1)

        beta = torch.max(self.U)
        alpha = torch.min(self.U)
        # group_counts = [0] * (maxBWPow - 1)

        s = None
        for idx, deno in enumerate(self.res_denos[:self.maxBWPow]):
            if s is None:
                s = (beta - alpha) / deno
                vals = (s * self.round(self.U / s)).unsqueeze(0)
            else:
                s = s / deno
                res_errer = self.U - vals.sum(0)

                if self.update_partition:
                    sorted_err, idx_maps = torch.sort(res_errer, dim=0)
                    self.transfer_matrix = torch.nn.functional.one_hot(idx_maps, num_classes=self.N).float().detach()

                    # transfered_error = torch.einsum('ab, cba -> cb', res_errer, self.transfer_matrix)

                    delta_sorted_err = sorted_err[1:,:] - sorted_err[:-1,:]
                    delta_sorted_err = delta_sorted_err / torch.max(delta_sorted_err)
                    mean_split_point = df_lt(delta_sorted_err, torch.sigmoid(self.thres_mean.repeat_interleave(self.D1)), 0.01)
                    ### May need let the partitioning shared in a channel.
                    self.split_point = mean_split_point.detach()
                    self.npartitions[idx - 1] += torch.round(mean_split_point).mean(dim=1).sum().item()

                    # stt-round
                    round_split = torch.cat([self.zeroD, self.round(mean_split_point)])
                    # # soft
                    # round_split = torch.cat([self.zeroD, mean_split_point])
                else:
                    sorted_err = torch.einsum('ab, cba -> cb', res_errer, self.transfer_matrix)

                    self.npartitions[idx - 1] += torch.round(self.split_point).mean(dim=1).sum().item()

                    # stt-round
                    round_split = torch.cat([self.zeroD, self.round(self.split_point)])
                    # # soft
                    # round_split = torch.cat([self.zeroD, self.split_point])


                clustering_vec = rbf_wmeans(round_split.cumsum(dim=0).unsqueeze(2), self.rbf_mu, self.rbf_sigma, self.eps)
                inner_grouped = torch.einsum('abc, bc -> ab', clustering_vec, (sorted_err.unsqueeze(2) * clustering_vec).sum(dim=0) / (clustering_vec.sum(dim=0)+self.eps))

                grouped = torch.einsum('cb, cba -> ab', inner_grouped, self.transfer_matrix)
                quantized = s * self.round(grouped / s)

                vals = torch.cat([vals, quantized.unsqueeze(0)], dim=0)

        w = vals.sum(0).view(self.N, self.D1, self.D2)

        act = torch.einsum('bnd, ndl -> bnl', x, w)

        if torch.sum(torch.isnan(act)) != 0:
            print('act nan')
            print(act)
            exit()

        return act


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
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = []
    correct = 0
    total = 0
    

    if epoch % 10 == 0:
        tuning = True
        model.update_partition = True
    else:
        tuning = False
        model.update_partition = False

    # thres = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        it_loss = 0

        inputs = inputs.view(inputs.shape[0], -1)
        
        # if not tuning:
        #     assert inputs.shape[0] % 4 == 0
        #     inputs = inputs.view(int(inputs.shape[0]/4), 4, *inputs.shape[1:])
        # else:
        #     inputs = inputs.unsqueeze(1).expand(inputs.shape[0], 4, *inputs.shape[1:])

        inputs = inputs.view(int(inputs.shape[0]/4), 4, *inputs.shape[1:])

        if not tuning:
            optim_v.zero_grad()
        else:
            optim_u.zero_grad()

        outputs = model(inputs)

        outputs = outputs.reshape(int(outputs.shape[0] * outputs.shape[1]), *outputs.shape[2:])

        loss = criterion(outputs, targets)

        it_loss += loss.item()
        loss.backward()

        if not tuning:
            optim_v.step()
        else:
            optim_u.step()

        train_loss.append(it_loss)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print('Train loss: %.3f' % np.mean(train_loss))
    print('Train accuracy: %.3f%%' % (correct * 100.0/total))

criterion = nn.CrossEntropyLoss()

model = dense_res_bit_tree_meanvar_freeze_fine_partition_wkernel_baens(N=4, D1=784, D2=10, bw=3, quantize=True)

parameters_u = []
parameters_v = []
for k, v in model.named_parameters():
    if 'thres' in k:
        parameters_v.append(v)
    else:
        parameters_u.append(v)

optim_u = torch.optim.Adam(parameters_u, lr=0.01, weight_decay=1e-4)
optim_v = torch.optim.Adam(parameters_v, lr=0.01, weight_decay=1e-4)

for i in range(100):
    train(i)
