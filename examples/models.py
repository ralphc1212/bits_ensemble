import sys
sys.path.append('../../')
import torch
import torch.nn as nn
import torch.nn.functional as F
import modules
import time

class ECELoss(nn.Module):
    def __init__(self, n_bins=15):
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

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim, mode, N, bitwidth):
        super(LogisticRegression, self).__init__()
        self.N = N
        self.mode = mode
        latent_dim = 128
        if mode == 'vanilla':
            self.ens_list = nn.ModuleList()
            for i in range(N):
                self.ens_list.append(nn.Sequential(*[torch.nn.Linear(input_dim, latent_dim),
                    nn.ReLU(),torch.nn.Linear(latent_dim, output_dim)]))
        elif mode == 'baens':
            # batchensemble
            self.ens = nn.Sequential(*[modules.dense_baens(N, input_dim, latent_dim),
                nn.ReLU(),modules.dense_baens(N, latent_dim, output_dim)])
        elif mode == 'quant-baens':
            # independent member quantization
            self.ens = nn.Sequential(*[modules.dense_baens_quant(N, input_dim, latent_dim),
                nn.ReLU(),modules.dense_baens_quant(N, latent_dim, output_dim)])
        elif mode == 'kv-baens':
            # our quantization scheme
            self.ens = nn.Sequential(*[modules.dense_res_bit_tree_meanvar_freeze_fine_partition_wkernel_kv_baens(N, input_dim, latent_dim),
                nn.ReLU(),modules.dense_res_bit_tree_meanvar_freeze_fine_partition_wkernel_kv_baens(N, latent_dim, output_dim)])

    def forward(self, x):
        if self.mode == 'vanilla':
            if self.training:
                outputs = 0
                for member in self.ens_list:
                    outputs += member(x)
                return outputs / self.N
            else:
                outputs = []
                for member in self.ens_list:
                    outputs.append(member(x))

                return torch.stack(outputs).permute(1,0,2)
        else:
            if self.training:
                assert x.shape[0] % self.N == 0
                x = x.view(int(x.shape[0]/self.N), self.N, *x.shape[1:])
            else:
                x = x.unsqueeze(1).expand(x.shape[0], self.N, *x.shape[1:])

            outputs = self.ens(x)

            if self.training:
                outputs = outputs.reshape(int(outputs.shape[0] * outputs.shape[1]), *outputs.shape[2:])
            else:
                outputs = outputs
            return outputs

class ILR(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ILR, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, 128)
        self.linear2 = torch.nn.Linear(128, output_dim)

    def forward(self, x, member_index=None):
        outputs = self.linear2(F.relu(self.linear1(x)))
        return outputs

def set_temp_b(module, temp):
    if isinstance(module, modules.dense_binary_pca_baens):
        module.temp = temp
    if hasattr(module, 'children'):
        for submodule in module.children():
            set_temp_b(submodule, temp)

class mnist_binary_pca_741(nn.Module):
    def __init__(self, N=1, K=1, bitwidth=1, batchnorm=False):
        super(mnist_binary_pca_741, self).__init__()
        self.l1 = modules.dense_binary_pca(N = N, D1=784, D2=400, K=K, bitwidth=bitwidth)
        self.l2 = modules.dense_binary_pca(N = N, D1=400, D2=10, K=K, bitwidth=bitwidth)
        self.batchnorm = batchnorm

        if batchnorm:
            self.bn1 = nn.BatchNorm1d(400)
        # self.bn2 = nn.BatchNorm1d(400)

    def forward(self, x):
        if self.batchnorm:
            x = F.relu(self.bn1(self.l1(x)))
        else:
            x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

class mnist_binary_pca_baens_741(nn.Module):
    def __init__(self, N=1, K=1, bitwidth=1, batchnorm=False):
        super(mnist_binary_pca_baens_741, self).__init__()
        self.N = N
        self.l1 = modules.dense_binary_pca_baens(N = N, D1=784, D2=400, K=K, bitwidth=bitwidth)
        self.l2 = modules.dense_binary_pca_baens(N = N, D1=400, D2=10, K=K, bitwidth=bitwidth)
        self.batchnorm = batchnorm

        if batchnorm:
            self.bn1 = nn.BatchNorm1d(400)
        # self.bn2 = nn.BatchNorm1d(400)

    def forward(self, x):
        if self.training:
            assert x.shape[0] % self.N == 0
            x = x.view(self.N, int(x.shape[0]/self.N), *x.shape[1:])
        else:
            x = x.unsqueeze(0).expand(self.N, x.shape[0], *x.shape[1:])

        if self.batchnorm:
            x = F.relu(self.bn1(self.l1(x)))
        else:
            x = F.relu(self.l1(x))
        x = self.l2(x)

        if self.training:
            x = x.view(int(x.shape[0] * x.shape[1]), *x.shape[2:])

        return x

class mnist_binary_pca_7441(nn.Module):
    def __init__(self, N=1, K=1, bitwidth=1, batchnorm=False):
        super(mnist_binary_pca_7441, self).__init__()
        self.l1 = modules.dense_binary_pca(N = N, D1=784, D2=400, K=K, bitwidth=bitwidth)
        self.l2 = modules.dense_binary_pca(N = N, D1=400, D2=400, K=K, bitwidth=bitwidth)
        self.l3 = modules.dense_binary_pca(N = N, D1=400, D2=10, K=K, bitwidth=bitwidth)
        self.batchnorm = batchnorm
        if batchnorm:
            self.bn1 = nn.BatchNorm1d(400)
            self.bn2 = nn.BatchNorm1d(400)

    def forward(self, x):
        if self.batchnorm:
            x = F.relu(self.bn1(self.l1(x)))
            x = F.relu(self.bn2(self.l2(x)))
        else:
            x = F.relu(self.l1(x))
            x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


def set_temp_tcp(module, temp):
    if isinstance(module, modules.dense_decompo_cp):
        module.temp = temp
    if hasattr(module, 'children'):
        for submodule in module.children():
            set_temp_tcp(submodule, temp)

class mnist_decompo_cp_741(nn.Module):
    def __init__(self, N=1, bitwidth=1, K=1, batchnorm=False):
        super(mnist_decompo_cp_741, self).__init__()
        self.l1 = modules.dense_decompo_cp(N = N, D1=784, D2=400, bw=bitwidth, K=K)
        self.l2 = modules.dense_decompo_cp(N = N, D1=400, D2=10, bw=bitwidth, K=K)

        self.batchnorm = batchnorm
        if batchnorm:
            self.bn1 = nn.BatchNorm1d(400)
        # self.bn2 = nn.BatchNorm1d(400)

    def forward(self, x):
        if self.batchnorm:
            x = F.relu(self.bn1(self.l1(x)))
        else:
            x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

class mnist_decompo_cp_7441(nn.Module):
    def __init__(self, N=1, bitwidth=1, K=1, batchnorm=False):
        super(mnist_decompo_cp_7441, self).__init__()
        self.l1 = modules.dense_decompo_cp(N = N, D1=784, D2=400, bw=bitwidth, K=K)
        self.l2 = modules.dense_decompo_cp(N = N, D1=400, D2=400, bw=bitwidth, K=K)
        self.l3 = modules.dense_decompo_cp(N = N, D1=400, D2=10, bw=bitwidth, K=K)

        self.batchnorm = batchnorm
        if batchnorm:
            self.bn1 = nn.BatchNorm1d(400)
            self.bn2 = nn.BatchNorm1d(400)

    def forward(self, x):
        if self.batchnorm:
            x = F.relu(self.bn1(self.l1(x)))
            x = F.relu(self.bn2(self.l2(x)))
        else:
            x = F.relu(self.l1(x))
            x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


def set_temp_ttucker(module, temp):
    if isinstance(module, modules.dense_decompo_tucker):
        module.temp = temp
    if hasattr(module, 'children'):
        for submodule in module.children():
            set_temp_tcp(submodule, temp)


class mnist_decompo_tucker_741(nn.Module):
    def __init__(self, N=1, bitwidth=1, KN=1, Kbw=1, batchnorm=False):
        super(mnist_decompo_tucker_741, self).__init__()
        self.l1 = modules.dense_decompo_tucker(N = N, D1=784, D2=400, bw=bitwidth, KN=KN, KD=16, Kbw=Kbw)
        self.l2 = modules.dense_decompo_tucker(N = N, D1=400, D2=10, bw=bitwidth, KN=KN, KD=8, Kbw=Kbw)

        self.batchnorm = batchnorm
        if batchnorm:
            self.bn1 = nn.BatchNorm1d(400)
        # self.bn2 = nn.BatchNorm1d(400)

    def forward(self, x):
        if self.batchnorm:
            x = F.relu(self.bn1(self.l1(x)))
        else:
            x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

class mnist_decompo_tucker_7441(nn.Module):
    def __init__(self, N=1, bitwidth=1, KN=1, Kbw=1, batchnorm=False):
        super(mnist_decompo_tucker_7441, self).__init__()
        self.l1 = modules.dense_decompo_tucker(N = N, D1=784, D2=400, bw=bitwidth, KN=KN, KD=16, Kbw=Kbw)
        self.l2 = modules.dense_decompo_tucker(N = N, D1=400, D2=400, bw=bitwidth, KN=KN, KD=8, Kbw=Kbw)
        self.l3 = modules.dense_decompo_tucker(N = N, D1=400, D2=10, bw=bitwidth, KN=KN, KD=8, Kbw=Kbw)

        self.batchnorm = batchnorm
        if batchnorm:
            self.bn1 = nn.BatchNorm1d(400)
            self.bn2 = nn.BatchNorm1d(400)

    def forward(self, x):
        if self.batchnorm:
            x = F.relu(self.bn1(self.l1(x)))
            x = F.relu(self.bn2(self.l2(x)))
        else:
            x = F.relu(self.l1(x))
            x = F.relu(self.l2(x))
        x = self.l3(x)
        return x



def set_temp_tcp_baens(module, temp):
    if isinstance(module, modules.dense_decompo_cp_baens):
        module.temp = temp
    if hasattr(module, 'children'):
        for submodule in module.children():
            set_temp_tcp_baens(submodule, temp)

class mnist_decompo_cp_baens_741(nn.Module):
    def __init__(self, N=1, bitwidth=1, K=1, batchnorm=False):
        super(mnist_decompo_cp_baens_741, self).__init__()
        self.N = N
        self.l1 = modules.dense_decompo_cp_baens(N = N, D1=784, D2=400, bw=bitwidth, K=K)
        self.l2 = modules.dense_decompo_cp_baens(N = N, D1=400, D2=10, bw=bitwidth, K=K)

        self.batchnorm = batchnorm
        if batchnorm:
            self.bn1 = nn.BatchNorm1d(400)
        # self.bn2 = nn.BatchNorm1d(400)

    def forward(self, x):
        if self.training:
            assert x.shape[0] % self.N == 0
            x = x.view(self.N, int(x.shape[0]/self.N), *x.shape[1:])
        else:
            x = x.unsqueeze(0).expand(self.N, x.shape[0], *x.shape[1:])

        if self.batchnorm:
            x = F.relu(self.bn1(self.l1(x)))
        else:
            x = F.relu(self.l1(x))
        x = self.l2(x)

        if self.training:
            x = x.view(int(x.shape[0] * x.shape[1]), *x.shape[2:])

        return x

class mnist_decompo_cp_baens_7441(nn.Module):
    def __init__(self, N=1, bitwidth=1, K=1, batchnorm=False):
        super(mnist_decompo_cp_baens_7441, self).__init__()
        self.N = N
        self.l1 = modules.dense_decompo_cp_baens(N = N, D1=784, D2=400, bw=bitwidth, K=K, quantize=True)
        self.l2 = modules.dense_decompo_cp_baens(N = N, D1=400, D2=400, bw=bitwidth, K=K, quantize=True)
        self.l3 = modules.dense_decompo_cp_baens(N = N, D1=400, D2=10, bw=bitwidth, K=K, quantize=True)

        self.batchnorm = batchnorm
        if batchnorm:
            self.bn1 = nn.BatchNorm1d(400)
            self.bn2 = nn.BatchNorm1d(400)

    def forward(self, x):
        if self.training:
            assert x.shape[0] % self.N == 0
            x = x.view(self.N, int(x.shape[0]/self.N), *x.shape[1:])
        else:
            x = x.unsqueeze(0).expand(self.N, x.shape[0], *x.shape[1:])

        if self.batchnorm:
            x = self.l1(x)
            x = F.relu(self.bn1(x.view(int(x.shape[0] * x.shape[1]), *x.shape[2:])))
            x = x.view(self.N, int(x.shape[0]/self.N), *x.shape[1:])
            x = self.l2(x)
            x = F.relu(self.bn2(x.view(int(x.shape[0] * x.shape[1]), *x.shape[2:])))
            x = self.l3(x.view(self.N, int(x.shape[0]/self.N), *x.shape[1:]))

        else:
            x = F.relu(self.l1(x))
            x = F.relu(self.l2(x))

            x = self.l3(x)

        if self.training:
            x = x.view(int(x.shape[0] * x.shape[1]), *x.shape[2:])

        return x


def set_temp_dbcp_baens(module, temp):
    if isinstance(module, modules.dense_decompo_db_baens):
        module.temp = temp
    if hasattr(module, 'children'):
        for submodule in module.children():
            set_temp_dbcp_baens(submodule, temp)

class mnist_decompo_dbcp_baens_741(nn.Module):
    def __init__(self, N=1, bitwidth=1, K=1, batchnorm=False):
        super(mnist_decompo_dbcp_baens_741, self).__init__()
        self.N = N
        self.l1 = modules.dense_decompo_db_baens(N = N, D1=784, D2=400, bw=bitwidth, K=K)
        self.l2 = modules.dense_decompo_db_baens(N = N, D1=400, D2=10, bw=bitwidth, K=K)

        self.batchnorm = batchnorm
        if batchnorm:
            self.bn1 = nn.BatchNorm1d(400)
        # self.bn2 = nn.BatchNorm1d(400)

    def forward(self, x):
        if self.training:
            assert x.shape[0] % self.N == 0
            x = x.view(self.N, int(x.shape[0]/self.N), *x.shape[1:])
        else:
            x = x.unsqueeze(0).expand(self.N, x.shape[0], *x.shape[1:])

        if self.batchnorm:
            x = F.relu(self.bn1(self.l1(x)))
        else:
            x = F.relu(self.l1(x))
        x = self.l2(x)

        if self.training:
            x = x.view(int(x.shape[0] * x.shape[1]), *x.shape[2:])

        return x


def set_temp_tcp(module, temp):
    if isinstance(module, modules.dense_decompo_cp):
        module.temp = temp
    if hasattr(module, 'children'):
        for submodule in module.children():
            set_temp_tcp(submodule, temp)

class mnist_conv_decompo_cp_baens_7441(nn.Module):
    def __init__(self, N=1, bitwidth=1, K=1, batchnorm=False):
        super(mnist_conv_decompo_cp_baens_7441, self).__init__()
        self.N = N
        self.conv1 = modules.Conv2d_decompo_cp_baens(1, 20, 5, N = N, bw=bitwidth, K=K, bias=True, quantize=True)
        self.conv2 = modules.Conv2d_decompo_cp_baens(20, 50, 5, N = N, bw=bitwidth, K=K, bias=True, quantize=True)

        self.l1 = modules.dense_decompo_cp_baens(N = N, D1=50*4*4, D2=400, bw=bitwidth, K=K, quantize=True)
        self.l2 = modules.dense_decompo_cp_baens(N = N, D1=400, D2=10, bw=bitwidth, K=K, quantize=True)

        self.batchnorm = batchnorm
        if batchnorm:
            self.bn1 = nn.BatchNorm2d(20)
            self.bn2 = nn.BatchNorm2d(50)
            self.bn3 = nn.BatchNorm1d(400)
        # self._init_weights()


    def _init_weights(self):
        for layer in self.children():
            if hasattr(layer, 'weight'): nn.init.xavier_uniform(layer.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        if self.training:
            assert x.shape[0] % self.N == 0
            x = x.view(int(x.shape[0]/self.N), int(self.N * x.shape[1]), *x.shape[2:])
        else:
            x = x.unsqueeze(1).expand(x.shape[0], self.N, *x.shape[1:])
            x = x.view(x.shape[0], int(x.shape[1] * x.shape[2]), *x.shape[3:])

        if self.batchnorm:
            x = self.conv1(x)
            x = F.relu(self.bn1(F.max_pool2d(x.view(int(x.shape[0] * self.N), int(x.shape[1] / self.N), *x.shape[2:]),2)))
            x = x.view(int(x.shape[0]/self.N), int(self.N * x.shape[1]), *x.shape[2:])
            x = F.max_pool2d(self.conv2(x), 2)
            x = F.relu(self.bn2(x.view(int(x.shape[0] * self.N), int(x.shape[1] / self.N), *x.shape[2:])))

            x = self.l1(x.view(int(x.shape[0]/self.N), self.N, -1))
            x = F.relu(self.bn3(x.reshape(int(x.shape[0] * x.shape[1]), *x.shape[2:])))
            x = self.l2(x.view(int(x.shape[0]/self.N), self.N, *x.shape[1:]))

        else:
            x = self.conv1(x)
            x = F.relu(F.max_pool2d(x, 2))
            x = self.conv2(x)
            x = F.relu(F.max_pool2d(x, 2))

            x = self.l1(x.view(x.shape[0], self.N, -1))
            x = F.relu(x)
            x = self.l2(x)
            # print(x.shape)

        if self.training:
            x = x.reshape(int(x.shape[0] * x.shape[1]), *x.shape[2:])

        return x


class mnist_conv_decompo_nbcp_baens(nn.Module):
    def __init__(self, N=1, bitwidth=1, K=1, batchnorm=False):
        super(mnist_conv_decompo_nbcp_baens, self).__init__()
        self.N = N
        self.conv1 = modules.Conv2d_decompo_nbcp_baens(1, 20, 5, N = N, bw=bitwidth, K=K, bias=True, quantize=True, first=True)
        self.conv2 = modules.Conv2d_decompo_nbcp_baens(20, 50, 5, N = N, bw=bitwidth, K=K, bias=True, quantize=True, first=False)

        self.l1 = modules.dense_decompo_nbcp_baens(N = N, D1=50*4*4, D2=400, bw=bitwidth, K=K, quantize=True)
        self.l2 = modules.dense_decompo_nbcp_baens(N = N, D1=400, D2=10, bw=bitwidth, K=K, quantize=True)

        self.batchnorm = batchnorm
        if batchnorm:
            self.bn1 = nn.BatchNorm2d(20)
            self.bn2 = nn.BatchNorm2d(50)
            self.bn3 = nn.BatchNorm1d(400)
        # self._init_weights()

    def _init_weights(self):
        for layer in self.children():
            if hasattr(layer, 'weight'): nn.init.xavier_uniform(layer.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        if self.training:
            assert x.shape[0] % self.N == 0
            x = x.view(int(x.shape[0]/self.N), int(self.N * x.shape[1]), *x.shape[2:])
        else:
            x = x.unsqueeze(1).expand(x.shape[0], self.N, *x.shape[1:])
            x = x.view(x.shape[0], int(x.shape[1] * x.shape[2]), *x.shape[3:])


        if self.batchnorm:
            x = F.relu(self.bn1(F.max_pool2d(self.conv1(x), 2)))
            x = F.max_pool2d(self.conv2(x), 2)
            x = F.relu(self.bn2(x))

            x = self.l1(x.view(int(x.shape[0]/self.N), self.N, -1))
            x = F.relu(self.bn3(x.reshape(int(x.shape[0] * x.shape[1]), *x.shape[2:])))
            x = self.l2(x.view(int(x.shape[0]/self.N), self.N, *x.shape[1:]))
        else:
            x = self.conv1(x)
            x = F.relu(F.max_pool2d(x, 2))
            x = self.conv2(x)
            x = F.relu(F.max_pool2d(x, 2))

            x = self.l1(x.view(x.shape[0], self.N, -1))
            x = F.relu(x)
            x = self.l2(x)
            # print(x.shape)

        if self.training:
            x = x.reshape(int(x.shape[0] * x.shape[1]), *x.shape[2:])

        return x

def set_temp_nbcp_baens(module, temp):
    if isinstance(module, modules.dense_decompo_nbcp_baens) or isinstance(module, modules.Conv2d_decompo_nbcp_baens):
        module.temp = temp
    if hasattr(module, 'children'):
        for submodule in module.children():
            set_temp_nbcp_baens(submodule, temp)


class mnist_res_bit_ensemble_baens_741(nn.Module):
    def __init__(self, N=1, bitwidth=1, batchnorm=False):
        super(mnist_res_bit_ensemble_baens_741, self).__init__()
        self.N = N
        self.l1 = modules.dense_res_bit_baens(N = N, D1=784, D2=400, bw=bitwidth)
        self.l2 = modules.dense_res_bit_baens(N = N, D1=400, D2=10, bw=bitwidth)

        self.batchnorm = batchnorm
        if batchnorm:
            self.bn1 = nn.BatchNorm1d(400)
        # self.bn2 = nn.BatchNorm1d(400)

    def forward(self, x):
        if self.training:
            assert x.shape[0] % self.N == 0
            x = x.view(int(x.shape[0]/self.N), self.N, *x.shape[1:])
        else:
            x = x.unsqueeze(1).expand(x.shape[0], self.N, *x.shape[1:])

        if self.batchnorm:
            x = F.relu(self.bn1(self.l1(x)))
        else:
            x = F.relu(self.l1(x))
        x = self.l2(x)

        if self.training:
            x = x.reshape(int(x.shape[0] * x.shape[1]), *x.shape[2:])

        return x

class mnist_shared_res_bit_ensemble_baens_741(nn.Module):
    def __init__(self, N=1, bitwidth=1, batchnorm=False):
        super(mnist_shared_res_bit_ensemble_baens_741, self).__init__()
        self.N = N
        self.l1 = modules.dense_shared_res_bit_baens(N = N, D1=784, D2=400, bw=bitwidth)
        self.l2 = modules.dense_shared_res_bit_baens(N = N, D1=400, D2=10, bw=bitwidth)

        self.batchnorm = batchnorm
        if batchnorm:
            self.bn1 = nn.BatchNorm1d(400)
        # self.bn2 = nn.BatchNorm1d(400)

    def forward(self, x):
        if self.training:
            assert x.shape[0] % self.N == 0
            x = x.view(int(x.shape[0]/self.N), self.N, *x.shape[1:])
        else:
            x = x.unsqueeze(1).expand(x.shape[0], self.N, *x.shape[1:])

        if self.batchnorm:
            x = F.relu(self.bn1(self.l1(x)))
        else:
            x = F.relu(self.l1(x))
        x = self.l2(x)

        if self.training:
            x = x.reshape(int(x.shape[0] * x.shape[1]), *x.shape[2:])

        return x

class mnist_res_bit_tree_ensemble_baens_741(nn.Module):
    def __init__(self, N=1, bitwidth=1, batchnorm=False):
        super(mnist_res_bit_tree_ensemble_baens_741, self).__init__()
        self.N = N
        self.l1 = modules.dense_res_bit_tree_baens(N = N, D1=784, D2=400, bw=bitwidth)
        self.l2 = modules.dense_res_bit_tree_baens(N = N, D1=400, D2=10, bw=bitwidth)

        self.batchnorm = batchnorm
        if batchnorm:
            self.bn1 = nn.BatchNorm1d(400)
        # self.bn2 = nn.BatchNorm1d(400)

    def forward(self, x):
        if self.training:
            assert x.shape[0] % self.N == 0
            x = x.view(int(x.shape[0]/self.N), self.N, *x.shape[1:])
        else:
            x = x.unsqueeze(1).expand(x.shape[0], self.N, *x.shape[1:])

        if self.batchnorm:
            x = F.relu(self.bn1(self.l1(x)))
        else:
            x = F.relu(self.l1(x))
        x = self.l2(x)

        if self.training:
            x = x.reshape(int(x.shape[0] * x.shape[1]), *x.shape[2:])

        return x

class mnist_conv_res_bit_ensemble_baens(nn.Module):
    def __init__(self, N=1, bitwidth=1, batchnorm=False):
        super(mnist_conv_res_bit_ensemble_baens, self).__init__()
        self.N = N
        self.conv1 = modules.Conv2d_res_bit_baens(1, 20, 5, N = N, bw=bitwidth, bias=True, quantize=True, first=True)
        self.conv2 = modules.Conv2d_res_bit_baens(20, 50, 5, N = N, bw=bitwidth, bias=True, quantize=True, first=False)

        self.l1 = modules.dense_res_bit_baens(N = N, D1=50*4*4, D2=400, bw=bitwidth, quantize=True)
        self.l2 = modules.dense_res_bit_baens(N = N, D1=400, D2=10, bw=bitwidth, quantize=True)

        self.batchnorm = batchnorm
        if batchnorm:
            self.bn1 = nn.BatchNorm2d(20)
            self.bn2 = nn.BatchNorm2d(50)
            self.bn3 = nn.BatchNorm1d(400)
        # self._init_weights()

    def _init_weights(self):
        for layer in self.children():
            if hasattr(layer, 'weight'): nn.init.xavier_uniform(layer.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        if self.training:
            assert x.shape[0] % self.N == 0
            x = x.view(int(x.shape[0]/self.N), int(self.N * x.shape[1]), *x.shape[2:])
        else:
            x = x.unsqueeze(1).expand(x.shape[0], self.N, *x.shape[1:])
            x = x.view(x.shape[0], int(x.shape[1] * x.shape[2]), *x.shape[3:])

        if self.batchnorm:
            x = F.relu(self.bn1(F.max_pool2d(self.conv1(x), 2)))
            x = F.max_pool2d(self.conv2(x), 2)
            x = F.relu(self.bn2(x))

            x = self.l1(x.view(int(x.shape[0]/self.N), self.N, -1))
            x = F.relu(self.bn3(x.reshape(int(x.shape[0] * x.shape[1]), *x.shape[2:])))
            x = self.l2(x.view(int(x.shape[0]/self.N), self.N, *x.shape[1:]))
        else:
            x = self.conv1(x)
            x = F.relu(F.max_pool2d(x, 2))
            x = self.conv2(x)
            x = F.relu(F.max_pool2d(x, 2))

            x = self.l1(x.view(x.shape[0], self.N, -1))
            x = F.relu(x)
            x = self.l2(x)
            # print(x.shape)

        if self.training:
            x = x.reshape(int(x.shape[0] * x.shape[1]), *x.shape[2:])

        return x


class mnist_conv_shared_res_bit_ensemble_baens(nn.Module):
    def __init__(self, N=1, bitwidth=1, batchnorm=False):
        super(mnist_conv_shared_res_bit_ensemble_baens, self).__init__()
        self.N = N
        self.conv1 = modules.Conv2d_shared_res_bit_baens(1, 20, 5, N = N, bw=bitwidth, bias=True, quantize=True, first=True)
        self.conv2 = modules.Conv2d_shared_res_bit_baens(20, 50, 5, N = N, bw=bitwidth, bias=True, quantize=True, first=False)

        self.l1 = modules.dense_shared_res_bit_baens(N = N, D1=50*4*4, D2=400, bw=bitwidth, quantize=True)
        self.l2 = modules.dense_shared_res_bit_baens(N = N, D1=400, D2=10, bw=bitwidth, quantize=True)

        self.batchnorm = batchnorm
        if batchnorm:
            self.bn1 = nn.BatchNorm2d(20)
            self.bn2 = nn.BatchNorm2d(50)
            self.bn3 = nn.BatchNorm1d(400)
        # self._init_weights()

    def _init_weights(self):
        for layer in self.children():
            if hasattr(layer, 'weight'): nn.init.xavier_uniform(layer.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        if self.training:
            assert x.shape[0] % self.N == 0
            x = x.view(int(x.shape[0]/self.N), int(self.N * x.shape[1]), *x.shape[2:])
        else:
            x = x.unsqueeze(1).expand(x.shape[0], self.N, *x.shape[1:])
            x = x.view(x.shape[0], int(x.shape[1] * x.shape[2]), *x.shape[3:])


        if self.batchnorm:
            x = F.relu(self.bn1(F.max_pool2d(self.conv1(x), 2)))
            x = F.max_pool2d(self.conv2(x), 2)
            x = F.relu(self.bn2(x))

            x = self.l1(x.view(int(x.shape[0]/self.N), self.N, -1))
            x = F.relu(self.bn3(x.reshape(int(x.shape[0] * x.shape[1]), *x.shape[2:])))
            x = self.l2(x.view(int(x.shape[0]/self.N), self.N, *x.shape[1:]))
        else:
            x = self.conv1(x)
            x = F.relu(F.max_pool2d(x, 2))
            x = self.conv2(x)
            x = F.relu(F.max_pool2d(x, 2))

            x = self.l1(x.view(x.shape[0], self.N, -1))
            x = F.relu(x)
            x = self.l2(x)
            # print(x.shape)

        if self.training:
            x = x.reshape(int(x.shape[0] * x.shape[1]), *x.shape[2:])

        return x



class mnist_conv_res_bit_tree_ensemble_baens(nn.Module):
    def __init__(self, N=1, bitwidth=1, batchnorm=False):
        super(mnist_conv_res_bit_tree_ensemble_baens, self).__init__()
        self.N = N
        self.conv1 = modules.Conv2d_res_bit_tree_baens(1, 20, 5, N = N, bw=bitwidth, bias=True, quantize=True, first=True)
        self.conv2 = modules.Conv2d_res_bit_tree_baens(20, 50, 5, N = N, bw=bitwidth, bias=True, quantize=True, first=False)

        self.l1 = modules.dense_res_bit_tree_baens(N = N, D1=50*4*4, D2=400, bw=bitwidth, quantize=True)
        self.l2 = modules.dense_res_bit_tree_baens(N = N, D1=400, D2=10, bw=bitwidth, quantize=True)

        self.batchnorm = batchnorm
        if batchnorm:
            self.bn1 = nn.BatchNorm2d(20)
            self.bn2 = nn.BatchNorm2d(50)
            self.bn3 = nn.BatchNorm1d(400)
        # self._init_weights()

    def _init_weights(self):
        for layer in self.children():
            if hasattr(layer, 'weight'): nn.init.xavier_uniform(layer.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        if self.training:
            assert x.shape[0] % self.N == 0
            x = x.view(int(x.shape[0]/self.N), int(self.N * x.shape[1]), *x.shape[2:])
        else:
            x = x.unsqueeze(1).expand(x.shape[0], self.N, *x.shape[1:])
            x = x.view(x.shape[0], int(x.shape[1] * x.shape[2]), *x.shape[3:])

        if self.batchnorm:
            x = F.relu(self.bn1(F.max_pool2d(self.conv1(x), 2)))
            x = F.max_pool2d(self.conv2(x), 2)
            x = F.relu(self.bn2(x))

            x = self.l1(x.view(int(x.shape[0]/self.N), self.N, -1))
            x = F.relu(self.bn3(x.reshape(int(x.shape[0] * x.shape[1]), *x.shape[2:])))
            x = self.l2(x.view(int(x.shape[0]/self.N), self.N, *x.shape[1:]))
        else:
            x = self.conv1(x)
            x = F.relu(F.max_pool2d(x, 2))
            x = self.conv2(x)
            x = F.relu(F.max_pool2d(x, 2))

            x = self.l1(x.view(x.shape[0], self.N, -1))
            x = F.relu(x)
            x = self.l2(x)
            # print(x.shape)

        if self.training:
            x = x.reshape(int(x.shape[0] * x.shape[1]), *x.shape[2:])

        return x


class mnist_conv_res_bit_tree_meanvar_freeze_fine_partition_wkernel_ensemble(nn.Module):
    def __init__(self, N=1, bitwidth=1, batchnorm=False):
        super(mnist_conv_res_bit_tree_meanvar_freeze_fine_partition_wkernel_ensemble, self).__init__()
        self.N = N
        self.conv1 = modules.Conv2d_res_bit_tree_meanvar_freeze_fine_partition_wkernel_baens(1, 20, 5, N = N, bw=bitwidth, bias=True, quantize=True, first=True)
        self.conv2 = modules.Conv2d_res_bit_tree_meanvar_freeze_fine_partition_wkernel_baens(20, 50, 5, N = N, bw=bitwidth, bias=True, quantize=True, first=False)

        self.l1 = modules.dense_res_bit_tree_meanvar_freeze_fine_partition_wkernel_baens(N = N, D1=50*4*4, D2=400, bw=bitwidth, quantize=True)
        self.l2 = modules.dense_res_bit_tree_meanvar_freeze_fine_partition_wkernel_baens(N = N, D1=400, D2=10, bw=bitwidth, quantize=True)
        # self.l2 = modules.dense_res_bit_tree_meanvar_baens(N = N, D1=400, D2=10, bw=bitwidth, quantize=True)
        self.tune = False

        self.batchnorm = batchnorm
        if batchnorm:
            self.bn1 = nn.BatchNorm2d(20)
            self.bn2 = nn.BatchNorm2d(50)
            self.bn3 = nn.BatchNorm1d(400)
        # self._init_weights()

    def _init_weights(self):
        for layer in self.children():
            if hasattr(layer, 'weight'): nn.init.xavier_uniform(layer.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        if self.training and not self.tune:
            assert x.shape[0] % self.N == 0
            x = x.view(int(x.shape[0]/self.N), int(self.N * x.shape[1]), *x.shape[2:])
        else:
            x = x.unsqueeze(1).expand(x.shape[0], self.N, *x.shape[1:])
            x = x.view(x.shape[0], int(x.shape[1] * x.shape[2]), *x.shape[3:])

        if self.batchnorm:
            x = F.relu(self.bn1(F.max_pool2d(self.conv1(x), 2)))
            x = F.max_pool2d(self.conv2(x), 2)
            x = F.relu(self.bn2(x))

            x = self.l1(x.view(int(x.shape[0]/self.N), self.N, -1))
            x = F.relu(self.bn3(x.reshape(int(x.shape[0] * x.shape[1]), *x.shape[2:])))
            x = self.l2(x.view(int(x.shape[0]/self.N), self.N, *x.shape[1:]))
        else:
            x = self.conv1(x)
            x = F.relu(F.max_pool2d(x, 2))
            x = self.conv2(x)
            x = F.relu(F.max_pool2d(x, 2))

            x = self.l1(x.view(x.shape[0], self.N, -1))
            x = F.relu(x)
            x = self.l2(x)
            # print(x.shape)

        if self.training and not self.tune:
            x = x.reshape(int(x.shape[0] * x.shape[1]), *x.shape[2:])

        return x


# class mnist_conv_res_bit_tree_meanvar_freeze_fine_partition_wkernel_ensemble(nn.Module):
#     def __init__(self, N=1, bitwidth=1, batchnorm=False):
#         super(mnist_conv_res_bit_tree_meanvar_freeze_fine_partition_wkernel_ensemble, self).__init__()
#         self.N = N
#         self.conv1 = modules.Conv2d_res_bit_tree_meanvar_freeze_fine_partition_wkernel_wact_baens(1, 20, 5, N = N, bw=bitwidth, bias=True, quantize=True, first=True)
#         self.conv2 = modules.Conv2d_res_bit_tree_meanvar_freeze_fine_partition_wkernel_wact_baens(20, 50, 5, N = N, bw=bitwidth, bias=True, quantize=True, first=False)

#         self.l1 = modules.dense_res_bit_tree_meanvar_freeze_fine_partition_wkernel_wact_baens(N = N, D1=50*4*4, D2=400, bw=bitwidth, quantize=True)
#         self.l2 = modules.dense_res_bit_tree_meanvar_freeze_fine_partition_wkernel_wact_baens(N = N, D1=400, D2=10, bw=bitwidth, quantize=True)

#         self.batchnorm = batchnorm
#         if batchnorm:
#             self.bn1 = nn.BatchNorm2d(20)
#             self.bn2 = nn.BatchNorm2d(50)
#             self.bn3 = nn.BatchNorm1d(400)
#         # self._init_weights()

#     def _init_weights(self):
#         for layer in self.children():
#             if hasattr(layer, 'weight'): nn.init.xavier_uniform(layer.weight, gain=nn.init.calculate_gain('relu'))

#     def forward(self, x):
#         if self.training:
#             assert x.shape[0] % self.N == 0
#             x = x.view(int(x.shape[0]/self.N), int(self.N * x.shape[1]), *x.shape[2:])
#         else:
#             x = x.unsqueeze(1).expand(x.shape[0], self.N, *x.shape[1:])
#             x = x.view(x.shape[0], int(x.shape[1] * x.shape[2]), *x.shape[3:])

#         if self.batchnorm:
#             x = F.relu(self.bn1(F.max_pool2d(self.conv1(x), 2)))
#             x = F.max_pool2d(self.conv2(x), 2)
#             x = F.relu(self.bn2(x))

#             x = self.l1(x.view(int(x.shape[0]/self.N), self.N, -1))
#             x = F.relu(self.bn3(x.reshape(int(x.shape[0] * x.shape[1]), *x.shape[2:])))
#             x = self.l2(x.view(int(x.shape[0]/self.N), self.N, *x.shape[1:]))
#         else:
#             x = self.conv1(x)
#             x = F.relu(F.max_pool2d(x, 2))
#             x = self.conv2(x)
#             x = F.relu(F.max_pool2d(x, 2))

#             x = self.l1(x.view(x.shape[0], self.N, -1))
#             x = F.relu(x)
#             x = self.l2(x)
#             # print(x.shape)

#         if self.training:
#             x = x.reshape(int(x.shape[0] * x.shape[1]), *x.shape[2:])

#         return x


cfgs = {
    '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '11-wide': [96, 'M', 192, 'M', 384, 384, 'M', 768, 768, 'M', 768, 768, 'M'],
    '11-wide-2x': [128, 'M', 256, 'M', 512, 512, 'M', 1024, 1024, 'M', 1024, 1024, 'M'],
    '11-wide-3x': [192, 'M', 384, 'M', 768, 768, 'M', 1536, 1536, 'M', 1536, 1536, 'M'],
    '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    # '16': [64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    '16-wide': [96, 96, 'M', 192, 192, 'M', 384, 384, 384, 'M', 768, 768, 768, 'M', 768, 768, 768, 'M'],
    '16-wide-2x': [128, 128, 'M', 256, 256, 'M', 512, 512, 512, 'M', 1024, 1024, 1024, 'M', 1024, 1024, 1024, 'M'],
    '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    '19-wide': [96, 96, 'M', 192, 192, 'M', 384, 384, 384, 384, 'M', 768, 768, 768, 768, 'M', 768, 768, 768, 768, 'M'],
    '11-trunc': [96, 'M', 126, 'M', 252, 252, 'M', 504, 504, 'M', 504, 504, 'M'],
}

class VGG(nn.Module):
    def __init__(self, output_shape, batch_norm, name, N, bitwidth, K):
        super(VGG, self).__init__()
        config = cfgs[name]
        self.N = N
        self.bitwidth = bitwidth
        self.K = K
        self.features = self.make_layers(config, batch_norm)
        self.dense1 = modules.dense_decompo_nbcp_baens(N=N, D1=config[-2], D2=int(config[-2]/2.), bw=bitwidth, K=K, quantize=True)
        self.dense2 = modules.dense_decompo_nbcp_baens(N=N, D1=int(config[-2]/2.), D2=output_shape, bw=bitwidth, K=K, quantize=True)
        self.bn1 = nn.BatchNorm1d(int(config[-2]/2.))

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for idx, v in enumerate(cfg):
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = modules.Conv2d_decompo_nbcp_baens(in_channels, v, kernel_size=3, padding=1, N = self.N, bw=self.bitwidth, K=self.K, bias=True, quantize=True, first=(idx==0))
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.Hardtanh(inplace=True)]
                else:
                    layers += [conv2d, nn.Hardtanh(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
 
    def forward(self, x):
        if self.training:
            assert x.shape[0] % self.N == 0
            x = x.view(int(x.shape[0]/self.N), int(self.N * x.shape[1]), *x.shape[2:])
        else:
            x = x.unsqueeze(1).expand(x.shape[0], self.N, *x.shape[1:])
            x = x.reshape(x.shape[0], int(x.shape[1] * x.shape[2]), *x.shape[3:])

        out = self.features(x)
        out = out.view(int(out.shape[0]/self.N), self.N, -1)
        # print(out.shape)
        # print(self.dense1.U.shape)
        # print(self.dense1.V.shape)

        out = self.dense1(out)
        out = F.hardtanh(self.bn1(out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])), inplace=True)
        out = self.dense2(out.view(int(out.shape[0]/self.N), self.N, *out.shape[1:]))

        if self.training:
            out = out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])
        
        return out

def set_temp_res_bit_baens(module, temp):
    if isinstance(module, modules.dense_res_bit_baens) or isinstance(module, modules.Conv2d_res_bit_baens):
        module.temp = temp
    if hasattr(module, 'children'):
        for submodule in module.children():
            set_temp_nbcp_baens(submodule, temp)

class VGG_res_bit_ensemble(nn.Module):
    def __init__(self, output_shape, batch_norm, name, N, bitwidth):
        super(VGG_res_bit_ensemble, self).__init__()
        config = cfgs[name]
        self.N = N
        self.bitwidth = bitwidth
        self.features = self.make_layers(config, batch_norm)
        self.dense1 = modules.dense_res_bit_baens(N=N, D1=config[-2], D2=int(config[-2]/2.), bw=bitwidth, quantize=True)
        self.dense2 = modules.dense_res_bit_baens(N=N, D1=int(config[-2]/2.), D2=output_shape, bw=bitwidth, quantize=True)
        self.bn1 = nn.BatchNorm1d(int(config[-2]/2.))

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for idx, v in enumerate(cfg):
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = modules.Conv2d_res_bit_baens(in_channels, v, kernel_size=3, padding=1, N=self.N, bw=self.bitwidth, bias=True, quantize=True, first=(idx==0))
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.training:
            assert x.shape[0] % self.N == 0
            x = x.view(int(x.shape[0]/self.N), int(self.N * x.shape[1]), *x.shape[2:])
        else:
            x = x.unsqueeze(1).expand(x.shape[0], self.N, *x.shape[1:])
            x = x.reshape(x.shape[0], int(x.shape[1] * x.shape[2]), *x.shape[3:])

        out = self.features(x)
        out = out.view(int(out.shape[0]/self.N), self.N, -1)

        out = self.dense1(out)
        out = F.hardtanh(self.bn1(out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])), inplace=True)
        out = self.dense2(out.view(int(out.shape[0]/self.N), self.N, *out.shape[1:]))

        if self.training:
            out = out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])

        return out


class VGG_res_bit_tree_ensemble(nn.Module):
    def __init__(self, output_shape, batch_norm, name, N, bitwidth):
        super(VGG_res_bit_tree_ensemble, self).__init__()
        config = cfgs[name]
        self.N = N
        self.bitwidth = bitwidth
        self.features = self.make_layers(config, batch_norm)
        self.dense1 = modules.dense_res_bit_tree_baens(N=N, D1=config[-2], D2=int(config[-2]/2.), bw=bitwidth, quantize=True)
        self.dense2 = modules.dense_res_bit_tree_baens(N=N, D1=int(config[-2]/2.), D2=output_shape, bw=bitwidth, quantize=True)
        self.bn1 = nn.BatchNorm1d(int(config[-2]/2.))

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for idx, v in enumerate(cfg):
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = modules.Conv2d_res_bit_tree_baens(in_channels, v, kernel_size=3, padding=1, N = self.N, bw=self.bitwidth, bias=True, quantize=True, first=(idx==0))
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.training:
            assert x.shape[0] % self.N == 0
            x = x.view(int(x.shape[0]/self.N), int(self.N * x.shape[1]), *x.shape[2:])
        else:
            x = x.unsqueeze(1).expand(x.shape[0], self.N, *x.shape[1:])
            x = x.reshape(x.shape[0], int(x.shape[1] * x.shape[2]), *x.shape[3:])

        out = self.features(x)
        out = out.view(int(out.shape[0]/self.N), self.N, -1)

        out = self.dense1(out)
        out = F.hardtanh(self.bn1(out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])), inplace=True)
        out = self.dense2(out.view(int(out.shape[0]/self.N), self.N, *out.shape[1:]))

        if self.training:
            out = out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])
        
        return out


class VGG_res_bit_tree_meanvar_ensemble(nn.Module):
    def __init__(self, output_shape, batch_norm, name, N, bitwidth):
        super(VGG_res_bit_tree_meanvar_ensemble, self).__init__()
        config = cfgs[name]
        self.N = N
        self.bitwidth = bitwidth
        self.features = self.make_layers(config, batch_norm)
        self.dense1 = modules.dense_res_bit_tree_meanvar_baens(N=N, D1=config[-2], D2=int(config[-2]/2.), bw=bitwidth, quantize=True)
        self.dense2 = modules.dense_res_bit_tree_meanvar_baens(N=N, D1=int(config[-2]/2.), D2=output_shape, bw=bitwidth, quantize=True)
        self.bn1 = nn.BatchNorm1d(int(config[-2]/2.))

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for idx, v in enumerate(cfg):
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = modules.Conv2d_res_bit_tree_meanvar_baens(in_channels, v, kernel_size=3, padding=1, N = self.N, bw=self.bitwidth, bias=True, quantize=True, first=(idx==0))
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.training:
            assert x.shape[0] % self.N == 0
            x = x.view(int(x.shape[0]/self.N), int(self.N * x.shape[1]), *x.shape[2:])
        else:
            x = x.unsqueeze(1).expand(x.shape[0], self.N, *x.shape[1:])
            x = x.reshape(x.shape[0], int(x.shape[1] * x.shape[2]), *x.shape[3:])

        out = self.features(x)
        out = out.view(int(out.shape[0]/self.N), self.N, -1)

        out = self.dense1(out)
        out = F.hardtanh(self.bn1(out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])), inplace=True)
        out = self.dense2(out.view(int(out.shape[0]/self.N), self.N, *out.shape[1:]))

        if self.training:
            out = out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])
        return out


class VGG_res_bit_tree_meanvar_freeze_partition_ensemble(nn.Module):
    def __init__(self, output_shape, batch_norm, name, N, bitwidth):
        super(VGG_res_bit_tree_meanvar_freeze_partition_ensemble, self).__init__()
        config = cfgs[name]
        self.N = N
        self.bitwidth = bitwidth
        self.features = self.make_layers(config, batch_norm)
        self.dense1 = modules.dense_res_bit_tree_meanvar_freeze_partition_baens(N=N, D1=config[-2], D2=int(config[-2]/2.), bw=bitwidth, quantize=True)
        self.dense2 = modules.dense_res_bit_tree_meanvar_freeze_partition_baens(N=N, D1=int(config[-2]/2.), D2=output_shape, bw=bitwidth, quantize=True)
        self.bn1 = nn.BatchNorm1d(int(config[-2]/2.))

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for idx, v in enumerate(cfg):
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = modules.Conv2d_res_bit_tree_meanvar_freeze_partition_baens(in_channels, v, kernel_size=3, padding=1, N = self.N, bw=self.bitwidth, bias=True, quantize=True, first=(idx==0))
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.training:
            assert x.shape[0] % self.N == 0
            x = x.view(int(x.shape[0]/self.N), int(self.N * x.shape[1]), *x.shape[2:])
        else:
            x = x.unsqueeze(1).expand(x.shape[0], self.N, *x.shape[1:])
            x = x.reshape(x.shape[0], int(x.shape[1] * x.shape[2]), *x.shape[3:])

        out = self.features(x)
        out = out.view(int(out.shape[0]/self.N), self.N, -1)

        out = self.dense1(out)
        out = F.hardtanh(self.bn1(out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])), inplace=True)
        out = self.dense2(out.view(int(out.shape[0]/self.N), self.N, *out.shape[1:]))

        if self.training:
            out = out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])
        return out

class VGG_res_bit_tree_meanvar_freeze_fine_partition_ensemble(nn.Module):
    def __init__(self, output_shape, batch_norm, name, N, bitwidth):
        super(VGG_res_bit_tree_meanvar_freeze_fine_partition_ensemble, self).__init__()
        config = cfgs[name]
        self.N = N
        self.bitwidth = bitwidth
        self.features = self.make_layers(config, batch_norm)
        self.dense1 = modules.dense_res_bit_tree_meanvar_freeze_fine_partition_baens(N=N, D1=config[-2], D2=int(config[-2]/2.), bw=bitwidth, quantize=True)
        self.dense2 = modules.dense_res_bit_tree_meanvar_freeze_fine_partition_baens(N=N, D1=int(config[-2]/2.), D2=output_shape, bw=bitwidth, quantize=True)
        self.bn1 = nn.BatchNorm1d(int(config[-2]/2.))

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for idx, v in enumerate(cfg):
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = modules.Conv2d_res_bit_tree_meanvar_freeze_fine_partition_baens(in_channels, v, kernel_size=3, padding=1, N = self.N, bw=self.bitwidth, bias=True, quantize=True, first=(idx==0))
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.training:
            assert x.shape[0] % self.N == 0
            x = x.view(int(x.shape[0]/self.N), int(self.N * x.shape[1]), *x.shape[2:])
        else:
            x = x.unsqueeze(1).expand(x.shape[0], self.N, *x.shape[1:])
            x = x.reshape(x.shape[0], int(x.shape[1] * x.shape[2]), *x.shape[3:])

        out = self.features(x)
        out = out.view(int(out.shape[0]/self.N), self.N, -1)

        out = self.dense1(out)
        out = F.hardtanh(self.bn1(out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])), inplace=True)
        out = self.dense2(out.view(int(out.shape[0]/self.N), self.N, *out.shape[1:]))

        if self.training:
            out = out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])
        return out


class VGG_res_bit_tree_meanvar_freeze_fine_partition_wkernel_ensemble(nn.Module):
    def __init__(self, output_shape, batch_norm, name, N, bitwidth):
        super(VGG_res_bit_tree_meanvar_freeze_fine_partition_wkernel_ensemble, self).__init__()
        config = cfgs[name]
        self.N = N
        self.bitwidth = bitwidth
        self.tune = True

        self.features = self.make_layers(config, batch_norm)
        self.dense1 = modules.dense_res_bit_tree_meanvar_freeze_fine_partition_wkernel_baens(N=N, D1=config[-2], D2=int(config[-2]/2.), bw=bitwidth, quantize=True)
        self.dense2 = modules.dense_res_bit_tree_meanvar_freeze_fine_partition_wkernel_baens(N=N, D1=int(config[-2]/2.), D2=output_shape, bw=bitwidth, quantize=True)
        self.bn1 = nn.BatchNorm1d(int(config[-2]/2.))

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for idx, v in enumerate(cfg):
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = modules.Conv2d_res_bit_tree_meanvar_freeze_fine_partition_wkernel_baens(in_channels, v, kernel_size=3, padding=1, N = self.N, bw=self.bitwidth, bias=True, quantize=True, first=(idx==0))
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.training and not self.tune:
            assert x.shape[0] % self.N == 0
            x = x.view(int(x.shape[0]/self.N), int(self.N * x.shape[1]), *x.shape[2:])
        else:
            x = x.unsqueeze(1).expand(x.shape[0], self.N, *x.shape[1:])
            x = x.reshape(x.shape[0], int(x.shape[1] * x.shape[2]), *x.shape[3:])

        out = self.features(x)
        out = out.view(int(out.shape[0]/self.N), self.N, -1)

        out = self.dense1(out)
        out = F.hardtanh(self.bn1(out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])), inplace=True)
        out = self.dense2(out.view(int(out.shape[0]/self.N), self.N, *out.shape[1:]))

        if self.training and not self.tune:
            out = out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])
        return out


class VGG_res_bit_tree_meanvar_freeze_fine_partition_wkernel_kv_ensemble(nn.Module):
    def __init__(self, output_shape, batch_norm, name, N, bitwidth):
        super(VGG_res_bit_tree_meanvar_freeze_fine_partition_wkernel_kv_ensemble, self).__init__()
        config = cfgs[name]
        self.N = N
        self.bitwidth = bitwidth
        self.tune = True

        self.features = self.make_layers(config, batch_norm)
        self.dense1 = modules.dense_res_bit_tree_meanvar_freeze_fine_partition_wkernel_kv_baens(N=N, D1=config[-2], D2=int(config[-2]/2.), bw=bitwidth, quantize=True)
        self.dense2 = modules.dense_res_bit_tree_meanvar_freeze_fine_partition_wkernel_kv_baens(N=N, D1=int(config[-2]/2.), D2=output_shape, bw=bitwidth, quantize=True)
        self.bn1 = nn.BatchNorm1d(int(config[-2]/2.))

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for idx, v in enumerate(cfg):
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = modules.Conv2d_res_bit_tree_meanvar_freeze_fine_partition_wkernel_kv_baens(in_channels, v, kernel_size=3, 
                    padding=1, N = self.N, bw=self.bitwidth, bias=True, quantize=True, first=(idx==0))
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.training and not self.tune:
            assert x.shape[0] % self.N == 0
            x = x.view(int(x.shape[0]/self.N), int(self.N * x.shape[1]), *x.shape[2:])
        else:
            x = x.unsqueeze(1).expand(x.shape[0], self.N, *x.shape[1:])
            x = x.reshape(x.shape[0], int(x.shape[1] * x.shape[2]), *x.shape[3:])

        out = self.features(x)
        out = out.view(int(out.shape[0]/self.N), self.N, -1)

        out = self.dense1(out)
        out = F.hardtanh(self.bn1(out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])), inplace=True)
        out = self.dense2(out.view(int(out.shape[0]/self.N), self.N, *out.shape[1:]))

        if self.training and not self.tune:
            out = out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])
        return out

class VGG_res_bit_tree_meanvar_freeze_fine_partition_wkernel_kv_gn_ensemble(nn.Module):
    def __init__(self, output_shape, batch_norm, name, N, bitwidth, bias=False):
        super(VGG_res_bit_tree_meanvar_freeze_fine_partition_wkernel_kv_gn_ensemble, self).__init__()
        config = cfgs[name]
        self.N = N
        self.bitwidth = bitwidth
        self.tune = True
        self.bias = bias

        self.features = self.make_layers(config, batch_norm)
        self.dense1 = modules.dense_res_bit_tree_meanvar_freeze_fine_partition_wkernel_kv_gn_baens(N=N, D1=config[-2], D2=int(config[-2]/2.), bw=bitwidth, quantize=True, bias=self.bias)
        self.dense2 = modules.dense_res_bit_tree_meanvar_freeze_fine_partition_wkernel_kv_gn_baens(N=N, D1=int(config[-2]/2.), D2=output_shape, bw=bitwidth, quantize=True, bias=self.bias)
        self.bn_list = nn.ModuleList()
        for i in range(N):
            self.bn_list.append(nn.BatchNorm1d(int(config[-2]/2.)))
        # nn.BatchNorm1d(int(config[-2]/2.))

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for idx, v in enumerate(cfg):
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = modules.Conv2d_res_bit_tree_meanvar_freeze_fine_partition_wkernel_kv_gn_baens(in_channels, v, kernel_size=3, 
                    padding=1, N = self.N, bw=self.bitwidth, bias=self.bias, quantize=True, first=(idx==0))
                if batch_norm:
                    layers += [conv2d, nn.GroupNorm(self.N, int(self.N*v)), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.training or self.tune:
            assert x.shape[0] % self.N == 0
            x = x.view(int(x.shape[0]/self.N), int(self.N * x.shape[1]), *x.shape[2:])
        else:
            x = x.unsqueeze(1).expand(x.shape[0], self.N, *x.shape[1:])
            x = x.reshape(x.shape[0], int(x.shape[1] * x.shape[2]), *x.shape[3:])

        out = self.features(x)
        out = out.view(out.shape[0], self.N, -1)
        out = self.dense1(out)
        chunks = []
        for i in range(self.N):
            chunks.append(F.hardtanh(self.bn_list[i](out[:, i, :]), inplace=True))
        out = torch.stack(chunks, dim=1)
        # out = F.hardtanh(self.bn1(out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])), inplace=True)
        # out = self.dense2(out.view(int(out.shape[0]/self.N), self.N, *out.shape[1:]))
        out = self.dense2(out)

        if self.training or self.tune:
            out = out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])

        return out

class VGG_full_ens(nn.Module):
    def __init__(self, output_shape, batch_norm, name, N, bias=False):
        super(VGG_full_ens, self).__init__()
        config = cfgs[name]
        self.N = N
        self.bias = bias

        self.features = self.make_layers(config, batch_norm)
        self.dense1 = modules.dense_full_ens(N=N, D1=config[-2], D2=int(config[-2]/2.))
        self.dense2 = modules.dense_full_ens(N=N, D1=int(config[-2]/2.), D2=output_shape)
        self.bn_list = nn.ModuleList()
        for i in range(N):
            self.bn_list.append(nn.BatchNorm1d(int(config[-2]/2.)))
        # nn.BatchNorm1d(int(config[-2]/2.))

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for idx, v in enumerate(cfg):
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = modules.Conv2d_full_ens(in_channels, v, kernel_size=3, 
                    padding=1, N = self.N, first=(idx==0))
                if batch_norm:
                    layers += [conv2d, nn.GroupNorm(self.N, int(self.N*v)), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):

        if self.training:
            assert x.shape[0] % self.N == 0
            x = x.view(int(x.shape[0]/self.N), int(self.N * x.shape[1]), *x.shape[2:])
        else:
            x = x.unsqueeze(1).expand(x.shape[0], self.N, *x.shape[1:])
            x = x.reshape(x.shape[0], int(x.shape[1] * x.shape[2]), *x.shape[3:])

        out = self.features(x)
        out = out.view(out.shape[0], self.N, -1)
        out = self.dense1(out)
        chunks = []
        for i in range(self.N):
            chunks.append(F.hardtanh(self.bn_list[i](out[:, i, :]), inplace=True))

        out = torch.stack(chunks, dim=1)
        # out = F.hardtanh(self.bn1(out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])), inplace=True)
        # out = self.dense2(out.view(int(out.shape[0]/self.N), self.N, *out.shape[1:]))
        out = self.dense2(out)

        if self.training:
            out = out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])

        return out



class VGG_res_bit_tree_meanvar_freeze_fine_partition_wkernel_wact_ensemble(nn.Module):
    def __init__(self, output_shape, batch_norm, name, N, bitwidth):
        super(VGG_res_bit_tree_meanvar_freeze_fine_partition_wkernel_wact_ensemble, self).__init__()
        config = cfgs[name]
        self.N = N
        self.bitwidth = bitwidth
        self.features = self.make_layers(config, batch_norm)
        self.dense1 = modules.dense_res_bit_tree_meanvar_freeze_fine_partition_wkernel_wact_baens(N=N, D1=config[-2], D2=int(config[-2]/2.), bw=bitwidth, quantize=True)
        self.dense2 = modules.dense_res_bit_tree_meanvar_freeze_fine_partition_wkernel_wact_baens(N=N, D1=int(config[-2]/2.), D2=output_shape, bw=bitwidth, quantize=True)
        self.bn1 = nn.BatchNorm1d(int(config[-2]/2.))

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for idx, v in enumerate(cfg):
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = modules.Conv2d_res_bit_tree_meanvar_freeze_fine_partition_wkernel_wact_baens(in_channels, v, kernel_size=3, padding=1, N = self.N, bw=self.bitwidth, bias=True, quantize=True, first=(idx==0))
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.training:
            assert x.shape[0] % self.N == 0
            x = x.view(int(x.shape[0]/self.N), int(self.N * x.shape[1]), *x.shape[2:])
        else:
            x = x.unsqueeze(1).expand(x.shape[0], self.N, *x.shape[1:])
            x = x.reshape(x.shape[0], int(x.shape[1] * x.shape[2]), *x.shape[3:])

        out = self.features(x)
        out = out.view(int(out.shape[0]/self.N), self.N, -1)

        out = self.dense1(out)
        out = F.hardtanh(self.bn1(out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])), inplace=True)
        out = self.dense2(out.view(int(out.shape[0]/self.N), self.N, *out.shape[1:]))

        if self.training:
            out = out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])
        return out


class VGG_res_bit_tree_meanvar_freeze_fine_partition_wkernel_wact_wstats_ensemble(nn.Module):
    def __init__(self, output_shape, batch_norm, name, N, bitwidth):
        super(VGG_res_bit_tree_meanvar_freeze_fine_partition_wkernel_wact_wstats_ensemble, self).__init__()
        config = cfgs[name]
        self.N = N
        self.bitwidth = bitwidth
        self.features = self.make_layers(config, batch_norm)
        self.dense1 = modules.dense_res_bit_tree_meanvar_freeze_fine_partition_wkernel_wact_wstats_baens(N=N, D1=config[-2], D2=int(config[-2]/2.), bw=bitwidth, quantize=True)
        self.dense2 = modules.dense_res_bit_tree_meanvar_freeze_fine_partition_wkernel_wact_wstats_baens(N=N, D1=int(config[-2]/2.), D2=output_shape, bw=bitwidth, quantize=True)
        self.bn1 = nn.BatchNorm1d(int(config[-2]/2.))

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for idx, v in enumerate(cfg):
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = modules.Conv2d_res_bit_tree_meanvar_freeze_fine_partition_wkernel_wact_wstats_baens(in_channels, v, kernel_size=3, padding=1, N = self.N, bw=self.bitwidth, bias=True, quantize=True, first=(idx==0))
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.training:
            assert x.shape[0] % self.N == 0
            x = x.view(int(x.shape[0]/self.N), int(self.N * x.shape[1]), *x.shape[2:])
        else:
            x = x.unsqueeze(1).expand(x.shape[0], self.N, *x.shape[1:])
            x = x.reshape(x.shape[0], int(x.shape[1] * x.shape[2]), *x.shape[3:])

        out = self.features(x)
        out = out.view(int(out.shape[0]/self.N), self.N, -1)

        out = self.dense1(out)
        out = F.hardtanh(self.bn1(out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])), inplace=True)
        out = self.dense2(out.view(int(out.shape[0]/self.N), self.N, *out.shape[1:]))

        if self.training:
            out = out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])
        return out

class VGG_baens(nn.Module):
    def __init__(self, output_shape, batch_norm, name, N):
        super(VGG_baens, self).__init__()
        config = cfgs[name]
        self.N = N
        self.features = self.make_layers(config, batch_norm)
        self.dense1 = modules.dense_baens(N=N, D1=config[-2], D2=config[-2])
        self.dense2 = modules.dense_baens(N=N, D1=config[-2], D2=output_shape)
        self.bn1 = nn.BatchNorm1d(config[-2])

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for idx, v in enumerate(cfg):
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = modules.Conv2d_baens(in_channels, v, kernel_size=3, padding=1, N = self.N, bias=True, first=(idx==0))
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.training:
            assert x.shape[0] % self.N == 0
            x = x.view(int(x.shape[0]/self.N), int(self.N * x.shape[1]), *x.shape[2:])
        else:
            x = x.unsqueeze(1).expand(x.shape[0], self.N, *x.shape[1:])
            x = x.reshape(x.shape[0], int(x.shape[1] * x.shape[2]), *x.shape[3:])

        out = self.features(x)
        out = out.view(int(out.shape[0]/self.N), self.N, -1)

        out = self.dense1(out)
        out = F.hardtanh(self.bn1(out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])), inplace=True)
        out = self.dense2(out.view(int(out.shape[0]/self.N), self.N, *out.shape[1:]))

        if self.training:
            out = out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])
        return out

class VGG_baens_ref(nn.Module):
    def __init__(self, output_shape, batch_norm, config, N, mode='feat-approx', blk_cnt=0):
        super(VGG_baens_ref, self).__init__()
        # config = cfgs[name]
        self.N = N
        self.features = self.make_layers(config, batch_norm, blk_cnt)
        self.mode = mode
        if mode == 'feat-approx':
            pass
        elif 'dense' in mode:
            self.dense1 = modules.dense_baens(N=N, D1=config[-2], D2=int(config[-2]/2.))
            if mode == 'dense2-approx':
                self.dense2 = modules.dense_baens(N=N, D1=int(config[-2]/2.), D2=output_shape)
                self.bn1 = nn.BatchNorm1d(int(config[-2]/2.))

    def make_layers(self, cfg, batch_norm=False, blk_cnt=0):
        layers = []
        in_channels = 3
        cnt = 0
        seq_module = nn.Sequential()

        for idx, v in enumerate(cfg):
            blk_cnter = 0
            if v == 'M':
                seq_module.add_module('{}'.format(cnt), 
                    nn.MaxPool2d(kernel_size=2, stride=2))
                layers += []
                cnt += 1
            else:
                seq_module.add_module('{}'.format(cnt), 
                    modules.Conv2d_baens(in_channels, v, kernel_size=3, padding=1, N = self.N, bias=True, first=(idx==0)))
                cnt += 1
                if blk_cnter == blk_cnt and idx == len(cfg) and self.mode == 'feat-approx':
                    return seq_module
                blk_cnter += 1

                if batch_norm:
                    seq_module.add_module('{}'.format(cnt), 
                    nn.BatchNorm2d(v))
                cnt += 1
                if blk_cnter == blk_cnt and idx == len(cfg) and self.mode == 'feat-approx':
                    return seq_module
                blk_cnter += 1

                seq_module.add_module('{}'.format(cnt), 
                    nn.ReLU(inplace=True))
                cnt += 1
                blk_cnter += 1

                in_channels = v
        return seq_module

    def forward(self, x):
        if self.training:
            assert x.shape[0] % self.N == 0
            x = x.view(int(x.shape[0]/self.N), int(self.N * x.shape[1]), *x.shape[2:])
        else:
            x = x.unsqueeze(1).expand(x.shape[0], self.N, *x.shape[1:])
            x = x.reshape(x.shape[0], int(x.shape[1] * x.shape[2]), *x.shape[3:])

        out = self.features(x)
        if self.mode == 'feat-approx':
            return out

        out = out.view(int(out.shape[0]/self.N), self.N, -1)

        out = self.dense1(out)
        if self.mode == 'dense1-approx':
            return out
        out = F.hardtanh(self.bn1(out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])), inplace=True)
        out = self.dense2(out.view(int(out.shape[0]/self.N), self.N, *out.shape[1:]))

        if self.training:
            out = out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])
        return out

class VGG_baens_kv(nn.Module):
    def __init__(self, output_shape, batch_norm, config, N, mode='feat-approx', blk_cnt=0):
        super(VGG_baens_kv, self).__init__()
        # config = cfgs[name]
        self.N = N
        self.mode = mode
        self.features = self.make_layers(config, batch_norm, blk_cnt)
        if mode == 'feat-approx':
            pass
        elif 'dense' in mode:
            self.dense1 = modules.dense_kv_baens(N=N, D1=config[-2], D2=int(config[-2]/2.))
            if mode == 'dense2-approx':
                self.bn1 = nn.BatchNorm1d(int(config[-2]/2.))
                self.dense2 = modules.dense_kv_baens(N=N, D1=int(config[-2]/2.), D2=output_shape)

    def make_layers(self, cfg, batch_norm=False, blk_cnt=0):
        layers = []
        in_channels = 3
        cnt = 0
        seq_module = nn.Sequential()

        for idx, v in enumerate(cfg):
            blk_cnter = 0
            if v == 'M':
                seq_module.add_module('{}'.format(cnt), 
                    nn.MaxPool2d(kernel_size=2, stride=2))
                layers += []
                cnt += 1
            else:
                seq_module.add_module('{}'.format(cnt), 
                    modules.Conv2d_kv_baens(in_channels, v, kernel_size=3, padding=1, N = self.N, bias=True, first=(idx==0)))
                cnt += 1
                if blk_cnter == blk_cnt and idx == len(cfg) - 1 and self.mode == 'feat-approx':
                    return seq_module
                blk_cnter += 1

                if batch_norm:
                    seq_module.add_module('{}'.format(cnt), 
                    nn.BatchNorm2d(v))
                cnt += 1
                if blk_cnter == blk_cnt and idx == len(cfg) - 1 and self.mode == 'feat-approx':
                    return seq_module
                blk_cnter += 1

                seq_module.add_module('{}'.format(cnt), 
                    nn.ReLU(inplace=True))
                cnt += 1
                blk_cnter += 1

                in_channels = v
        return seq_module

    def forward(self, x):
        if self.training:
            assert x.shape[0] % self.N == 0
            x = x.view(int(x.shape[0]/self.N), int(self.N * x.shape[1]), *x.shape[2:])
        else:
            x = x.unsqueeze(1).expand(x.shape[0], self.N, *x.shape[1:])
            x = x.reshape(x.shape[0], int(x.shape[1] * x.shape[2]), *x.shape[3:])

        out = self.features(x)
        if self.mode == 'feat-approx':
            return out

        out = out.view(int(out.shape[0]/self.N), self.N, -1)

        out = self.dense1(out)
        if self.mode == 'dense1-approx':
            return out
        out = F.hardtanh(self.bn1(out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])), inplace=True)
        out = self.dense2(out.view(int(out.shape[0]/self.N), self.N, *out.shape[1:]))

        if self.training:
            out = out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])
        return out


class VGG_baens_quant(nn.Module):
    def __init__(self, output_shape, batch_norm, name, N):
        super(VGG_baens_quant, self).__init__()
        config = cfgs[name]
        self.N = N
        self.features = self.make_layers(config, batch_norm)
        self.dense1 = modules.dense_baens_quant(N=N, D1=config[-2], D2=int(config[-2]/2.))
        self.dense2 = modules.dense_baens_quant(N=N, D1=int(config[-2]/2.), D2=output_shape)
        self.bn_list = nn.ModuleList()
        for i in range(N):
            self.bn_list.append(nn.BatchNorm1d(int(config[-2]/2.)))

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for idx, v in enumerate(cfg):
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = modules.Conv2d_baens_quant(in_channels, v, kernel_size=3, padding=1, N = self.N, bias=True, first=(idx==0))
                if batch_norm:
                    layers += [conv2d, nn.GroupNorm(self.N, int(self.N*v)), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.training:
            assert x.shape[0] % self.N == 0
            x = x.view(int(x.shape[0]/self.N), int(self.N * x.shape[1]), *x.shape[2:])
        else:
            x = x.unsqueeze(1).expand(x.shape[0], self.N, *x.shape[1:])
            x = x.reshape(x.shape[0], int(x.shape[1] * x.shape[2]), *x.shape[3:])

        out = self.features(x)
        out = out.view(out.shape[0], self.N, -1)
        out = self.dense1(out)
        chunks = []
        for i in range(self.N):
            chunks.append(F.hardtanh(self.bn_list[i](out[:, i, :]), inplace=True))

        out = torch.stack(chunks, dim=1)
        # out = F.hardtanh(self.bn1(out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])), inplace=True)
        # out = self.dense2(out.view(int(out.shape[0]/self.N), self.N, *out.shape[1:]))
        out = self.dense2(out)

        if self.training:
            out = out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])
        return out


class VGG_res_bit_tree_uni_normal_ensemble(nn.Module):
    def __init__(self, output_shape, batch_norm, name, N, bitwidth):
        super(VGG_res_bit_tree_uni_normal_ensemble, self).__init__()
        config = cfgs[name]
        self.N = N
        self.bitwidth = bitwidth
        self.features = self.make_layers(config, batch_norm)
        self.dense1 = modules.dense_res_bit_tree_uni_normal_baens(N=N, D1=config[-2], D2=int(config[-2]/2.), bw=bitwidth, quantize=True)
        self.dense2 = modules.dense_res_bit_tree_uni_normal_baens(N=N, D1=int(config[-2]/2.), D2=output_shape, bw=bitwidth, quantize=True)
        self.bn1 = nn.BatchNorm1d(int(config[-2]/2.))

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for idx, v in enumerate(cfg):
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = modules.Conv2d_res_bit_tree_uni_normal_baens(in_channels, v, kernel_size=3, padding=1, N = self.N, bw=self.bitwidth, bias=True, quantize=True, first=(idx==0))
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.training:
            assert x.shape[0] % self.N == 0
            x = x.view(int(x.shape[0]/self.N), int(self.N * x.shape[1]), *x.shape[2:])
        else:
            x = x.unsqueeze(1).expand(x.shape[0], self.N, *x.shape[1:])
            x = x.reshape(x.shape[0], int(x.shape[1] * x.shape[2]), *x.shape[3:])

        out = self.features(x)
        out = out.view(int(out.shape[0]/self.N), self.N, -1)

        out = self.dense1(out)
        out = F.hardtanh(self.bn1(out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])), inplace=True)
        out = self.dense2(out.view(int(out.shape[0]/self.N), self.N, *out.shape[1:]))

        if self.training:
            out = out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])

        return out


class VGG_res_bit_tree_half_ensemble(nn.Module):
    def __init__(self, output_shape, batch_norm, name, N, bitwidth):
        super(VGG_res_bit_tree_half_ensemble, self).__init__()
        config = cfgs[name]
        self.N = N
        self.bitwidth = bitwidth
        self.features = self.make_layers(config, batch_norm)
        self.dense1 = modules.dense_res_bit_tree_half_baens(N=N, D1=config[-2], D2=int(config[-2]/2.), bw=bitwidth, quantize=True)
        self.dense2 = modules.dense_res_bit_tree_half_baens(N=N, D1=int(config[-2]/2.), D2=output_shape, bw=bitwidth, quantize=True)
        self.bn1 = nn.BatchNorm1d(int(config[-2]/2.))

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for idx, v in enumerate(cfg):
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = modules.Conv2d_res_bit_tree_half_baens(in_channels, v, kernel_size=3, padding=1, N = self.N, bw=self.bitwidth, bias=True, quantize=True, first=(idx==0))
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.training:
            assert x.shape[0] % self.N == 0
            x = x.view(int(x.shape[0]/self.N), int(self.N * x.shape[1]), *x.shape[2:])
        else:
            x = x.unsqueeze(1).expand(x.shape[0], self.N, *x.shape[1:])
            x = x.reshape(x.shape[0], int(x.shape[1] * x.shape[2]), *x.shape[3:])

        out = self.features(x)
        out = out.view(int(out.shape[0]/self.N), self.N, -1)

        out = self.dense1(out)
        out = F.hardtanh(self.bn1(out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])), inplace=True)
        out = self.dense2(out.view(int(out.shape[0]/self.N), self.N, *out.shape[1:]))

        if self.training:
            out = out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])

        return out


class VGG_res_bit_tree_fixed_ensemble(nn.Module):
    def __init__(self, output_shape, batch_norm, name, N, bitwidth, partition):
        super(VGG_res_bit_tree_fixed_ensemble, self).__init__()
        config = cfgs[name]
        self.N = N
        self.bitwidth = bitwidth
        self.partition = partition
        self.features = self.make_layers(config, batch_norm)
        self.dense1 = modules.dense_res_bit_tree_fixed_baens(N=N, D1=config[-2], D2=int(config[-2]/2.), bw=bitwidth, quantize=True, partition = partition)
        self.dense2 = modules.dense_res_bit_tree_fixed_baens(N=N, D1=int(config[-2]/2.), D2=output_shape, bw=bitwidth, quantize=True, partition = partition)
        self.bn1 = nn.BatchNorm1d(int(config[-2]/2.))

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for idx, v in enumerate(cfg):
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = modules.Conv2d_res_bit_tree_fixed_baens(in_channels, v, kernel_size=3, padding=1, N = self.N, bw=self.bitwidth, bias=True, quantize=True, first=(idx==0), partition = self.partition)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.training:
            assert x.shape[0] % self.N == 0
            x = x.view(int(x.shape[0]/self.N), int(self.N * x.shape[1]), *x.shape[2:])
        else:
            x = x.unsqueeze(1).expand(x.shape[0], self.N, *x.shape[1:])
            x = x.reshape(x.shape[0], int(x.shape[1] * x.shape[2]), *x.shape[3:])

        out = self.features(x)
        out = out.view(int(out.shape[0]/self.N), self.N, -1)

        out = self.dense1(out)
        out = F.hardtanh(self.bn1(out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])), inplace=True)
        out = self.dense2(out.view(int(out.shape[0]/self.N), self.N, *out.shape[1:]))

        if self.training:
            out = out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])

        return out


class VGG_independent_baens(nn.Module):
    def __init__(self, output_shape, batch_norm, name, N):
        super(VGG_independent_baens, self).__init__()
        config = cfgs[name]
        self.N = N
        self.features = self.make_layers(config, batch_norm)
        self.dense1 = modules.dense_independent_baens(N=N, D1=config[-2], D2=int(config[-2]/2.), quantize=True)
        self.dense2 = modules.dense_independent_baens(N=N, D1=int(config[-2]/2.), D2=output_shape, quantize=True)
        self.bn1 = nn.BatchNorm1d(int(config[-2]/2.))

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for idx, v in enumerate(cfg):
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = modules.Conv2d_independent_baens(in_channels, v, kernel_size=3, padding=1, N = self.N, bias=True, quantize=True, first=(idx==0))
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.training:
            assert x.shape[0] % self.N == 0
            x = x.view(int(x.shape[0]/self.N), int(self.N * x.shape[1]), *x.shape[2:])
        else:
            x = x.unsqueeze(1).expand(x.shape[0], self.N, *x.shape[1:])
            x = x.reshape(x.shape[0], int(x.shape[1] * x.shape[2]), *x.shape[3:])

        out = self.features(x)
        out = out.view(int(out.shape[0]/self.N), self.N, -1)

        out = self.dense1(out)
        out = F.hardtanh(self.bn1(out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])), inplace=True)
        out = self.dense2(out.view(int(out.shape[0]/self.N), self.N, *out.shape[1:]))

        if self.training:
            out = out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])
        
        return out


def conv3x3_baens(in_planes, out_planes, stride=1, N=4, bitwidth=3):
    return modules.Conv2d_baens(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True, N=N, first=True)

def conv_init_baens(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d_baens') != -1:
        init.xavier_uniform_(m.U, gain=np.sqrt(2))
        init.xavier_uniform_(m.S, gain=np.sqrt(2))
        init.xavier_uniform_(m.R, gain=np.sqrt(2))

        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class wide_basic_baens(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1, N=4, bitwidth=3):
        super(wide_basic_baens, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = modules.Conv2d_baens(in_planes, planes, kernel_size=3, padding=1, bias=True, N=N)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = modules.Conv2d_baens(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True, N=N)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                modules.Conv2d_baens(in_planes, planes, kernel_size=1, stride=stride, bias=True, N=N),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class Wide_ResNet_baens(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, N):
        super(Wide_ResNet_baens, self).__init__()
        self.in_planes = 16

        self.N = N

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3_baens(3,nStages[0], N=N)
        self.layer1 = self._wide_layer(wide_basic_baens, nStages[1], n, dropout_rate, stride=1, N=N)
        self.layer2 = self._wide_layer(wide_basic_baens, nStages[2], n, dropout_rate, stride=2, N=N)
        self.layer3 = self._wide_layer(wide_basic_baens, nStages[3], n, dropout_rate, stride=2, N=N)
        self.bn1 = nn.BatchNorm2d(nStages[3])
        self.linear1 = modules.dense_baens(N=N, D1=nStages[3], D2=int(nStages[3]/2))
        self.bn2 = nn.BatchNorm1d(int(nStages[3]/2))
        self.linear2 = modules.dense_baens(N=N, D1=int(nStages[3]/2), D2=num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride, N=4):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride, N=N))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.training:
            assert x.shape[0] % self.N == 0
            x = x.view(int(x.shape[0]/self.N), int(self.N * x.shape[1]), *x.shape[2:])
        else:
            x = x.unsqueeze(1).expand(x.shape[0], self.N, *x.shape[1:])
            x = x.reshape(x.shape[0], int(x.shape[1] * x.shape[2]), *x.shape[3:])
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        # out = out.view(out.size(0), -1)
        out = out.view(int(out.shape[0]/self.N), self.N, -1)
        out = self.linear1(out)
        out = F.hardtanh(self.bn2(out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])), inplace=True)
        out = self.linear2(out.view(int(out.shape[0]/self.N), self.N, *out.shape[1:]))

        if self.training:
            out = out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])
        return out

def conv3x3_kv_baens(in_planes, out_planes, stride=1, N=4, bitwidth=3):
    return modules.Conv2d_res_bit_tree_meanvar_freeze_fine_partition_wkernel_kv_baens(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True, N=N, bw=bitwidth, first=True)

def conv_init_kv_baens(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d_res_bit') != -1:
        init.xavier_uniform_(m.K, gain=np.sqrt(2))
        init.xavier_uniform_(m.V, gain=np.sqrt(2))
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class wide_basic_kv_baens(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1, N=4, bitwidth=3):
        super(wide_basic_kv_baens, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = modules.Conv2d_res_bit_tree_meanvar_freeze_fine_partition_wkernel_kv_baens(in_planes, planes, kernel_size=3, padding=1, bias=True, N=N, bw=bitwidth)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = modules.Conv2d_res_bit_tree_meanvar_freeze_fine_partition_wkernel_kv_baens(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True, N=N, bw=bitwidth)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                modules.Conv2d_res_bit_tree_meanvar_freeze_fine_partition_wkernel_kv_baens(in_planes, planes, kernel_size=1, stride=stride, bias=True, N=N, bw=bitwidth),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class Wide_ResNet_kv_baens(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, N, bitwidth):
        super(Wide_ResNet_kv_baens, self).__init__()
        self.in_planes = 16

        self.N = N
        self.bitwidth = bitwidth
        self.tune = True

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3_kv_baens(3, nStages[0], N=N, bitwidth=bitwidth)
        self.bn0 = nn.BatchNorm2d(nStages[0])
        self.layer1 = self._wide_layer(wide_basic_kv_baens, nStages[1], n, dropout_rate, stride=1, N=N, bitwidth=bitwidth)
        self.layer2 = self._wide_layer(wide_basic_kv_baens, nStages[2], n, dropout_rate, stride=2, N=N, bitwidth=bitwidth)
        self.layer3 = self._wide_layer(wide_basic_kv_baens, nStages[3], n, dropout_rate, stride=2, N=N, bitwidth=bitwidth)
        self.bn1 = nn.BatchNorm2d(nStages[3])
        self.linear1 = modules.dense_res_bit_tree_meanvar_freeze_fine_partition_wkernel_kv_baens(N=N, D1=nStages[3], D2=int(nStages[3]/2))
        self.bn2 = nn.BatchNorm1d(int(nStages[3]/2))
        self.linear2 = modules.dense_res_bit_tree_meanvar_freeze_fine_partition_wkernel_kv_baens(N=N, D1=int(nStages[3]/2), D2=num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride, N=4, bitwidth=3):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride, N=N, bitwidth=bitwidth))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.training:
            assert x.shape[0] % self.N == 0
            x = x.view(int(x.shape[0]/self.N), int(self.N * x.shape[1]), *x.shape[2:])
        else:
            x = x.unsqueeze(1).expand(x.shape[0], self.N, *x.shape[1:])
            x = x.reshape(x.shape[0], int(x.shape[1] * x.shape[2]), *x.shape[3:])
        out = self.conv1(x)
        out = F.relu(self.bn0(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        # out = out.view(out.size(0), -1)
        out = out.view(int(out.shape[0]/self.N), self.N, -1)
        out = self.linear1(out)
        out = F.hardtanh(self.bn2(out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])), inplace=True)
        out = self.linear2(out.view(int(out.shape[0]/self.N), self.N, *out.shape[1:]))

        if self.training:
            out = out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])
        return out


#### resnet 18

class BasicBlock_baens(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, N=4):
        super(BasicBlock_baens, self).__init__()
        self.conv1 = modules.Conv2d_baens(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, N=N)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = modules.Conv2d_baens(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, N=N)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                modules.Conv2d_baens(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, N=N),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_baens(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, N=4):
        super(ResNet_baens, self).__init__()
        self.in_planes = 64
        self.N = N

        self.conv1 = modules.Conv2d_baens(3, 64, 
            kernel_size=3, stride=1, padding=1, bias=False, N=N, first=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, N=N)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, N=N)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, N=N)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, N=N)
        self.linear1 = modules.dense_baens(N=N, D1=512*block.expansion, D2=256*block.expansion)
        self.bn2 = nn.BatchNorm1d(256*block.expansion)
        self.linear2 = modules.dense_baens(N=N, D1=256*block.expansion, D2=num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, N=4):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, N=N))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.training:
            assert x.shape[0] % self.N == 0
            x = x.view(int(x.shape[0]/self.N), int(self.N * x.shape[1]), *x.shape[2:])
        else:
            x = x.unsqueeze(1).expand(x.shape[0], self.N, *x.shape[1:])
            x = x.reshape(x.shape[0], int(x.shape[1] * x.shape[2]), *x.shape[3:])

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)

        out = out.view(int(out.shape[0]/self.N), self.N, -1)
        out = self.linear1(out)
        out = F.relu(self.bn2(out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])), inplace=True)
        out = self.linear2(out.view(int(out.shape[0]/self.N), self.N, *out.shape[1:]))

        if self.training:
            out = out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])
        return out

def ResNet18(N=4):
    return ResNet_baens(BasicBlock_baens, [2, 2, 2, 2], N=N)


class BasicBlock_kv_baens(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, N=4, bitwidth=3):
        super(BasicBlock_kv_baens, self).__init__()
        self.conv1 = modules.Conv2d_res_bit_tree_meanvar_freeze_fine_partition_wkernel_kv_gn_baens(in_planes, planes, 
            kernel_size=3, stride=stride, padding=1, bias=False, N=N, bw=bitwidth)
        self.bn1 = nn.GroupNorm(N, int(N * planes))
        self.conv2 = modules.Conv2d_res_bit_tree_meanvar_freeze_fine_partition_wkernel_kv_gn_baens(planes, planes, 
            kernel_size=3, stride=1, padding=1, bias=False, N=N, bw=bitwidth)
        self.bn2 = nn.GroupNorm(N, int(N * planes))

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                modules.Conv2d_res_bit_tree_meanvar_freeze_fine_partition_wkernel_kv_gn_baens(in_planes, self.expansion*planes, 
                    kernel_size=1, stride=stride, bias=False, N=N, bw=bitwidth),
                nn.GroupNorm(N, int(N * self.expansion*planes))
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_kv_baens(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, N=4, bitwidth=3):
        super(ResNet_kv_baens, self).__init__()
        self.in_planes = 64
        self.N = N

        self.conv1 = modules.Conv2d_res_bit_tree_meanvar_freeze_fine_partition_wkernel_kv_gn_baens(3, 64, 
            kernel_size=3, stride=1, padding=1, bias=False, N=N, bw=bitwidth, first=True)
        self.bn1 = nn.GroupNorm(self.N, int(64*self.N))

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, N=N, bitwidth=bitwidth)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, N=N, bitwidth=bitwidth)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, N=N, bitwidth=bitwidth)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, N=N, bitwidth=bitwidth)
        self.linear1 = modules.dense_res_bit_tree_meanvar_freeze_fine_partition_wkernel_kv_gn_baens(N=N, bw=bitwidth, 
            D1=512*block.expansion, D2=256*block.expansion)
        self.bn_list = nn.ModuleList()
        for i in range(N):
            self.bn_list.append(nn.BatchNorm1d(256*block.expansion))
        self.linear2 = modules.dense_res_bit_tree_meanvar_freeze_fine_partition_wkernel_kv_gn_baens(N=N, bw=bitwidth, 
            D1=256*block.expansion, D2=num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, N=4, bitwidth=3):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, N=N, bitwidth=bitwidth))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.training or self.tune:
            assert x.shape[0] % self.N == 0
            x = x.view(int(x.shape[0]/self.N), int(self.N * x.shape[1]), *x.shape[2:])
        else:
            x = x.unsqueeze(1).expand(x.shape[0], self.N, *x.shape[1:])
            x = x.reshape(x.shape[0], int(x.shape[1] * x.shape[2]), *x.shape[3:])

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(int(out.shape[0]), self.N, -1)

        out = self.linear1(out)
        chunks = []
        for i in range(self.N):
            chunks.append(F.hardtanh(self.bn_list[i](out[:, i, :]), inplace=True))
        out = torch.stack(chunks, dim=1)
        out = self.linear2(out)

        # out = F.relu(self.bn2(out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])), inplace=True)
        # out = self.linear2(out.view(int(out.shape[0]/self.N), self.N, *out.shape[1:]))

        if self.training or self.tune:
            out = out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])
        return out

def ResNet18_kv(N=4, bitwidth=3, num_classes=10):
    return ResNet_kv_baens(BasicBlock_kv_baens, [2, 2, 2, 2], N=N, bitwidth=bitwidth, num_classes=num_classes)



class BasicBlock_full_ens(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, N=4, ):
        super(BasicBlock_full_ens, self).__init__()
        self.conv1 = modules.Conv2d_full_ens(in_planes, planes, 
            kernel_size=3, stride=stride, padding=1, N=N)
        self.bn1 = nn.GroupNorm(N, int(N * planes))
        self.conv2 = modules.Conv2d_full_ens(planes, planes, 
            kernel_size=3, stride=1, padding=1, N=N)
        self.bn2 = nn.GroupNorm(N, int(N * planes))

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                modules.Conv2d_full_ens(in_planes, self.expansion*planes, 
                    kernel_size=1, stride=stride, N=N),
                nn.GroupNorm(N, int(N * self.expansion*planes))
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_full_ens(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, N=4):
        super(ResNet_full_ens, self).__init__()
        self.in_planes = 64
        self.N = N

        self.conv1 = modules.Conv2d_full_ens(3, 64, 
            kernel_size=3, stride=1, padding=1, N=N, first=True)
        self.bn1 = nn.GroupNorm(self.N, int(64*self.N))

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, N=N)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, N=N)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, N=N)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, N=N)
        self.linear1 = modules.dense_full_ens(N=N, 
            D1=512*block.expansion, D2=256*block.expansion)
        self.bn_list = nn.ModuleList()
        for i in range(N):
            self.bn_list.append(nn.BatchNorm1d(256*block.expansion))
        self.linear2 = modules.dense_full_ens(N=N,
            D1=256*block.expansion, D2=num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, N=4):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, N=N))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.training:
            assert x.shape[0] % self.N == 0
            x = x.view(int(x.shape[0]/self.N), int(self.N * x.shape[1]), *x.shape[2:])
        else:
            x = x.unsqueeze(1).expand(x.shape[0], self.N, *x.shape[1:])
            x = x.reshape(x.shape[0], int(x.shape[1] * x.shape[2]), *x.shape[3:])

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(int(out.shape[0]), self.N, -1)

        out = self.linear1(out)
        chunks = []
        for i in range(self.N):
            chunks.append(F.hardtanh(self.bn_list[i](out[:, i, :]), inplace=True))
        out = torch.stack(chunks, dim=1)
        out = self.linear2(out)

        # out = F.relu(self.bn2(out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])), inplace=True)
        # out = self.linear2(out.view(int(out.shape[0]/self.N), self.N, *out.shape[1:]))

        if self.training:
            out = out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])
        return out

def ResNet18_full_ens(N=4, num_classes=10):
    return ResNet_full_ens(BasicBlock_full_ens, [2, 2, 2, 2], N=N, num_classes=num_classes)


class BasicBlock_baens_quant(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, N=4, ):
        super(BasicBlock_baens_quant, self).__init__()
        self.conv1 = modules.Conv2d_baens_quant(in_planes, planes, 
            kernel_size=3, stride=stride, padding=1, N=N)
        self.bn1 = nn.GroupNorm(N, int(N * planes))
        self.conv2 = modules.Conv2d_full_ens(planes, planes, 
            kernel_size=3, stride=1, padding=1, N=N)
        self.bn2 = nn.GroupNorm(N, int(N * planes))

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                modules.Conv2d_baens_quant(in_planes, self.expansion*planes, 
                    kernel_size=1, stride=stride, N=N),
                nn.GroupNorm(N, int(N * self.expansion*planes))
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_baens_quant(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, N=4):
        super(ResNet_baens_quant, self).__init__()
        self.in_planes = 64
        self.N = N

        self.conv1 = modules.Conv2d_baens_quant(3, 64, 
            kernel_size=3, stride=1, padding=1, N=N, first=True)
        self.bn1 = nn.GroupNorm(self.N, int(64*self.N))

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, N=N)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, N=N)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, N=N)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, N=N)
        self.linear1 = modules.dense_baens_quant(N=N, 
            D1=512*block.expansion, D2=256*block.expansion)
        self.bn_list = nn.ModuleList()
        for i in range(N):
            self.bn_list.append(nn.BatchNorm1d(256*block.expansion))
        self.linear2 = modules.dense_baens_quants(N=N,
            D1=256*block.expansion, D2=num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, N=4):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, N=N))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.training:
            assert x.shape[0] % self.N == 0
            x = x.view(int(x.shape[0]/self.N), int(self.N * x.shape[1]), *x.shape[2:])
        else:
            x = x.unsqueeze(1).expand(x.shape[0], self.N, *x.shape[1:])
            x = x.reshape(x.shape[0], int(x.shape[1] * x.shape[2]), *x.shape[3:])

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(int(out.shape[0]), self.N, -1)

        out = self.linear1(out)
        chunks = []
        for i in range(self.N):
            chunks.append(F.hardtanh(self.bn_list[i](out[:, i, :]), inplace=True))
        out = torch.stack(chunks, dim=1)
        out = self.linear2(out)

        # out = F.relu(self.bn2(out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])), inplace=True)
        # out = self.linear2(out.view(int(out.shape[0]/self.N), self.N, *out.shape[1:]))

        if self.training:
            out = out.reshape(int(out.shape[0] * out.shape[1]), *out.shape[2:])
        return out

def ResNet18_baens_quant(N=4, num_classes=10):
    return ResNet_full_ens(BasicBlock_baens_quant, [2, 2, 2, 2], N=N, num_classes=num_classes)

# class Bottleneck_baens(nn.Module):
#     expansion = 4

#     def __init__(self, in_planes, planes, stride=1, N=4):
#         super(Bottleneck_baens, self).__init__()
#         self.conv1 = modules.Conv2d_baens(in_planes, planes, 
#             kernel_size=1, stride=1, padding=0, bias=False, N=N)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = modules.Conv2d_baens(planes, planes, 
#             kernel_size=3, stride=stride, padding=1, bias=False, N=N)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = modules.Conv2d_baens(planes, self.expansion * planes, 
#             kernel_size=1, stride=1, padding=0, bias=False, N=N)
#         self.bn3 = nn.BatchNorm2d(self.expansion*planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 modules.Conv2d_baens(planes, self.expansion * planes, 
#                         kernel_size=1, stride=stride, padding=0, bias=False, N=N),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out




