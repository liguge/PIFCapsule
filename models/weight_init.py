import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import firwin

class Laplace_fast1(nn.Module):

    def __init__(self, out_channels, kernel_size, frequency, eps=0.3, mode='sigmoid'):
        super(Laplace_fast1, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.eps = eps
        self.mode = mode
        self.fre = frequency
        self.a_ = torch.linspace(0, self.out_channels, self.out_channels).view(-1, 1)
        self.b_ = torch.linspace(0, self.out_channels, self.out_channels).view(-1, 1)
        self.time_disc = torch.linspace(0, self.kernel_size - 1, steps=int(self.kernel_size))

    def bi_damped_Laplace(self, p):
        ep = 0.2
        q = torch.tensor(1 - pow(ep, 2))
        ep1 = 0.5
        q1 = torch.tensor(1 - pow(ep1, 2))
        tal = 0.05
        w = 2 * torch.pi * self.fre
        M = p - tal
        A1 = torch.where(M > 0, torch.ones_like(M), torch.zeros_like(M))
        A2 = torch.where(A1 == 0.0, torch.ones_like(A1), torch.zeros_like(A1))

        if self.mode == 'vanilla':
            return A1 * torch.exp(((-ep / (torch.sqrt(q))) * (w * M))) * (torch.cos(w * M)) - \
        A2 * torch.exp(((ep1 / (torch.sqrt(q1))) * (w * M))) * (torch.cos(w * M))

        if self.mode == 'maxmin':
            a = 1 * torch.exp(((-ep / (torch.sqrt(q))) * (w * M)).sigmoid()) * (torch.cos(w * M)) - \
        A2 * torch.exp(((ep1 / (torch.sqrt(q1))) * (w * M)).sigmoid()) * (torch.cos(w * M))
            return (a - a.min()) / (a.max() - a.min())

        if self.mode == 'sigmoid':
            return 1 * torch.exp(((-ep / (torch.sqrt(q))) * (w * M)).sigmoid()) * (torch.cos(w * M)) - \
        A2 * torch.exp(((ep1 / (torch.sqrt(q1))) * (w * M)).sigmoid()) * (torch.cos(w * M))

        if self.mode == 'softmax':
            return 1 * torch.exp(F.softmax((-ep / (torch.sqrt(q))) * (w * M))) * (torch.cos(w * M)) - \
        A2 * torch.exp(F.softmax((ep1 / (torch.sqrt(q1))) * (w * M))) * (torch.cos(w * M))

        if self.mode == 'tanh':
            return 1 * torch.exp(((-ep / (torch.sqrt(q))) * (w * M)).tanh()) * (torch.cos(w * M)) - \
        A2 * torch.exp(((ep1 / (torch.sqrt(q1))) * (w * M)).tanh()) * (torch.cos(w * M))

        if self.mode == 'atan':
            return 1 * torch.exp(((-ep / (torch.sqrt(q))) * (w * M)).atan()) * (torch.cos(w * M)) - \
        A2 * torch.exp(((ep1 / (torch.sqrt(q1))) * (w * M)).atan()) * (torch.cos(w * M))

    def forward(self):
        p1 = (self.time_disc - self.b_) / (self.a_ + self.eps)
        return self.bi_damped_Laplace(p1).view(self.out_channels, 1, self.kernel_size)


class STFT_fast(nn.Module):


    def __init__(self, out_channels, kernel_size, frequency=1000):
        super(STFT_fast, self).__init__()
        self.kernel_size = kernel_size if kernel_size % 2 == 0 else kernel_size - 1
        self.fs = frequency
        self.out_channels = out_channels
        self.H = torch.zeros((self.out_channels, self.kernel_size), dtype=torch.float32)

    def forward(self):
        bw = 0.1 * self.fs / self.kernel_size
        filters = [torch.from_numpy(firwin(self.kernel_size, [bw * i + 0.01, bw * (i + 1) - 0.01], window='blackman', pass_zero=False, fs=self.fs)) for i in range(self.out_channels)]
        self.H = torch.stack(filters)
        self.output = self.H.unsqueeze(0).swapaxes(0, 1).float()
        mean = self.output.mean(dim=-1, keepdim=True)
        std = self.output.std(dim=-1, keepdim=True) + 1e-8
        self.output = (self.output - mean) / std
        return self.output



