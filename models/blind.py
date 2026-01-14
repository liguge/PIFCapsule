import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_gpu = torch.cuda.is_available()

class Parabolic_Cone(nn.Module):
    def __init__(self):
         super(Parabolic_Cone, self).__init__()
    def forward(self, x):
        return torch.mul(x, 2-x)


class Shrinkagev3p(nn.Module):
    def __init__(self, gap_size, channel):
        super(Shrinkagev3p, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(gap_size)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_abs = x.abs()
        x = self.gap(x)
        x = torch.flatten(x, 1)
        average = x    #CW
        x = self.fc(x)
        x = torch.mul(average, x).unsqueeze(2)
        x = x_abs - x
        x = torch.min(x, torch.zeros_like(x))


        return x

class CLASSBD(nn.Module):
    def __init__(self, threshold=0.6, halffilterL=16) -> object:
        super(CLASSBD, self).__init__()
        self.threshold = threshold
        self.qtfilter = nn.Sequential(
            nn.Conv1d(1, 64, 64, 1, 'same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 1, 64, 1, 'same'),
            nn.BatchNorm1d(1),
            nn.ReLU(),
            nn.Sigmoid()   #202412041111
        )
        self.halffilterL = halffilterL
        self.noise = Shrinkagev3p(1, 1)
        self.filter1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)


    def PNAR(self, y_2, z):
        z1 = torch.abs(z)
        M = torch.tensor(y_2.size(-1))
        y_3 = y_2 * z1
        y_4 = torch.abs(y_3) / torch.sqrt(M)
        norm_y_2 = torch.norm(y_2, 2)
        loss = y_4 / (norm_y_2 + torch.finfo(y_2.dtype).eps)
        return loss.mean()

    def forward(self, x):

        a1 = self.qtfilter(x)
        a2 = a1
        a3 = a2
        a2 = torch.squeeze(a2, dim=1)
        a2 = a2[:, self.halffilterL:-self.halffilterL]
        a2 = a2 - torch.mean(a2)
        z = self.noise(a2.unsqueeze(1)).squeeze()
        g = self.PNAR(a2, z)
        return a3, g


if __name__ == "__main__":
    import os

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    inputs = torch.randn(5, 1, 2048).cuda()
    model = CLASSBD(halffilterL=int(64/2)).cuda()
    outputs = model(inputs)
    print(outputs[0].size())
    print(outputs[1].size())