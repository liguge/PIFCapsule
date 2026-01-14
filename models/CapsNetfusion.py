import torch
import torch.nn as nn
from models.layers import CapsLen, CapsMask, PrimaryCaps, RoutingCaps
from models.weight_init import Laplace_fast1, STFT_fast
from models.blind import CLASSBD
from fft_conv_pytorch import FFTConv1d


class Stan(nn.Module):
    def __init__(self, input_dim):
        super(Stan, self).__init__()
        self.beta = nn.Parameter(torch.ones(input_dim))
    def forward(self, x):
        return F.tanh(x) + self.beta * x * F.tanh(x)
class EfficientCapsNet(nn.Module):
    def __init__(self, input_size=(1, 28, 28), num_class=4):
        super(EfficientCapsNet, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_size[0], out_channels=64, kernel_size=64, padding=542, stride=4)
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.conv2 = FFTConv1d(in_channels=input_size[0], out_channels=64, kernel_size=64, padding=542, stride=4, bias=True)

        self.bn2 = nn.BatchNorm1d(64)

        self.bd = CLASSBD(halffilterL=int(64/2))
        self.conv4 = nn.Conv1d(
            in_channels=input_size[0], out_channels=64, kernel_size=64, padding=542, stride=4)
        self.bn4 = nn.BatchNorm1d(64)
        self.primary_caps = PrimaryCaps(in_channels=64, kernel_size=9, capsule_size=(16, 8))
        self.routing_caps = RoutingCaps(in_capsules=(16, 8), out_capsules=(num_class, 16), class_number=num_class)
        self.len_final_caps = CapsLen()
        self.reset_parameters()

    def reset_parameters(self):

        for name, can in self.named_children():
            if name == 'conv1':
                for m in can.modules():
                    if isinstance(m, nn.Conv1d):
                        if m.kernel_size == (64,):   #将这个值改为别的数字，查看Laplace的效果。
                            m.weight.data = Laplace_fast1(out_channels=64, kernel_size=64, eps=0.1, frequency=100000,    #20480,25600
                                                         mode='sigmoid').forward()
                            nn.init.constant_(m.bias.data, 0.0)
            elif name == 'conv2':
                for m in can.modules():
                    if m.kernel_size == 64:
                        m.weight.data = STFT_fast(out_channels=64, kernel_size=64,
                                                  frequency=64000).forward()  # 20480,25600
                        nn.init.constant_(m.bias.data, 0.0)

            else:
                for m in can.modules():
                    if isinstance(m, nn.Conv1d):
                        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        """Initialize parameters with Kaiming normal distribution."""

    def Intelligent_spectrogram_loss(self, x):
        x = x.abs()
        c_m = x.mean(dim=-2) / (x.std(dim=-2) + torch.finfo(x.dtype).eps)
        c_k = x.mean(dim=-1) / (x.std(dim=-1) + torch.finfo(x.dtype).eps)
        q_f = (c_m / (c_m.max() + torch.finfo(x.dtype).eps)).mean()
        q_t = (c_k / (c_k.max() + torch.finfo(x.dtype).eps)).mean()
        return 2.0 * q_f * q_t / (q_f + q_t)

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = self.conv2(x)
        loss2 = 0.5*self.Intelligent_spectrogram_loss(x1 + x2)
        x2 = F.relu(self.bn2(x2))
        x3, g = self.bd(x)
        x3 = F.relu(self.bn4(self.conv4(x3)))

        x1 = self.primary_caps(x1)
        x2 = self.primary_caps(x2)
        x3 = self.primary_caps(x3)
        x = self.routing_caps(x1, x2, x3)
        return x, self.len_final_caps(x), loss2, g






class ReconstructionNet(nn.Module):
    def __init__(self, input_size=(1, 2048), num_classes=5, num_capsules=16):   ###需要修改
        super(ReconstructionNet, self).__init__()
        self.input_size = input_size
        # self.fc1 = nn.Linear(num_classes*num_capsules, 128)
        self.fc1 = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=num_classes*num_capsules-63, output_padding=1)
        self.bn1 = nn.BatchNorm1d(1)
        self.fc2 = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm1d(1)
        self.fc3 = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm1d(1)
        self.fc4 = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm1d(1)
        self.fc5 = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1,
                                      output_padding=1)


    def forward(self, x):

        x = F.relu((self.fc1(x.unsqueeze(1))))
        x = F.relu((self.fc2(x)))
        x = F.relu((self.fc3(x)))
        x = F.relu((self.fc4(x)))
        x = F.sigmoid(self.fc5(x))
        return x.view(-1, *self.input_size)  # reshape

class FinalCapsNet(nn.Module):
    def __init__(self, num_class=4):
        super(FinalCapsNet, self).__init__()
        self.efficient_capsnet = EfficientCapsNet(num_class=num_class)
        self.mask = CapsMask()
        self.generator = ReconstructionNet(num_classes=num_class)
        self.loss_scale1= nn.Parameter(torch.tensor([0.01]))
        self.loss_scale2 = nn.Parameter(torch.tensor([0.01]))
        self.loss_scale3 = nn.Parameter(torch.tensor([0.01]))
        self.loss_scale4 = nn.Parameter(torch.tensor([0.01]))


    def UW(self, loss1, loss2, loss3, loss4):
        loss1 = loss1 / (self.loss_scale1.exp() + torch.finfo(loss1.dtype).eps) + self.loss_scale1.abs()
        loss2 = loss2 / (self.loss_scale2.exp() + torch.finfo(loss2.dtype).eps) + self.loss_scale2.abs()
        loss3 = loss3 / (self.loss_scale3.exp() + torch.finfo(loss3.dtype).eps) + self.loss_scale3.abs()
        loss4 = loss4 / (self.loss_scale4.exp() + torch.finfo(loss4.dtype).eps) + self.loss_scale3.abs()

        return (loss1 + loss2 + loss3 + loss4).mean()

    def forward(self, x, y_true=None, mode='train'):
        x, x_len, loss2, g = self.efficient_capsnet(x)
        if mode == "train":
            masked = self.mask(x, y_true)
        elif mode == "val":
            masked = self.mask(x)   #3920,160
        x = self.generator(masked)   #3920,1,28,28
        return x, x_len, loss2, g

if __name__ == "__main__":

    from utils.losses import TotalLoss
    import torch.nn.functional as F
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.long).cuda()
    inputs = torch.randn(4, 1, 2048).cuda()
    model = FinalCapsNet().cuda()
    criterion = TotalLoss(recon_factor=0.000, bd_factor=0.001)  # 胶囊网络 0.0005
    out_inputs, out_labels, loss2, g = model(inputs, None, mode='val')
    loss = model.UW(1 * loss2.sum(), 1e4 * g.sum(),0, 0) + 0
    print("done!")
