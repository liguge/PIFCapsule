import torch
import torch.nn as nn
import torch.nn.functional as F

class Squash(nn.Module):
    def __init__(self, eps=1e-20):
        super(Squash, self).__init__()
        self.eps = eps

    def forward(self, x):
        norm = torch.linalg.norm(x, ord=2, dim=-1, keepdim=True)
        coef = 1 - 1 / (torch.exp(norm) + self.eps)
        unit = x / (norm + self.eps)
        return coef * unit


class PrimaryCaps(nn.Module):
    def __init__(
        self,
        in_channels,
        kernel_size,
        capsule_size,
        stride=1,
    ):
        super(PrimaryCaps, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.num_capsules, self.dim_capsules = capsule_size
        self.stride = stride

        self.dw_conv2d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=self.num_capsules * self.dim_capsules,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.squash = Squash()

    def forward(self, x):
        x = self.dw_conv2d(x)
        x = self.pool(x)
        x = x.view(-1, self.num_capsules, self.dim_capsules) # reshape  3920,16,8
        return self.squash(x)

class RoutingCaps1(nn.Module):
    def __init__(self, in_capsules, out_capsules, focusing_factor=3):
        super(RoutingCaps1, self).__init__()
        self.N0, self.D0 = in_capsules
        self.N1, self.D1 = out_capsules
        self.squash = Squash()
        self.focusing_factor = focusing_factor
        self.kernel_size = 65  #####9改为65
        # initialize routing parameters
        self.W = nn.Parameter(torch.Tensor(self.N1, self.N0, self.D0, self.D1))
        nn.init.kaiming_normal_(self.W)
        self.b = nn.Parameter(torch.zeros(self.N1, self.N0, 1))
        self.dwc = nn.Conv1d(in_channels=self.N1*self.D1, out_channels=self.N1*self.D1, kernel_size=self.kernel_size,
                             groups=self.N1, padding=self.kernel_size // 2)


    def forward(self, x1, x2, x3):

        u1 = (torch.einsum("...ji,kjiz->...kjz", x1, self.W).permute(0, 1, 3, 2).contiguous().view(-1, self.N1*self.D1, self.N0))
        u2 = torch.einsum("...ji,kjiz->...kjz", x2, self.W).permute(0, 1, 3, 2).contiguous().view(-1, self.N1 * self.D1,
                                                                                                 self.N0)
        u3 = torch.einsum("...ji,kjiz->...kjz", x3, self.W).permute(0, 1, 3, 2).contiguous().view(-1, self.N1 * self.D1,
                                                                                                 self.N0)
        focusing_factor = 3.0
        kernel_function = nn.ReLU()
        scale = 5.0
        ################################################################
        u1 = kernel_function(u1) + 1e-6
        u1 = u1 / scale
        u_norm = u1.norm(dim=-1, keepdim=True)
        u1 = u1 ** focusing_factor
        u1 = (u1 / u1.norm(dim=-1, keepdim=True)) * u_norm
        #############################################################
        u2 = kernel_function(u2) + 1e-6
        u2 = u2 / scale
        u_norm = u2.norm(dim=-1, keepdim=True)
        u2 = u2 ** focusing_factor
        u2 = (u2 / u2.norm(dim=-1, keepdim=True)) * u_norm
        u3 = kernel_function(u3) + 1e-6
        u3 = u3 / scale
        u_norm = u3.norm(dim=-1, keepdim=True)
        u3 = u3 ** focusing_factor
        u3 = (u3 / u3.norm(dim=-1, keepdim=True)) * u_norm

        z = (1 / (torch.einsum("b i c, b c -> b i", u1, u1.sum(dim=1)) + torch.finfo(u1.dtype).eps))
        kv = torch.einsum("b j c, b j d -> b c d", u2, u3)
        x = torch.einsum("b i c, b c d, b i -> b i d", u1, kv, z)
        x = (x + self.dwc(x)).sum(dim=-1)
        x = x.view(-1, self.N1, self.D1)

        return self.squash(x)
##################################################################################################
class RoutingCaps(nn.Module):
    def __init__(self, in_capsules, out_capsules, focusing_factor=3, class_number=9):
        super(RoutingCaps, self).__init__()
        self.N0, self.D0 = in_capsules
        self.N1, self.D1 = out_capsules
        self.squash = Squash()
        self.focusing_factor = focusing_factor
        self.kernel_size = 65  #####9改为65
        self.class_number = class_number
        self.l1 = nn.Linear(16*8, self.class_number*16*16, bias=False)
        self.l2 = nn.Linear(16*8, self.class_number*16*16, bias=False)
        self.l3 = nn.Linear(16*8, self.class_number*16*16, bias=False)


        self.dwc = nn.Conv1d(in_channels=self.N1*self.D1, out_channels=self.N1*self.D1, kernel_size=self.kernel_size,
                             groups=self.N1, padding=self.kernel_size // 2)

    def forward(self, x1, x2, x3):
        ########################################################
        u11 = self.l1(x1.view(-1, 16*8)).view(-1, self.class_number, 16, 16)
        u22 = self.l2(x2.view(-1, 16*8)).view(-1, self.class_number, 16, 16)
        u33 = self.l3(x3.view(-1, 16*8)).view(-1, self.class_number, 16, 16)
        u1 = u11.permute(0, 1, 3, 2).contiguous().view(-1, self.N1*self.D1, self.N0)
        u2 = u22.permute(0, 1, 3, 2).contiguous().view(-1, self.N1*self.D1, self.N0)
        u3 = u33.permute(0, 1, 3, 2).contiguous().view(-1, self.N1*self.D1, self.N0)
        ##########################################################################
        focusing_factor = 3.0
        kernel_function = nn.GELU()
        scale = 5.0
        ################################################################
        u1 = torch.abs(kernel_function(u1) + 1e-6)
        u1 = u1 / scale
        u_norm = u1.norm(dim=-1, keepdim=True)
        u1 = u1 ** focusing_factor
        u1 = (u1 / u1.norm(dim=-1, keepdim=True)) * u_norm
        #############################################################
        u2 = torch.abs(kernel_function(u2) + 1e-6)
        u2 = u2 / scale
        u_norm = u2.norm(dim=-1, keepdim=True)
        u2 = u2 ** focusing_factor
        u2 = (u2 / u2.norm(dim=-1, keepdim=True)) * u_norm
        u3 = torch.abs(kernel_function(u3) + 1e-6)
        u3 = u3 / scale
        u_norm = u3.norm(dim=-1, keepdim=True)
        u3 = u3 ** focusing_factor
        u3 = (u3 / u3.norm(dim=-1, keepdim=True)) * u_norm


        z = (1 / (torch.einsum("b i c, b c -> b i", u1, u1.sum(dim=1)) + torch.finfo(u1.dtype).eps))
        kv = torch.einsum("b j c, b j d -> b c d", u2, u3)
        x = torch.einsum("b i c, b c d, b i -> b i d", u1, kv, z)

        x = (x + self.dwc(x)).sum(dim=-1)
        x = x.view(-1, self.N1, self.D1)

        ###########################################################################
        x = x.unsqueeze(-1)

        s = torch.sum((u11+u22+u33) * x, dim=-2)
        return self.squash(s)
#################################################################################################################

############################################################################################################################################################################
class CapsLen(nn.Module):
    def __init__(self, eps=1e-7):
        super(CapsLen, self).__init__()
        self.eps = eps

    def forward(self, x):
        return torch.sqrt(
            torch.sum(x**2, dim=-1) + self.eps
        )  # (batch_size, num_capsules)


class CapsMask(nn.Module):
    def __init__(self):
        super(CapsMask, self).__init__()

    def forward(self, x, y_true=None):
        if y_true is not None:  # training mode
            mask = y_true
        else:  # testing mode
            # convert list of maximum value's indices to one-hot tensor
            temp = torch.sqrt(torch.sum(x**2, dim=-1))
            mask = F.one_hot(torch.argmax(temp, dim=1), num_classes=temp.shape[1])
        masked = x * mask.unsqueeze(-1)
        return masked.view(x.shape[0], -1) # reshape
