import torch
import torch.nn as nn
import math
class MarginLoss(nn.Module):
    def __init__(self, m_pos=0.9, m_neg=0.1, lambda_=0.5):
        super(MarginLoss, self).__init__()
        self.m_pos = m_pos
        self.m_neg = m_neg
        self.lambda_ = lambda_

    def forward(self, targets, digit_probs):
        assert targets.shape is not digit_probs.shape
        present_losses = (
            targets * torch.clamp_min(self.m_pos - digit_probs, min=0.0) ** 2
        )
        absent_losses = (1 - targets) * torch.clamp_min(
            digit_probs - self.m_neg, min=0.0
        ) ** 2
        losses = present_losses + self.lambda_ * absent_losses
        return torch.mean(torch.sum(losses, dim=1))
def hilbert(x):
    N = x.shape[-1]
    # Xf = torch.fft.fft(x)
    if (N % 2 == 0):
        x[..., 1: N // 2] *= 2
        x[..., N // 2 + 1:] = 0
    else:
        x[..., 1: (N + 1) // 2] *= 2
        x[..., (N + 1) // 2:] = 0
    return torch.fft.ifft(x)
def get_envelope_frequency(x, fs, ret_analytic=False, **kwargs):
    analytic = hilbert(x)
    envelope = analytic.abs()
    en_raw_abs = envelope - torch.mean(envelope)
    es_raw = torch.fft.fft(en_raw_abs)
    es_raw_abs = torch.abs(es_raw) * 2 / (len(en_raw_abs)+1e15)
    sub = torch.cat((analytic[..., 1:] - analytic[..., :-1],
                     (analytic[..., -1] - analytic[..., -2]).unsqueeze(-1)
                     ), axis=-1)
    add = torch.cat((analytic[..., 1:] + analytic[..., :-1],
                     2 * analytic[..., -1].unsqueeze(-1)
                     ), axis=-1)
    freq = 2 * fs * ((sub / (add+1e15)).imag)
    freq[freq.isinf()] = 0
    del sub, add
    freq /= (2 * math.pi)
    return (es_raw_abs, freq) if not ret_analytic else (envelope, freq, analytic)
class ReconstructionLoss(nn.Module):
    def forward(self, reconstructions, input_images):
        return torch.nn.MSELoss(reduction="mean")(reconstructions, input_images)

class G_lplq(nn.Module):
    def __init__(self, p=2.0, q=4.0):
        super(G_lplq, self).__init__()
        self.p = torch.tensor(p)
        self.q = torch.tensor(q)

    def forward(self, reconstructions):
        return torch.sign(torch.log(self.q/self.p))*(torch.norm(reconstructions, self.p)/(torch.norm(reconstructions, self.q)+1e-15)) ** self.p
class TotalLoss(nn.Module):
    def __init__(self, m_pos=0.9, m_neg=0.1, lambda_=0.5, recon_factor=0.0005, bd_factor=0.000):  #0.001
        super(TotalLoss, self).__init__()
        self.margin_loss = MarginLoss(m_pos, m_neg, lambda_)
        self.recon_loss = ReconstructionLoss()
        self.recon_factor = recon_factor#0.0005
        self.bd = G_lplq(p=2.0, q=4.0)
        self.bd_factor = bd_factor
        self.fs = 64000
    def forward(self, input_images, targets, reconstructions, digit_probs):
        margin = self.margin_loss(targets, digit_probs)
        recon = self.recon_loss(reconstructions, input_images)
        f1 = torch.fft.fft2(reconstructions - torch.mean(reconstructions, dim=-1).unsqueeze(1))
        es = abs(f1)
        es_raw_abs, _ = get_envelope_frequency(es, self.fs)
        bd_loss = self.bd(es_raw_abs)
        ############################################################################################
        return margin, recon, bd_loss  # + + self.bd_factor * k.mean() #
        ###########################################################################################