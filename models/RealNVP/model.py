import torch
import torch.nn as nn
import torch.distributions as D

class RealNVP(nn.Module):
    def __init__(self, device='cuda'):
        super(RealNVP, self).__init__()

        pixels = 4 * 22 * 22
        self.prior = D.MultivariateNormal(torch.zeros(pixels).to(device), torch.eye(pixels).to(device))
        total_dim = 4 * 22 * 22
        d = total_dim//2

        mask_even = torch.zeros(total_dim)
        mask_even[:d] = 1
        mask_odd = torch.zeros(total_dim)
        mask_odd[d:] = 1

        self.mask_even = nn.Parameter(mask_even, requires_grad=False)
        self.mask_odd = nn.Parameter(mask_odd, requires_grad=False)

        def create_mask(num):
            masks = []
            for i in range(num):
              if i % 2 == 0:
                masks.append(self.mask_even)
              else:
                masks.append(self.mask_odd)
            return masks

        self.mask = create_mask(6)

        ts = lambda: nn.Sequential(
            nn.Linear(1936, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1936),
            nn.Tanh())

        tt = lambda: nn.Sequential(
            nn.Linear(1936, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1936))

        self.t = torch.nn.ModuleList([tt() for _ in range(len(self.mask))])
        self.s = torch.nn.ModuleList([ts() for _ in range(len(self.mask))])

    def g(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x*self.mask[i]
            s = self.s[i](x_)*(1 - self.mask[i])
            t = self.t[i](x_)*(1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def f(self, x):
        x = x.view(x.shape[0], -1)
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1-self.mask[i])
            t = self.t[i](z_) * (1-self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J

    def log_prob(self,x):
        z, logp = self.f(x)
        return self.prior.log_prob(z) + logp

    def sample(self, batchSize):
        z = self.prior.sample((batchSize,))
        x = self.g(z)
        tanh = nn.Tanh()
        return tanh(x.view(batchSize, 4, 22, 22))
