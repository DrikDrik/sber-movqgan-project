import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()
        query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, H * W)
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        value = self.value(x).view(batch_size, -1, H * W)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        return x + self.gamma * out

class Generator(nn.Module):
    def __init__(self, latent_dim=128, n_groups=8):
        super(Generator, self).__init__()
        self.fc = nn.Linear(latent_dim, 512 * 11 * 11)
        self.gn1 = nn.GroupNorm(n_groups, 512)
        self.conv_transpose = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.gn2 = nn.GroupNorm(n_groups, 256)
        self.attention1 = SelfAttention(256)
        self.gn3 = nn.GroupNorm(n_groups, 256)
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.gn4 = nn.GroupNorm(n_groups, 128)
        self.attention2 = SelfAttention(128)
        self.gn5 = nn.GroupNorm(n_groups, 128)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.gn6 = nn.GroupNorm(n_groups, 64)
        self.conv3 = nn.Conv2d(64, 4, kernel_size=3, padding=1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 512, 11, 11)
        x = F.leaky_relu(self.gn1(x), negative_slope=0.2)
        x = F.leaky_relu(self.gn2(self.conv_transpose(x)), negative_slope=0.2)  # -> 22x22
        x = self.gn3(self.attention1(x))
        x = F.leaky_relu(self.gn4(self.conv1(x)), negative_slope=0.2)
        x = self.gn5(self.attention2(x))
        x = F.leaky_relu(self.gn6(self.conv2(x)), negative_slope=0.2)
        x = torch.tanh(self.conv3(x))
        return x

class Critic(nn.Module):
    def __init__(self, n_groups=8):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=2, padding=1)
        self.gn1 = nn.GroupNorm(n_groups, 64)
        self.attention1 = SelfAttention(64)
        self.gn2 = nn.GroupNorm(n_groups, 64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.gn3 = nn.GroupNorm(n_groups, 128)
        self.attention2 = SelfAttention(128)
        self.gn4 = nn.GroupNorm(n_groups, 128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.gn5 = nn.GroupNorm(n_groups, 256)
        self.fc1 = nn.Linear(256 * 3 * 3, 128)
        self.gn6 = nn.GroupNorm(n_groups, 128)
        self.fc2 = nn.Linear(128, 1)


    def forward(self, x):
        x = F.leaky_relu(self.gn1(self.conv1(x)), 0.2)
        x = self.gn2(self.attention1(x))
        x = F.leaky_relu(self.gn3(self.conv2(x)), 0.2)
        x = self.gn4(self.attention2(x))
        x = F.leaky_relu(self.gn5(self.conv3(x)), 0.2)
        x = x.view(-1, 256 * 3 * 3)
        x = F.leaky_relu(self.gn6(self.fc1(x)), 0.2)
        x = self.fc2(x)
        return x

def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    interpolated.requires_grad_(True)
    pred = critic(interpolated)
    gradients = torch.autograd.grad(outputs=pred, inputs=interpolated,
                                    grad_outputs=torch.ones(pred.size(), device=device),
                                    create_graph=True, retain_graph=True)[0]
    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, dim=1)
    gp = ((grad_norm - 1) ** 2).mean()
    return gp

def generate_new_codes(generator, latent_dim, n_samples=16, c=4, h=22, w=22, device="cpu"):
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(n_samples, latent_dim, device=device) 
        generated_codes = generator(noise)
    generator.train()
    return generated_codes
