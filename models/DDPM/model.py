
def get_timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        # Add timestep embedding
        time_emb = self.time_mlp(t).view(t.shape[0], -1, 1, 1)
        h = h + time_emb
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.res_conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        # reshape for multi-head
        q = q.view(B, self.num_heads, self.head_dim, H * W)
        k = k.view(B, self.num_heads, self.head_dim, H * W)
        v = v.view(B, self.num_heads, self.head_dim, H * W)
        # attention
        attn = torch.einsum("bhdn,bhdm->bhnm", q, k) * (self.head_dim ** -0.5)
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum("bhnm,bhdm->bhdn", attn, v)
        # Use reshape to handle non-contiguous tensors
        out = out.reshape(B, C, H, W)
        out = self.proj(out)
        return x + out


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 128,
        channel_mults: tuple = (1, 2, 4),
        time_embed_dim: int = 128,
        time_mlp_dim: int = 512,
        num_heads: int = 4,
    ):
        super().__init__()
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_mlp_dim),
            nn.SiLU(),
            nn.Linear(time_mlp_dim, time_mlp_dim),
        )
        # Initial conv
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        # Downsampling path
        channels = [base_channels * m for m in channel_mults]  # [128, 256, 512]
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        prev_ch = base_channels
        for ch in channels:
            self.down_blocks.append(nn.ModuleList([
                ResidualBlock(prev_ch, ch, time_mlp_dim),
                ResidualBlock(ch, ch, time_mlp_dim),
            ]))
            self.downsamples.append(Downsample(ch))
            prev_ch = ch
        # Bottleneck
        self.mid_blocks = nn.ModuleList([
            ResidualBlock(prev_ch, prev_ch * 2, time_mlp_dim),
            AttentionBlock(prev_ch * 2, num_heads),
            ResidualBlock(prev_ch * 2, prev_ch, time_mlp_dim),
        ])
        # Upsampling path
        self.up_blocks = nn.ModuleList()
        for ch in reversed(channels):
            self.up_blocks.append(nn.ModuleList([
                ResidualBlock(prev_ch + ch, ch, time_mlp_dim),
                ResidualBlock(ch, ch, time_mlp_dim),
                AttentionBlock(ch, num_heads) if ch >= 256 else nn.Identity(),
            ]))
            prev_ch = ch
        # Final normalization and conv
        self.out_norm = nn.GroupNorm(32, base_channels)
        self.out_conv = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        # x: [B, 4, 22, 22], timesteps: [B]
        # Time embedding
        t_emb = get_timestep_embedding(timesteps, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)
        # Initial conv
        h = self.init_conv(x)
        # Downsample
        skips = []
        for (block1, block2), down in zip(self.down_blocks, self.downsamples):
            h = block1(h, t_emb)
            h = block2(h, t_emb)
            skips.append(h)
            h = down(h)
        # Bottleneck
        h = self.mid_blocks[0](h, t_emb)
        h = self.mid_blocks[1](h)
        h = self.mid_blocks[2](h, t_emb)
        # Upsample
        for (block1, block2, attn), skip in zip(self.up_blocks, reversed(skips)):
            h = F.interpolate(h, size=skip.shape[-2:], mode='nearest')
            h = torch.cat([h, skip], dim=1)
            h = block1(h, t_emb)
            h = block2(h, t_emb)
            h = attn(h)
        # Final conv
        h = self.out_norm(h)
        h = F.silu(h)
        return self.out_conv(h)

class DDPM(nn.Module):
    def __init__(self, network, 
                 n_steps=1000,
                 min_beta=0.0001, 
                 max_beta=1,
                 device=None,
                 image_chw=(4, 22, 22)):
        super().__init__()
        self.n_steps = n_steps
        self.device = device
        self.image_chw = image_chw
        self.network = network.to(device) 
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        self.alphas = (1. - self.betas).to(device)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(device)

    def forward(self, x0, t, eta=None):
        if eta is None:
            eta = torch.randn(size=x0.shape, device=device, requires_grad=False) 

        noised_x = self.alpha_bars[t]**0.5 * x0 + (1 - self.alpha_bars[t])**0.5 * eta
        return noised_x

    def backward(self, x, t):
        eta_pred = self.network(x, t)
        return eta_pred
