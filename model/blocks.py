import torch
import torch.nn as nn

def pad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p

class ConvBlock(torch.nn.Module):
    """Conv -> BN -> SiLU"""
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, d=1, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, pad(k, p, d), d, g, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_ch, 0.001, 0.03)
        self.act = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(torch.nn.Module):
    def __init__(self, ch, add=True):
        super().__init__()
        self.add_residual = add
        self.blocks = torch.nn.Sequential(
            ConvBlock(ch, ch, 3),
            ConvBlock(ch, ch, 3)
        )

    def forward(self, x):
        return self.blocks(x) + x if self.add_residual else self.blocks(x)


class CSPBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n=1, add=True):
        super().__init__()
        self.split_conv = ConvBlock(in_ch, out_ch // 2)
        self.path_conv = ConvBlock(in_ch, out_ch // 2)
        self.final_conv = ConvBlock((2 + n) * out_ch // 2, out_ch)
        self.residual_blocks = torch.nn.ModuleList([ResidualBlock(out_ch // 2, add) for _ in range(n)])

    def forward(self, x):
        left = self.split_conv(x)
        right = self.path_conv(x)
        out_list = [left, right]
        out_list.extend(block(out_list[-1]) for block in self.residual_blocks)
        return self.final_conv(torch.cat(out_list, dim=1))


class SPPBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        self.reduce_conv = ConvBlock(in_ch, in_ch // 2)
        self.final_conv = ConvBlock(in_ch * 2, out_ch)
        self.maxpool = torch.nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x):
        x = self.reduce_conv(x)
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)
        return self.final_conv(torch.cat([x, y1, y2, y3], 1))


class RepViTBlock(nn.Module):
    """
    RepViT Block: Re-parameterized Vision Transformer block for mobile devices
    Based on: "Revisiting Mobile CNN From ViT Perspective"
    """
    def __init__(self, dim, kernel_size=3, mlp_ratio=2., use_se=False, se_ratio=0.25):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.use_se = use_se
        
        # MBConv-style structure with reparameterization
        self.token_mixer = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, 1, kernel_size//2, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU6(inplace=True) if not use_se else nn.Identity(),
        )
        
        # Channel mixer (MLP replacement)
        hidden_dim = int(dim * mlp_ratio)
        self.channel_mixer = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(dim),
        )
        
        # Squeeze-and-Excitation (optional)
        if use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim, int(dim * se_ratio), 1, 1, 0),
                nn.ReLU6(inplace=True),
                nn.Conv2d(int(dim * se_ratio), dim, 1, 1, 0),
                nn.Sigmoid()
            )
        
        # Re-parameterization branches
        self.reparam_branches = nn.ModuleList()
        
        # Add 1x1 branch for reparameterization
        self.reparam_branches.append(nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0, groups=dim, bias=False),
            nn.BatchNorm2d(dim)
        ))
        
        # Add identity branch for reparameterization
        self.reparam_branches.append(nn.BatchNorm2d(dim))
        
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        # Training: use multi-branch structure
        if self.training:
            # Token mixer with multiple branches
            out = 0
            for branch in self.reparam_branches:
                out += branch(x)
            out = self.token_mixer[1](out)  # BN only
            if len(self.token_mixer) > 2:  # Activation if present
                out = self.token_mixer[2](out)
        else:
            # Inference: use single path (will be reparameterized)
            out = self.token_mixer(x)
        
        # Squeeze-and-Excitation
        if self.use_se:
            out = out * self.se(out)
        
        # Channel mixer
        out = out + self.channel_mixer(out)
        
        return self.act(out)

    def reparameterize(self):
        """
        Convert multi-branch architecture to single branch for inference
        This should be called after training and before deployment
        """
        if not self.training:
            return
            
        # Re-parameterize token mixer
        kernel, bias = self._fuse_bn(self.token_mixer[0], self.token_mixer[1])
        
        # Fuse with reparameterization branches
        for branch in self.reparam_branches:
            if isinstance(branch, nn.Sequential):
                branch_kernel, branch_bias = self._fuse_bn(branch[0], branch[1])
            else:  # Identity branch (just BN)
                branch_kernel = torch.eye(self.dim).view(self.dim, self.dim, 1, 1)
                branch_bias = -branch.running_mean * branch.weight / torch.sqrt(branch.running_var + branch.eps) + branch.bias
                
            kernel += branch_kernel
            bias += branch_bias
        
        # Create fused convolution
        fused_conv = nn.Conv2d(
            self.dim, self.dim, 
            kernel_size=self.token_mixer[0].kernel_size,
            stride=1,
            padding=self.token_mixer[0].padding,
            groups=self.dim,
            bias=True
        )
        fused_conv.weight.data = kernel
        fused_conv.bias.data = bias
        
        # Replace token mixer with fused version
        self.token_mixer = nn.Sequential(
            fused_conv,
            *list(self.token_mixer[2:])  # Keep activation if exists
        )
        
        # Remove reparameterization branches
        self.reparam_branches = nn.ModuleList()

    def _fuse_bn(self, conv, bn):
        """
        Fuse Conv and BN layers
        """
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        
        fused_kernel = kernel * t
        fused_bias = beta - running_mean * gamma / std
        
        return fused_kernel, fused_bias


class RepViTCSPBlock(nn.Module):
    """
    CSP-style block with RepViT for better efficiency
    """
    def __init__(self, in_ch, out_ch, n=1, use_se=False):
        super().__init__()
        self.split_conv = ConvBlock(in_ch, out_ch // 2)
        self.path_conv = ConvBlock(in_ch, out_ch // 2)
        self.final_conv = ConvBlock((2 + n) * out_ch // 2, out_ch)
        self.repvit_blocks = nn.ModuleList([
            RepViTBlock(out_ch // 2, use_se=use_se) for _ in range(n)
        ])

    def forward(self, x):
        left = self.split_conv(x)
        right = self.path_conv(x)
        out_list = [left, right]
        out_list.extend(block(out_list[-1]) for block in self.repvit_blocks)
        return self.final_conv(torch.cat(out_list, dim=1))

    def reparameterize(self):
        """Reparameterize all RepViT blocks"""
        for block in self.repvit_blocks:
            if hasattr(block, 'reparameterize'):
                block.reparameterize()


# Example usage and test
if __name__ == "__main__":
    # Test RepViT block
    x = torch.randn(2, 64, 32, 32)
    block = RepViTBlock(64, use_se=True)
    
    # Training mode
    block.train()
    out = block(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape (training): {out.shape}")
    
    # Switch to eval and reparameterize
    block.eval()
    block.reparameterize()
    out_inference = block(x)
    print(f"Output shape (inference): {out_inference.shape}")
    
    # Test RepViTCSPBlock
    csp_block = RepViTCSPBlock(128, 64, n=2, use_se=True)
    x2 = torch.randn(2, 128, 32, 32)
    out_csp = csp_block(x2)
    print(f"CSP Input shape: {x2.shape}")
    print(f"CSP Output shape: {out_csp.shape}")
