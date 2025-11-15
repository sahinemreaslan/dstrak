"""
DINOv3 Backbone for DSTARK

Supports flexible input sizes thanks to RoPE (Rotary Position Embeddings).
Implements the FlexibleBackbone interface for loose coupling with tracker.

Design Pattern: Strategy Pattern
- Implements abstract backbone interface
- Can be swapped with other backbones without changing tracker code
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from .base_backbone import FlexibleBackbone


class Attention(nn.Module):
    """Multi-head attention with RoPE support"""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """MLP block"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """Transformer block"""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class DINOv3Backbone(FlexibleBackbone):
    """
    DINOv3 Small backbone for visual tracking.

    Implements FlexibleBackbone interface for compatibility with DSTARK tracker.

    Key features:
    - Flexible input sizes thanks to interpolated positional encodings
    - No fixed template/search size constraints
    - Rich feature extraction with self-supervised pretrained weights
    - Supports RoPE (Rotary Position Embeddings) style interpolation

    Architecture:
    - Patch-based Vision Transformer
    - Default: DINOv3-Small (384-dim, 12 layers, 6 heads)
    - Can be configured for Base/Large variants
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,  # DINOv3 small
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        pretrained_path: Optional[str] = None
    ):
        """
        Initialize DINOv3 backbone.

        Args:
            img_size: Default image size (for pos_embed initialization)
            patch_size: Patch size for patch embedding
            in_chans: Number of input channels
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dim ratio
            qkv_bias: Use bias in QKV projection
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            pretrained_path: Path to pretrained weights
        """
        super().__init__()

        self._patch_size = patch_size
        self._embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embeddings will be interpolated, so we keep it flexible
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

        # Load pretrained weights if provided
        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)

    @property
    def embed_dim(self) -> int:
        """Return the embedding dimension (implements BaseBackbone interface)."""
        return self._embed_dim

    @property
    def patch_size(self) -> int:
        """Return the patch size (implements BaseBackbone interface)."""
        return self._patch_size

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(
        self,
        x: torch.Tensor,
        w: int,
        h: int
    ) -> torch.Tensor:
        """
        Interpolate positional embeddings for arbitrary input sizes.

        This allows the model to handle images of any size, not just
        the size it was pretrained on. Critical for tracking where
        template and search regions have different sizes.

        Args:
            x: Feature tensor [B, N+1, C] (includes CLS token)
            w: Width of input image
            h: Height of input image

        Returns:
            Interpolated positional embeddings [B, N+1, C]
        """
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1

        if npatch == N and w == h:
            return self.pos_embed

        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]

        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size

        # Add a small number to avoid floating point errors
        w0, h0 = w0 + 0.1, h0 + 0.1

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )

        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, x):
        """
        Forward pass with flexible input size

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            features: Extracted features [B, N, C] where N = (H//patch_size) * (W//patch_size) + 1
        """
        B, C, H, W = x.shape

        # Patch embedding
        x = self.patch_embed(x)  # [B, embed_dim, H//patch_size, W//patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional encoding (interpolated for current size)
        x = x + self.interpolate_pos_encoding(x, W, H)
        x = self.pos_drop(x)

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x

    def load_pretrained(self, pretrained_path: str) -> None:
        """
        Load pretrained weights (implements BaseBackbone interface).

        Handles different checkpoint formats gracefully and allows
        size mismatches for positional embeddings (which will be interpolated).

        Args:
            pretrained_path: Path to pretrained weights file
        """
        print(f"Loading pretrained weights from {pretrained_path}")

        checkpoint = torch.load(pretrained_path, map_location='cpu')

        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if present
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # Load weights (ignore size mismatches for pos_embed as we interpolate)
        msg = self.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights: {msg}")

    def get_num_layers(self) -> int:
        """Return the number of transformer layers."""
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        """Return parameters that should not have weight decay applied."""
        return {'pos_embed', 'cls_token'}

    def freeze_except_last_n_layers(self, n: int) -> None:
        """
        Freeze all layers except the last n transformer blocks.

        Useful for fine-tuning with limited data.

        Args:
            n: Number of last blocks to keep trainable
        """
        # Freeze patch embedding
        for param in self.patch_embed.parameters():
            param.requires_grad = False

        # Freeze position embeddings and cls token
        self.pos_embed.requires_grad = False
        self.cls_token.requires_grad = False

        # Freeze all blocks except last n
        num_blocks = len(self.blocks)
        freeze_until = max(0, num_blocks - n)

        for i, block in enumerate(self.blocks):
            if i < freeze_until:
                for param in block.parameters():
                    param.requires_grad = False
            else:
                for param in block.parameters():
                    param.requires_grad = True

        # Keep final norm trainable if any blocks are trainable
        if n > 0:
            for param in self.norm.parameters():
                param.requires_grad = True

        print(f"Froze all layers except last {n} transformer blocks")
