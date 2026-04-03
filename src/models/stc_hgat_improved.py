"""
stc_hgat_improved.py
--------------------
Improved STC-HGAT with Phase 3 enhancements:
- Gated fusion mechanism
- Multi-head cross-attention
- Enhanced temporal modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional
import numpy as np

from src.models.stc_hgat_model import (
    HyperGATModule, HGATModule, PositionalEncoding, 
    infonce_loss, adaptive_weight_loss
)


class GatedFusion(nn.Module):
    """
    Gated fusion mechanism to adaptively combine spatial and temporal features
    Instead of simple sum: h = h_spatial + h_temporal
    Use gating: h = gate * h_spatial + (1 - gate) * h_temporal
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
    
    def forward(self, h_spatial: Tensor, h_temporal: Tensor) -> Tensor:
        """
        Parameters
        ----------
        h_spatial : (B, N, H)
        h_temporal : (B, N, H)
        
        Returns
        -------
        h_fused : (B, N, H)
        """
        # Concatenate features
        h_concat = torch.cat([h_spatial, h_temporal], dim=-1)  # (B, N, 2H)
        
        # Compute gate
        gate = self.gate_net(h_concat)  # (B, N, H)
        
        # Gated fusion
        h_fused = gate * h_spatial + (1 - gate) * h_temporal
        
        return h_fused


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention between spatial and temporal features
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """
        Parameters
        ----------
        query : (B, N, H) - e.g., spatial features
        key : (B, N, H) - e.g., temporal features
        value : (B, N, H) - e.g., temporal features
        
        Returns
        -------
        output : (B, N, H)
        """
        B, N, H = query.shape
        
        # Project and reshape to multi-head
        Q = self.q_proj(query).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, N, head_dim)
        K = self.k_proj(key).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, V)  # (B, heads, N, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, N, H)  # (B, N, H)
        
        # Output projection
        out = self.out_proj(out)
        
        return out


class MultiScaleTemporalBlock(nn.Module):
    """
    Multi-scale temporal modeling with different receptive fields
    """
    def __init__(self, hidden_dim: int, scales: list = [1, 3, 7]):
        super().__init__()
        self.scales = scales
        
        # Temporal convolutions at different scales
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=k, padding=k//2)
            for k in scales
        ])
        
        # Fusion layer
        self.fusion = nn.Linear(hidden_dim * len(scales), hidden_dim)
    
    def forward(self, h: Tensor) -> Tensor:
        """
        Parameters
        ----------
        h : (B, N, T, H)
        
        Returns
        -------
        h_multi : (B, N, T, H)
        """
        B, N, T, H = h.shape
        
        # Reshape for conv1d: (B*N, H, T)
        h_flat = h.reshape(B * N, T, H).transpose(1, 2)
        
        # Apply multi-scale convolutions
        outputs = []
        for conv in self.convs:
            out = F.relu(conv(h_flat))  # (B*N, H, T)
            outputs.append(out)
        
        # Concatenate scales
        h_concat = torch.cat(outputs, dim=1)  # (B*N, H*scales, T)
        h_concat = h_concat.transpose(1, 2)  # (B*N, T, H*scales)
        
        # Fuse scales
        h_fused = self.fusion(h_concat)  # (B*N, T, H)
        
        # Reshape back
        h_multi = h_fused.reshape(B, N, T, H)
        
        return h_multi


class ImprovedSTCHGAT(nn.Module):
    """
    Improved STC-HGAT with Phase 3 enhancements
    """
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        num_stations: int = 79,
        num_regions: int = 5,
        num_hypergat_layers: int = 2,
        num_hgat_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        forecast_horizons: list = [1, 3, 7],
        use_gated_fusion: bool = True,
        use_cross_attention: bool = True,
        use_multiscale_temporal: bool = True,
    ):
        super().__init__()
        
        self.hidden = hidden_dim
        self.num_stations = num_stations
        self.use_gated_fusion = use_gated_fusion
        self.use_cross_attention = use_cross_attention
        self.use_multiscale_temporal = use_multiscale_temporal
        
        # Feature embedding
        self.feat_embed = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Spatial processing (HyperGAT)
        self.hypergat = HyperGATModule(
            hidden=hidden_dim,
            n_regions=num_regions,
            n_layers=num_hypergat_layers,
            dropout=dropout
        )
        
        # Temporal processing (HGAT)
        self.hgat = HGATModule(
            hidden=hidden_dim,
            dropout=dropout
        )
        
        # Multi-scale temporal modeling (optional)
        if use_multiscale_temporal:
            self.multiscale_temporal = MultiScaleTemporalBlock(hidden_dim)
        
        # Fusion mechanisms
        if use_gated_fusion:
            self.gated_fusion = GatedFusion(hidden_dim)
        
        if use_cross_attention:
            self.cross_attn_s2t = CrossAttentionFusion(hidden_dim, num_heads, dropout)
            self.cross_attn_t2s = CrossAttentionFusion(hidden_dim, num_heads, dropout)
        
        # Position encoding
        self.pos_enc = PositionalEncoding(hidden_dim, max_len=200)
        
        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(
        self,
        x: Tensor,
        spatial_graph = None,
        temporal_graph = None,
        H_inc: Tensor = None,
        membership: np.ndarray = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Parameters
        ----------
        x : (B, N, T, F)
        
        Returns
        -------
        pred : (B, N)
        h_spatial : (B, N, H)
        h_temporal : (B, N, H)
        """
        B, N, T, F = x.shape
        
        # Feature embedding
        x_flat = x.reshape(B * N * T, F)
        h_flat = self.feat_embed(x_flat)
        h = h_flat.reshape(B, N, T, self.hidden)
        
        # Multi-scale temporal modeling (optional)
        if self.use_multiscale_temporal:
            h = self.multiscale_temporal(h)
        
        # Spatial processing
        h_mean = h.mean(dim=2)  # (B, N, H)
        
        if H_inc is None:
            h_spatial = h_mean
        else:
            h_spatial = self.hypergat(h_mean, H_inc, membership)
        
        # Temporal processing
        h_temporal = self.hgat(h)
        
        # Enhanced fusion
        if self.use_cross_attention:
            # Bidirectional cross-attention
            h_s_enhanced = self.cross_attn_s2t(h_spatial, h_temporal, h_temporal)
            h_t_enhanced = self.cross_attn_t2s(h_temporal, h_spatial, h_spatial)
            
            # Residual connections
            h_spatial = h_spatial + h_s_enhanced
            h_temporal = h_temporal + h_t_enhanced
        
        if self.use_gated_fusion:
            h_fused = self.gated_fusion(h_spatial, h_temporal)
        else:
            h_fused = h_spatial + h_temporal
        
        # Enrich temporal features with fused representation
        h_enrich = h + h_fused.unsqueeze(2)
        
        # Position encoding with soft attention
        session_repr = self.pos_enc(h_enrich)
        
        # Prediction
        pred = self.head(session_repr).squeeze(-1)
        
        return pred, h_spatial, h_temporal
    
    def compute_loss(
        self,
        pred: Tensor,
        y: Tensor,
        h_spatial: Tensor,
        h_temporal: Tensor,
        lambda_contrastive: float = 0.1,
        temperature: float = 0.1,
    ) -> Tuple[Tensor, dict]:
        """
        Combined loss: L = L_r + λ · L_c
        
        Returns
        -------
        total_loss : Tensor
        loss_dict : dict with individual losses
        """
        # Regression loss (MSE)
        l_r = F.mse_loss(pred, y)
        
        # Contrastive loss (InfoNCE)
        l_c = infonce_loss(h_spatial, h_temporal, temperature=temperature)
        
        # Total loss
        total_loss = l_r + lambda_contrastive * l_c
        
        loss_dict = {
            'total': total_loss.item(),
            'regression': l_r.item(),
            'contrastive': l_c.item(),
        }
        
        return total_loss, loss_dict
