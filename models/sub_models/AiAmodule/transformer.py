"""
Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MultiheadAttention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""

import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from models.sub_models.AiAmodule.attention import AiAModule


def check_inf(tensor):
    return torch.isinf(tensor.detach()).any()


def check_nan(tensor):
    return torch.isnan(tensor.detach()).any()


def check_valid(tensor, type_name):
    if check_inf(tensor):
        print('%s is inf' % type_name)
    if check_nan(tensor):
        print('%s is nan' % type_name)


class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 dim_feedforward=2048, dropout=0.1, activation='relu', normalize_before=False,
                 divide_norm=False, use_AiA=True, match_dim=64, feat_size=400):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before,
                                                divide_norm=divide_norm, use_AiA=use_AiA,
                                                match_dim=match_dim, feat_size=feat_size)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        if num_encoder_layers == 0:
            self.encoder = None
        else:
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.d_feed = dim_feedforward
        # Try dividing norm to avoid NAN
        self.divide_norm = divide_norm
        self.scale_factor = float(d_model // nhead) ** 0.5

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def run_encoder(self, feat, mask, pos_emb, inr_emb):
        """
        Args:
            feat: (H1W1+H2W2, bs, C)
            mask: (bs, H1W1+H2W2)
            pos_embed: (H1W1+H2W2, bs, C)
        """

        return self.encoder(feat, src_key_padding_mask=mask, pos=pos_emb, inr=inr_emb)



class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        # clone 3 copies
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                inr: Optional[Tensor] = None):
        output = src  # (HW,B,C)

        for stack, layer in enumerate(self.layers):
            output = layer(output, src_key_padding_mask=src_key_padding_mask, pos=pos, inr=inr)

        if self.norm is not None:
            output = self.norm(output)
        return output

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation='relu', normalize_before=False, divide_norm=False,
                 use_AiA=True, match_dim=64, feat_size=400):
        super().__init__()
        self.self_attn = AiAModule(d_model, nhead, dropout=dropout,
                                   use_AiA=use_AiA, match_dim=match_dim, feat_size=feat_size)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before  # First normalization, then add

        self.divide_norm = divide_norm
        self.scale_factor = float(d_model // nhead) ** 0.5

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                inr: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)  # Add pos to src
        if self.divide_norm:
            # Encoder divide by norm
            q = q / torch.norm(q, dim=-1, keepdim=True) * self.scale_factor
            k = k / torch.norm(k, dim=-1, keepdim=True)
        # src2 = self.self_attn(q, k, value=src)[0]
        src2 = self.self_attn(query=q, key=k, value=src, pos_emb=inr, key_padding_mask=src_key_padding_mask)[0]
        # Add and norm
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # FFN
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # Add and Norm
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def build_transformer():
    return Transformer(
        d_model=256,
        dropout=0.1,
        nhead=8,
        dim_feedforward=1024,
        num_encoder_layers=3,
        normalize_before=False,
        divide_norm=False,
        use_AiA=True,
        match_dim=64,
        feat_size=400
    )


def _get_activation_fn(activation):
    """
    Return an activation function given a string.
    """

    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return F.gelu
    if activation == 'glu':
        return F.glu
    raise RuntimeError(f'ERROR: activation should be relu/gelu/glu, not {activation}')
