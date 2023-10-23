"""Modified from DETR's transformer.py

- Cross encoder layer is similar to the decoder layers in Transformer, but
  updates both source and target features
- Added argument to control whether value has position embedding or not for
  TransformerEncoderLayer and TransformerDecoderLayer
- Decoder layer now keeps track of attention weights
"""

import copy
from typing import Optional, List

from models.sub_models.cross_attention.position_embedding import PositionEmbeddingCoordsSine, \
    PositionEmbeddingLearned

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class TransformerCrossEncoder(nn.Module):

    def __init__(self, cross_encoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(cross_encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, src, tgt,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                src_pos: Optional[Tensor] = None,
                tgt_pos: Optional[Tensor] = None,
                src_xyz: Optional[Tensor] = None, 
                tgt_xyz: Optional[Tensor] = None):

        src_intermediate, tgt_intermediate = [], []

        for layer in self.layers:
            src, tgt = layer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask,
                             src_key_padding_mask=src_key_padding_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             src_pos=src_pos, tgt_pos=tgt_pos)
            if self.return_intermediate:
                src_intermediate.append(self.norm(src) if self.norm is not None else src)
                tgt_intermediate.append(self.norm(tgt) if self.norm is not None else tgt)

        if self.norm is not None:
            src = self.norm(src)
            tgt = self.norm(tgt)
            if self.return_intermediate:
                if len(self.layers) > 0:
                    src_intermediate.pop()
                    tgt_intermediate.pop()
                src_intermediate.append(src)
                tgt_intermediate.append(tgt)

        if self.return_intermediate:
            return torch.stack(src_intermediate), torch.stack(tgt_intermediate)

        return src, tgt

    def get_attentions(self):
        """For analysis: Retrieves the attention maps last computed by the individual layers."""

        src_satt_all, tgt_satt_all = [], []
        src_xatt_all, tgt_xatt_all = [], []

        for layer in self.layers:
            src_satt, tgt_satt = layer.satt_weights
            src_xatt, tgt_xatt = layer.xatt_weights

            src_satt_all.append(src_satt)
            tgt_satt_all.append(tgt_satt)
            src_xatt_all.append(src_xatt)
            tgt_xatt_all.append(tgt_xatt)

        src_satt_all = torch.stack(src_satt_all)
        tgt_satt_all = torch.stack(tgt_satt_all)
        src_xatt_all = torch.stack(src_xatt_all)
        tgt_xatt_all = torch.stack(tgt_xatt_all)

        return (src_satt_all, tgt_satt_all), (src_xatt_all, tgt_xatt_all)

class MaskedTransformerCrossEncoder(TransformerCrossEncoder):
    
    def __init__(self, cross_encoder_layer, num_layers, masking_radius, norm=None, return_intermediate=False):
        super().__init__(cross_encoder_layer, num_layers)
        self.layers = _get_clones(cross_encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert len(masking_radius) == num_layers
        self.masking_radius = masking_radius

    def compute_mask(self, xyz, radius):
        with torch.no_grad():
            dist = torch.cdist(xyz, xyz, p=2)
            # entries that are True in the mask do not contribute to self-attention
            # so points outside the radius are not considered
            mask = dist >= radius
        return mask, dist
    
    def forward(self, src, tgt,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                src_pos: Optional[Tensor] = None,
                tgt_pos: Optional[Tensor] = None,
                src_xyz: Optional[Tensor] = None, 
                tgt_xyz: Optional[Tensor] = None):

        src_intermediate, tgt_intermediate = [], []

        for idx, layer in enumerate(self.layers):
            if self.masking_radius[idx] > 0:
                src_mask, src_dist = self.compute_mask(src_xyz, self.masking_radius[idx])
                tgt_mask, src_dist = self.compute_mask(tgt_xyz, self.masking_radius[idx])
                # mask must be tiled to num_heads of the transformer
                bsz, n, n = src_mask.shape
                nhead = layer.nhead
                src_mask = src_mask.unsqueeze(1)
                src_mask = src_mask.repeat(1, nhead, 1, 1)
                src_mask = src_mask.view(bsz * nhead, n, n)
                tgt_mask = tgt_mask.unsqueeze(1)
                tgt_mask = tgt_mask.repeat(1, nhead, 1, 1)
                tgt_mask = tgt_mask.view(bsz * nhead, n, n)
                
            src, tgt = layer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask,
                             src_key_padding_mask=src_key_padding_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             src_pos=src_pos, tgt_pos=tgt_pos)
            if self.return_intermediate:
                src_intermediate.append(self.norm(src) if self.norm is not None else src)
                tgt_intermediate.append(self.norm(tgt) if self.norm is not None else tgt)

        if self.norm is not None:
            src = self.norm(src)
            tgt = self.norm(tgt)
            if self.return_intermediate:
                if len(self.layers) > 0:
                    src_intermediate.pop()
                    tgt_intermediate.pop()
                src_intermediate.append(src)
                tgt_intermediate.append(tgt)

        if self.return_intermediate:
            return torch.stack(src_intermediate), torch.stack(tgt_intermediate)

        return src, tgt

    def get_attentions(self):
        """For analysis: Retrieves the attention maps last computed by the individual layers."""

        src_satt_all, tgt_satt_all = [], []
        src_xatt_all, tgt_xatt_all = [], []

        for layer in self.layers:
            src_satt, tgt_satt = layer.satt_weights
            src_xatt, tgt_xatt = layer.xatt_weights

            src_satt_all.append(src_satt)
            tgt_satt_all.append(tgt_satt)
            src_xatt_all.append(src_xatt)
            tgt_xatt_all.append(tgt_xatt)

        src_satt_all = torch.stack(src_satt_all)
        tgt_satt_all = torch.stack(tgt_satt_all)
        src_xatt_all = torch.stack(src_xatt_all)
        tgt_xatt_all = torch.stack(tgt_xatt_all)

        return (src_satt_all, tgt_satt_all), (src_xatt_all, tgt_xatt_all)

class TransformerCrossEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 sa_val_has_pos_emb=False,
                 ca_val_has_pos_emb=False,
                 attention_type='dot_prod'
                 ):
        super().__init__()

        self.nhead = nhead

        # Self, cross attention layers
        if attention_type == 'dot_prod':
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        else:
            raise NotImplementedError

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.sa_val_has_pos_emb = sa_val_has_pos_emb
        self.ca_val_has_pos_emb = ca_val_has_pos_emb
        self.satt_weights, self.xatt_weights = None, None  # For analysis

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, tgt,
                     src_mask: Optional[Tensor] = None,
                     tgt_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     src_pos: Optional[Tensor] = None,
                     tgt_pos: Optional[Tensor] = None,):


        # Self attention
        src_w_pos = self.with_pos_embed(src, src_pos)
        q = k = src_w_pos
        src2, satt_weights_s = self.self_attn(q, k,
                              value=src_w_pos if self.sa_val_has_pos_emb else src,
                              attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        tgt_w_pos = self.with_pos_embed(tgt, tgt_pos)
        q = k = tgt_w_pos
        tgt2, satt_weights_t = self.self_attn(q, k,
                                              value=tgt_w_pos if self.sa_val_has_pos_emb else tgt,
                                              attn_mask=tgt_mask,
                                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross attention
        src_w_pos = self.with_pos_embed(src, src_pos)
        tgt_w_pos = self.with_pos_embed(tgt, tgt_pos)

        src2, xatt_weights_s = self.multihead_attn(query=self.with_pos_embed(src, src_pos),
                                                   key=tgt_w_pos,
                                                   value=tgt_w_pos if self.ca_val_has_pos_emb else tgt,
                                                   key_padding_mask=tgt_key_padding_mask)
        tgt2, xatt_weights_t = self.multihead_attn(query=self.with_pos_embed(tgt, tgt_pos),
                                                   key=src_w_pos,
                                                   value=src_w_pos if self.ca_val_has_pos_emb else src,
                                                   key_padding_mask=src_key_padding_mask)

        src = self.norm2(src + self.dropout2(src2))
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        # Position-wise feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm3(src)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        # Stores the attention weights for analysis, if required
        self.satt_weights = (satt_weights_s, satt_weights_t)
        self.xatt_weights = (xatt_weights_s, xatt_weights_t)

        return src, tgt

    def forward_pre(self, src, tgt,
                    src_mask: Optional[Tensor] = None,
                    tgt_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    src_pos: Optional[Tensor] = None,
                    tgt_pos: Optional[Tensor] = None,):


        # Self attention
        src2 = self.norm1(src)
        src2_w_pos = self.with_pos_embed(src2, src_pos)
        q = k = src2_w_pos
        src2, satt_weights_s = self.self_attn(q, k,
                                              value=src2_w_pos if self.sa_val_has_pos_emb else src2,
                                              attn_mask=src_mask,
                                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)

        tgt2 = self.norm1(tgt)
        tgt2_w_pos = self.with_pos_embed(tgt2, tgt_pos)
        q = k = tgt2_w_pos
        tgt2, satt_weights_t = self.self_attn(q, k,
                                              value=tgt2_w_pos if self.sa_val_has_pos_emb else tgt2,
                                              attn_mask=tgt_mask,
                                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)

        # Cross attention
        src2, tgt2 = self.norm2(src), self.norm2(tgt)
        src_w_pos = self.with_pos_embed(src2, src_pos)
        tgt_w_pos = self.with_pos_embed(tgt2, tgt_pos)

        src3, xatt_weights_s = self.multihead_attn(query=self.with_pos_embed(src2, src_pos),
                                                   key=tgt_w_pos,
                                                   value=tgt_w_pos if self.ca_val_has_pos_emb else tgt2,
                                                   key_padding_mask=tgt_key_padding_mask)
        tgt3, xatt_weights_t = self.multihead_attn(query=self.with_pos_embed(tgt2, tgt_pos),
                                                   key=src_w_pos,
                                                   value=src_w_pos if self.ca_val_has_pos_emb else src2,
                                                   key_padding_mask=src_key_padding_mask)

        src = src + self.dropout2(src3)
        tgt = tgt + self.dropout2(tgt3)

        # Position-wise feedforward
        src2 = self.norm3(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout3(src2)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        # Stores the attention weights for analysis, if required
        self.satt_weights = (satt_weights_s, satt_weights_t)
        self.xatt_weights = (xatt_weights_s, xatt_weights_t)

        return src, tgt

    def forward(self, src, tgt,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                src_pos: Optional[Tensor] = None,
                tgt_pos: Optional[Tensor] = None,):

        if self.normalize_before:
            return self.forward_pre(src, tgt, src_mask, tgt_mask,
                                    src_key_padding_mask, tgt_key_padding_mask, src_pos, tgt_pos)
        return self.forward_post(src, tgt, src_mask, tgt_mask,
                                 src_key_padding_mask, tgt_key_padding_mask, src_pos, tgt_pos)

class FlexibleTransformerEncoder(nn.Module):
    
    def __init__(self, self_layer, cross_layer, layer_list, masking_radius, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _flex_clones(self_layer, cross_layer, layer_list)
        self.num_layers = len(layer_list)
        self.layer_list = layer_list
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert len(masking_radius) == self.layer_list.count('s')
        self.masking_radius = masking_radius

    def compute_mask(self, xyz, radius):
        with torch.no_grad():
            dist = torch.cdist(xyz, xyz, p=2)
            # entries that are True in the mask do not contribute to self-attention
            # so points outside the radius are not considered
            mask = dist >= radius
        return mask, dist
    
    def forward(self, src, tgt,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                src_pos: Optional[Tensor] = None,
                tgt_pos: Optional[Tensor] = None,
                src_xyz: Optional[Tensor] = None, 
                tgt_xyz: Optional[Tensor] = None):

        src_intermediate, tgt_intermediate = [], []
        
        s_count=0

        for idx, layer in enumerate(self.layers):
            if self.layer_list[idx]=='s' and self.masking_radius[s_count] > 0:
                src_mask, src_dist = self.compute_mask(src_xyz, self.masking_radius[s_count])
                tgt_mask, src_dist = self.compute_mask(tgt_xyz, self.masking_radius[s_count])
                s_count += 1
                # mask must be tiled to num_heads of the transformer
                bsz, n, n = src_mask.shape
                nhead = layer.nhead
                src_mask = src_mask.unsqueeze(1)
                src_mask = src_mask.repeat(1, nhead, 1, 1)
                src_mask = src_mask.view(bsz * nhead, n, n)
                tgt_mask = tgt_mask.unsqueeze(1)
                tgt_mask = tgt_mask.repeat(1, nhead, 1, 1)
                tgt_mask = tgt_mask.view(bsz * nhead, n, n)
                
            src, tgt = layer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask,
                             src_key_padding_mask=src_key_padding_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             src_pos=src_pos, tgt_pos=tgt_pos)
            if self.return_intermediate:
                src_intermediate.append(self.norm(src) if self.norm is not None else src)
                tgt_intermediate.append(self.norm(tgt) if self.norm is not None else tgt)

        if self.norm is not None:
            src = self.norm(src)
            tgt = self.norm(tgt)
            if self.return_intermediate:
                if len(self.layers) > 0:
                    src_intermediate.pop()
                    tgt_intermediate.pop()
                src_intermediate.append(src)
                tgt_intermediate.append(tgt)

        if self.return_intermediate:
            return torch.stack(src_intermediate), torch.stack(tgt_intermediate)

        return src, tgt

    def get_attentions(self):
        """For analysis: Retrieves the attention maps last computed by the individual layers."""

        src_satt_all, tgt_satt_all = [], []
        src_xatt_all, tgt_xatt_all = [], []

        for layer in self.layers:
            src_satt, tgt_satt = layer.satt_weights
            src_xatt, tgt_xatt = layer.xatt_weights

            src_satt_all.append(src_satt)
            tgt_satt_all.append(tgt_satt)
            src_xatt_all.append(src_xatt)
            tgt_xatt_all.append(tgt_xatt)

        src_satt_all = torch.stack(src_satt_all)
        tgt_satt_all = torch.stack(tgt_satt_all)
        src_xatt_all = torch.stack(src_xatt_all)
        tgt_xatt_all = torch.stack(tgt_xatt_all)

        return (src_satt_all, tgt_satt_all), (src_xatt_all, tgt_xatt_all)

class TransformerSelfLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 sa_val_has_pos_emb=False,
                 ca_val_has_pos_emb=False,
                 attention_type='dot_prod'
                 ):
        super().__init__()

        self.nhead = nhead

        # Self, cross attention layers
        if attention_type == 'dot_prod':
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        else:
            raise NotImplementedError

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.sa_val_has_pos_emb = sa_val_has_pos_emb
        self.ca_val_has_pos_emb = ca_val_has_pos_emb
        self.satt_weights, self.xatt_weights = None, None  # For analysis

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, tgt,
                     src_mask: Optional[Tensor] = None,
                     tgt_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     src_pos: Optional[Tensor] = None,
                     tgt_pos: Optional[Tensor] = None,):


        # Self attention
        src_w_pos = self.with_pos_embed(src, src_pos)
        q = k = src_w_pos
        src2, satt_weights_s = self.self_attn(q, k,
                              value=src_w_pos if self.sa_val_has_pos_emb else src,
                              attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        tgt_w_pos = self.with_pos_embed(tgt, tgt_pos)
        q = k = tgt_w_pos
        tgt2, satt_weights_t = self.self_attn(q, k,
                                              value=tgt_w_pos if self.sa_val_has_pos_emb else tgt,
                                              attn_mask=tgt_mask,
                                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Position-wise feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm3(src)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        # Stores the attention weights for analysis, if required
        self.satt_weights = (satt_weights_s, satt_weights_t)

        return src, tgt

    def forward_pre(self, src, tgt,
                    src_mask: Optional[Tensor] = None,
                    tgt_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    src_pos: Optional[Tensor] = None,
                    tgt_pos: Optional[Tensor] = None,):


        # Self attention
        src2 = self.norm1(src)
        src2_w_pos = self.with_pos_embed(src2, src_pos)
        q = k = src2_w_pos
        src2, satt_weights_s = self.self_attn(q, k,
                                              value=src2_w_pos if self.sa_val_has_pos_emb else src2,
                                              attn_mask=src_mask,
                                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)

        tgt2 = self.norm1(tgt)
        tgt2_w_pos = self.with_pos_embed(tgt2, tgt_pos)
        q = k = tgt2_w_pos
        tgt2, satt_weights_t = self.self_attn(q, k,
                                              value=tgt2_w_pos if self.sa_val_has_pos_emb else tgt2,
                                              attn_mask=tgt_mask,
                                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)


        # Position-wise feedforward
        src2 = self.norm3(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout3(src2)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        # Stores the attention weights for analysis, if required
        self.satt_weights = (satt_weights_s, satt_weights_t)

        return src, tgt

    def forward(self, src, tgt,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                src_pos: Optional[Tensor] = None,
                tgt_pos: Optional[Tensor] = None,):

        if self.normalize_before:
            return self.forward_pre(src, tgt, src_mask, tgt_mask,
                                    src_key_padding_mask, tgt_key_padding_mask, src_pos, tgt_pos)
        return self.forward_post(src, tgt, src_mask, tgt_mask,
                                 src_key_padding_mask, tgt_key_padding_mask, src_pos, tgt_pos)
        
class TransformerCrossLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 sa_val_has_pos_emb=False,
                 ca_val_has_pos_emb=False,
                 attention_type='dot_prod'
                 ):
        super().__init__()

        self.nhead = nhead

        # Self, cross attention layers
        if attention_type == 'dot_prod':
            self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        else:
            raise NotImplementedError

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.sa_val_has_pos_emb = sa_val_has_pos_emb
        self.ca_val_has_pos_emb = ca_val_has_pos_emb
        self.satt_weights, self.xatt_weights = None, None  # For analysis

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, tgt,
                     src_mask: Optional[Tensor] = None,
                     tgt_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     src_pos: Optional[Tensor] = None,
                     tgt_pos: Optional[Tensor] = None,):

        # Cross attention
        src_w_pos = self.with_pos_embed(src, src_pos)
        tgt_w_pos = self.with_pos_embed(tgt, tgt_pos)

        src2, xatt_weights_s = self.multihead_attn(query=self.with_pos_embed(src, src_pos),
                                                   key=tgt_w_pos,
                                                   value=tgt_w_pos if self.ca_val_has_pos_emb else tgt,
                                                   key_padding_mask=tgt_key_padding_mask)
        tgt2, xatt_weights_t = self.multihead_attn(query=self.with_pos_embed(tgt, tgt_pos),
                                                   key=src_w_pos,
                                                   value=src_w_pos if self.ca_val_has_pos_emb else src,
                                                   key_padding_mask=src_key_padding_mask)

        src = self.norm2(src + self.dropout2(src2))
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        # Position-wise feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm3(src)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        # Stores the attention weights for analysis, if required
        self.xatt_weights = (xatt_weights_s, xatt_weights_t)

        return src, tgt

    def forward_pre(self, src, tgt,
                    src_mask: Optional[Tensor] = None,
                    tgt_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    src_pos: Optional[Tensor] = None,
                    tgt_pos: Optional[Tensor] = None,):

        # Cross attention
        src2, tgt2 = self.norm2(src), self.norm2(tgt)
        src_w_pos = self.with_pos_embed(src2, src_pos)
        tgt_w_pos = self.with_pos_embed(tgt2, tgt_pos)

        src3, xatt_weights_s = self.multihead_attn(query=self.with_pos_embed(src2, src_pos),
                                                   key=tgt_w_pos,
                                                   value=tgt_w_pos if self.ca_val_has_pos_emb else tgt2,
                                                   key_padding_mask=tgt_key_padding_mask)
        tgt3, xatt_weights_t = self.multihead_attn(query=self.with_pos_embed(tgt2, tgt_pos),
                                                   key=src_w_pos,
                                                   value=src_w_pos if self.ca_val_has_pos_emb else src2,
                                                   key_padding_mask=src_key_padding_mask)

        src = src + self.dropout2(src3)
        tgt = tgt + self.dropout2(tgt3)

        # Position-wise feedforward
        src2 = self.norm3(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout3(src2)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        # Stores the attention weights for analysis, if required
        self.xatt_weights = (xatt_weights_s, xatt_weights_t)

        return src, tgt

    def forward(self, src, tgt,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                src_pos: Optional[Tensor] = None,
                tgt_pos: Optional[Tensor] = None,):

        if self.normalize_before:
            return self.forward_pre(src, tgt, src_mask, tgt_mask,
                                    src_key_padding_mask, tgt_key_padding_mask, src_pos, tgt_pos)
        return self.forward_post(src, tgt, src_mask, tgt_mask,
                                 src_key_padding_mask, tgt_key_padding_mask, src_pos, tgt_pos)

class LuckTransformerEncoder(nn.Module):
    
    def __init__(self, hparams, layer_list, norm=None, return_intermediate=False):
        super().__init__()
        self.hparams = hparams
        self.num_neighs = hparams.num_neighs
        self.latent_dim = hparams.d_feedforward
        self.input_features = hparams.in_features_dim
        self.bb_size = self.hparams.bb_size
        
        self.layers = nn.ModuleList([])
        self.num_layers = len(layer_list)
        self.layer_list = layer_list
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.premlp = nn.Sequential(
                nn.Conv1d(self.input_features, self.bb_size * 4, kernel_size=1, bias=False), nn.BatchNorm1d(self.bb_size * 4), nn.LeakyReLU(negative_slope=0.2),
            )
        self.pos_embed = nn.ModuleList([PositionEmbeddingCoordsSine(3, self.bb_size * (2 ** (i+1)) * 2, scale= 1.0) for i in range(self.num_layers)])

        for i in range(self.num_layers):
            in_features =  self.bb_size * (2 ** (i+1)) * 2
            if self.layer_list[i] == 's':
                self.layers.append(LuckSelfLayer(
                in_features, self.hparams.nhead, self.hparams.d_feedforward, self.hparams.dropout,
                activation=self.hparams.transformer_act,
                normalize_before=self.hparams.pre_norm,
                sa_val_has_pos_emb=self.hparams.sa_val_has_pos_emb,
                ca_val_has_pos_emb=self.hparams.ca_val_has_pos_emb,
                ))
            elif self.layer_list[i] == 'c':
                self.layers.append(LuckCrossLayer(
                in_features, self.hparams.nhead, self.hparams.d_feedforward, self.hparams.dropout,
                activation=self.hparams.transformer_act,
                normalize_before=self.hparams.pre_norm,
                sa_val_has_pos_emb=self.hparams.sa_val_has_pos_emb,
                ca_val_has_pos_emb=self.hparams.ca_val_has_pos_emb,
                ))
            else: 
                assert(self.layer_list[i] in ["s", "c"]), "Please set layer_list only with 's' and 'c' representing 'self_attention_layer' and 'cross_attention_layer' respectively"
            
        last_in_dim = self.bb_size * 2 * sum([2 ** i for i in range(1,self.num_layers + 1,1)])
        self.mlp = nn.Sequential(
                nn.Conv1d(last_in_dim, self.latent_dim, kernel_size=1, bias=False), nn.BatchNorm1d(self.latent_dim), nn.LeakyReLU(negative_slope=0.2),
            )

    def compute_neibor_mask(self, neigh):
        with torch.no_grad():
            B, N, _ = neigh.shape
            mask = torch.full((B,N,N), True).to(neigh.device)
            mask.scatter_(2, neigh.long(), False)
        return mask
    
    def forward(self, src, tgt,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                src_xyz: Optional[Tensor] = None, 
                tgt_xyz: Optional[Tensor] = None,
                src_neigh: Optional[Tensor] = None, 
                tgt_neigh: Optional[Tensor] = None,):
        
        src_intermediate, tgt_intermediate = [], []

        src = self.premlp(src.permute(1,2,0)).permute(2,0,1)
        tgt = self.premlp(tgt.permute(1,2,0)).permute(2,0,1)
        
        for idx, layer in enumerate(self.layers):
            
            src_mask = self.compute_neibor_mask(src_neigh)
            tgt_mask = self.compute_neibor_mask(tgt_neigh)
            # mask must be tiled to num_heads of the transformer
            bsz, n, n = src_mask.shape
            nhead = layer.nhead
            src_mask = src_mask.unsqueeze(1)
            src_mask = src_mask.repeat(1, nhead, 1, 1)
            src_mask = src_mask.view(bsz * nhead, n, n)
            tgt_mask = tgt_mask.unsqueeze(1)
            tgt_mask = tgt_mask.repeat(1, nhead, 1, 1)
            tgt_mask = tgt_mask.view(bsz * nhead, n, n)
            
            src_pos = self.pos_embed[idx](src_xyz.reshape(-1,3)).reshape(-1,src_mask.shape[1], self.bb_size * (2 ** (idx+1)) * 2)
            tgt_pos = self.pos_embed[idx](tgt_xyz.reshape(-1,3)).reshape(-1,tgt_mask.shape[1], self.bb_size * (2 ** (idx+1)) * 2)
                
            src_out, tgt_out = layer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask,
                             src_key_padding_mask=src_key_padding_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             src_pos=src_pos.transpose(0,1), tgt_pos=tgt_pos.transpose(0,1))
            
            #print(tgt_out.shape)
            src = torch.cat((src, src_out), dim=2)
            tgt = torch.cat((tgt, tgt_out), dim=2)
            
            if self.return_intermediate:
                src_intermediate.append(src_out)
                tgt_intermediate.append(tgt_out)

        # if self.return_intermediate:
        #     return torch.stack(src_intermediate), torch.stack(tgt_intermediate)
        
        src = self.mlp(torch.cat(src_intermediate, dim=2).permute(1,2,0))
        tgt = self.mlp(torch.cat(tgt_intermediate, dim=2).permute(1,2,0))
        
        return src, tgt

    def get_attentions(self):
        """For analysis: Retrieves the attention maps last computed by the individual layers."""

        src_satt_all, tgt_satt_all = [], []
        src_xatt_all, tgt_xatt_all = [], []

        for layer in self.layers:
            src_satt, tgt_satt = layer.satt_weights
            src_xatt, tgt_xatt = layer.xatt_weights

            src_satt_all.append(src_satt)
            tgt_satt_all.append(tgt_satt)
            src_xatt_all.append(src_xatt)
            tgt_xatt_all.append(tgt_xatt)

        src_satt_all = torch.stack(src_satt_all)
        tgt_satt_all = torch.stack(tgt_satt_all)
        src_xatt_all = torch.stack(src_xatt_all)
        tgt_xatt_all = torch.stack(tgt_xatt_all)

        return (src_satt_all, tgt_satt_all), (src_xatt_all, tgt_xatt_all)
    
class TemplateTransformerEncoder(nn.Module):
    
    def __init__(self, hparams, layer_list, norm=None, return_intermediate=False):
        super().__init__()
        self.hparams = hparams
        self.num_neighs = hparams.num_neighs
        self.latent_dim = hparams.d_feedforward
        self.input_features = hparams.in_features_dim
        self.bb_size = self.hparams.bb_size
        
        self.layers = nn.ModuleList([])
        self.num_layers = len(layer_list)
        self.layer_list = layer_list
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.premlp = nn.Sequential(
                nn.Conv1d(self.input_features, self.bb_size * 4, kernel_size=1, bias=False), nn.BatchNorm1d(self.bb_size * 4), nn.LeakyReLU(negative_slope=0.2),
            )
        self.pos_embed = nn.ModuleList([PositionEmbeddingCoordsSine(3, self.bb_size * (2 ** (i+1)) * 2, scale= 1.0) for i in range(self.num_layers)])

        for i in range(self.num_layers):
            in_features =  self.bb_size * (2 ** (i+1)) * 2
            if self.layer_list[i] == 's':
                self.layers.append(TemplateSelfLayer(
                in_features, self.hparams.nhead, self.hparams.d_feedforward, self.hparams.dropout,
                activation=self.hparams.transformer_act,
                normalize_before=self.hparams.pre_norm,
                sa_val_has_pos_emb=self.hparams.sa_val_has_pos_emb,
                ca_val_has_pos_emb=self.hparams.ca_val_has_pos_emb,
                ))
            else: 
                assert(self.layer_list[i] in ["s"]), "Please set layer_list only with 's' representing 'self_attention_layer'"
            
        last_in_dim = self.bb_size * 2 * sum([2 ** i for i in range(1,self.num_layers + 1,1)])
        self.mlp = nn.Sequential(
                nn.Conv1d(last_in_dim, self.latent_dim, kernel_size=1, bias=False), nn.BatchNorm1d(self.latent_dim), nn.LeakyReLU(negative_slope=0.2),
            )

    def compute_neibor_mask(self, neigh):
        with torch.no_grad():
            B, N, _ = neigh.shape
            mask = torch.full((B,N,N), True).to(neigh.device)
            mask.scatter_(2, neigh.long(), False)
        return mask
    
    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                src_xyz: Optional[Tensor] = None, 
                src_neigh: Optional[Tensor] = None,):
        
        src_intermediate = []

        src = self.premlp(src.permute(1,2,0)).permute(2,0,1)
        
        for idx, layer in enumerate(self.layers):
            
            src_mask = self.compute_neibor_mask(src_neigh)
            # mask must be tiled to num_heads of the transformer
            bsz, n, n = src_mask.shape
            nhead = layer.nhead
            src_mask = src_mask.unsqueeze(1)
            src_mask = src_mask.repeat(1, nhead, 1, 1)
            src_mask = src_mask.view(bsz * nhead, n, n)
            
            src_pos = self.pos_embed[idx](src_xyz.reshape(-1,3)).reshape(-1,src_mask.shape[1], self.bb_size * (2 ** (idx+1)) * 2)
                
            src_out = layer(src, src_mask=src_mask, 
                             src_key_padding_mask=src_key_padding_mask,
                             src_pos=src_pos.transpose(0,1))
            
            #print(tgt_out.shape)
            src = torch.cat((src, src_out), dim=2)
            
            if self.return_intermediate:
                src_intermediate.append(src_out.permute(1,2,0))
    
        src = self.mlp(torch.cat(src_intermediate, dim=1))
        
        if self.return_intermediate:
            return src, src_intermediate
        
        return src

    def get_attentions(self):
        """For analysis: Retrieves the attention maps last computed by the individual layers."""

        src_satt_all, tgt_satt_all = [], []
        src_xatt_all, tgt_xatt_all = [], []

        for layer in self.layers:
            src_satt, tgt_satt = layer.satt_weights
            src_xatt, tgt_xatt = layer.xatt_weights

            src_satt_all.append(src_satt)
            tgt_satt_all.append(tgt_satt)
            src_xatt_all.append(src_xatt)
            tgt_xatt_all.append(tgt_xatt)

        src_satt_all = torch.stack(src_satt_all)
        tgt_satt_all = torch.stack(tgt_satt_all)
        src_xatt_all = torch.stack(src_xatt_all)
        tgt_xatt_all = torch.stack(tgt_xatt_all)

        return (src_satt_all, tgt_satt_all), (src_xatt_all, tgt_xatt_all)

class TemplateSelfLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 sa_val_has_pos_emb=False,
                 ca_val_has_pos_emb=False,
                 ):
        super().__init__()

        self.nhead = nhead

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)


        # Implementation of Feedforward model
        self.linear = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(d_model)



        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.sa_val_has_pos_emb = sa_val_has_pos_emb
        self.ca_val_has_pos_emb = ca_val_has_pos_emb
        self.satt_weights, self.xatt_weights = None, None  # For analysis

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     src_pos: Optional[Tensor] = None):


        # Self attention
        src_w_pos = self.with_pos_embed(src, src_pos)
        q = k = src_w_pos
        src2, satt_weights_s = self.self_attn(q, k,
                              value=src_w_pos if self.sa_val_has_pos_emb else src,
                              attn_mask=src_mask,
                              key_padding_mask= src_key_padding_mask)
        src = src - self.dropout(src2) # N B C
        src = self.linear(src.permute(1,2,0)) # B C N
        src = self.norm(src) # B C N
        src = self.activation(src) # B C N


        # Stores the attention weights for analysis, if required
        self.satt_weights = (satt_weights_s)

        return src.permute(2,0,1)

class SimilarityEncoder(nn.Module):
    
    def __init__(self, hparams, layer_list="ss", norm=None, return_intermediate=False):
        super().__init__()
        self.hparams = hparams
        self.num_neighs = hparams.num_neighs
        self.latent_dim = hparams.d_feedforward
        self.input_features = 1024
        self.bb_size = self.hparams.bb_size
        
        self.layers = nn.ModuleList([])
        self.num_layers = len(layer_list)
        self.layer_list = layer_list
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.premlp = nn.Sequential(
                nn.Conv1d(self.input_features, 512, kernel_size=1, bias=False), nn.BatchNorm1d(512), nn.LeakyReLU(negative_slope=0.2),
            )

        for i in range(self.num_layers):
            in_features =  512*(i+1)
            if self.layer_list[i] == 's':
                self.layers.append(TemplateSelfLayer(
                in_features, self.hparams.nhead, self.hparams.d_feedforward, self.hparams.dropout,
                activation=self.hparams.transformer_act,
                normalize_before=self.hparams.pre_norm,
                sa_val_has_pos_emb=self.hparams.sa_val_has_pos_emb,
                ca_val_has_pos_emb=self.hparams.ca_val_has_pos_emb,
                ))
            else: 
                assert(self.layer_list[i] in ["s"]), "Please set layer_list only with 's' representing 'self_attention_layer'"
            
        last_in_dim = 512*(self.num_layers+1)
        self.mlp = nn.Sequential(
                nn.Conv1d(last_in_dim, self.latent_dim, kernel_size=1, bias=False), nn.BatchNorm1d(self.latent_dim), nn.LeakyReLU(negative_slope=0.2),
            )

    def compute_neibor_mask(self, neigh):
        with torch.no_grad():
            B, N, _ = neigh.shape
            mask = torch.full((B,N,N), True).to(neigh.device)
            mask.scatter_(2, neigh.long(), False)
        return mask
    
    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                src_xyz: Optional[Tensor] = None, 
                src_neigh: Optional[Tensor] = None,):
        
        src_intermediate = []

        src = self.premlp(src.permute(1,2,0)).permute(2,0,1)
        
        for idx, layer in enumerate(self.layers):
            src_out = layer(src, src_mask=None, 
                             src_key_padding_mask=src_key_padding_mask,
                             src_pos=None)
            
            #print(tgt_out.shape)
            src = torch.cat((src, src_out), dim=2)
            
            if self.return_intermediate:
                src_intermediate.append(src_out.permute(1,2,0))
    
        src = self.mlp(torch.cat(src_intermediate, dim=1))
        
        if self.return_intermediate:
            return src, src_intermediate
        
        return src

    def get_attentions(self):
        """For analysis: Retrieves the attention maps last computed by the individual layers."""

        src_satt_all, tgt_satt_all = [], []
        src_xatt_all, tgt_xatt_all = [], []

        for layer in self.layers:
            src_satt, tgt_satt = layer.satt_weights
            src_xatt, tgt_xatt = layer.xatt_weights

            src_satt_all.append(src_satt)
            tgt_satt_all.append(tgt_satt)
            src_xatt_all.append(src_xatt)
            tgt_xatt_all.append(tgt_xatt)

        src_satt_all = torch.stack(src_satt_all)
        tgt_satt_all = torch.stack(tgt_satt_all)
        src_xatt_all = torch.stack(src_xatt_all)
        tgt_xatt_all = torch.stack(tgt_xatt_all)

        return (src_satt_all, tgt_satt_all), (src_xatt_all, tgt_xatt_all)
    
class SemidropLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 sa_val_has_pos_emb=False,
                 ca_val_has_pos_emb=False,
                 ):
        super().__init__()

        self.nhead = nhead

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)


        # Implementation of Feedforward model
        self.linear = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(d_model)



        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.sa_val_has_pos_emb = sa_val_has_pos_emb
        self.ca_val_has_pos_emb = ca_val_has_pos_emb
        self.satt_weights, self.xatt_weights = None, None  # For analysis

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     src_pos: Optional[Tensor] = None,
                     sim_matrix: Optional[Tensor] = None,):

        '''
        src = [N, B, C]
        src_mask = [B*nheads, N, N]
        src_pos = [N, B, C]
        '''
        # Self attention
        src_w_pos = self.with_pos_embed(src, src_pos)
        q = k = src_w_pos
        src2, satt_weights_s = self.self_attn(q, k,
                              value=src_w_pos if self.sa_val_has_pos_emb else src,
                              attn_mask=src_mask,
                              key_padding_mask= src_key_padding_mask)
        src = src - self.dropout(src2) # N B C
        src = self.linear(src.permute(1,2,0)) # B C N


        src2_sim, satt_weights_s_sim = self.self_attn(q, k,
                              value=sim_matrix ,
                              attn_mask=src_mask,
                              key_padding_mask= src_key_padding_mask)
        
        src_sim = self.linear(src2_sim.permute(1,2,0))
        src = self.norm(src+src_sim) # B C N
        src = self.activation(src) # B C N

        # Stores the attention weights for analysis, if required
        self.satt_weights = (satt_weights_s)

        return src.permute(2,0,1)
    
class SimilarityFusionEncoder(nn.Module):
    def __init__(self, hparams, layer_list, norm=None, return_intermediate=False):
        super().__init__()
        self.hparams = hparams
        self.num_neighs = hparams.num_neighs
        self.latent_dim = hparams.d_feedforward
        
        self.layers = nn.ModuleList([])
        self.num_layers = len(layer_list)
        self.layer_list = layer_list
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.pos_embed = nn.ModuleList([PositionEmbeddingCoordsSine(3, hparams.d_feedforward*(i+1), scale= 1.0) for i in range(self.num_layers)])

        self.premlp1 = nn.Sequential(
                nn.Conv1d(hparams.d_feedforward, hparams.d_feedforward, kernel_size=1, bias=False), nn.BatchNorm1d(hparams.d_feedforward), nn.LeakyReLU(negative_slope=0.2),
            )
        self.premlp2_1 = nn.Sequential(
                nn.Conv1d(2*hparams.d_feedforward, hparams.d_feedforward, kernel_size=1, bias=False), nn.BatchNorm1d(hparams.d_feedforward), nn.LeakyReLU(negative_slope=0.2),
            )
        self.premlp2_2 = nn.Sequential(
                nn.Conv1d(2*hparams.d_feedforward, 2*hparams.d_feedforward, kernel_size=1, bias=False), nn.BatchNorm1d(2*hparams.d_feedforward), nn.LeakyReLU(negative_slope=0.2),
            )

        for i in range(self.num_layers):
            if self.layer_list[i] == 's':
                in_features =  hparams.d_feedforward*(i+1)
                self.layers.append(SemidropLayer(
                in_features, self.hparams.nhead, self.hparams.d_feedforward, 0.2,
                activation=self.hparams.transformer_act,
                normalize_before=self.hparams.pre_norm,
                sa_val_has_pos_emb=self.hparams.sa_val_has_pos_emb,
                ca_val_has_pos_emb=self.hparams.ca_val_has_pos_emb,
                ))
            else: 
                assert(self.layer_list[i] in ["s"]), "Please set layer_list only with 's' representing 'self_attention_layer'"
        
        last_in_dim = 512*(self.num_layers+1)
        self.mlp = nn.Sequential(
                nn.Conv1d(last_in_dim, self.latent_dim, kernel_size=1, bias=False), nn.BatchNorm1d(self.latent_dim), nn.LeakyReLU(negative_slope=0.2),
            )

    def compute_neibor_mask(self, neigh):
        with torch.no_grad():
            B, N, _ = neigh.shape
            mask = torch.full((B,N,N), True).to(neigh.device)
            mask.scatter_(2, neigh.long(), False)
        return mask
    
    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                src_xyz: Optional[Tensor] = None, 
                src_neigh: Optional[Tensor] = None,
                sim_matrix: Optional[Tensor] = None):
        
        src_intermediate = []

        src = self.premlp1(src).permute(2,0,1)   # [B, C, N] ---> src  = [N, B, C]
        sim_embed = [self.premlp2_1(sim_matrix).permute(2,0,1), self.premlp2_2(sim_matrix).permute(2,0,1)]   # [B, C, N] ---> sim_embed  = [N, B, C] 
        
        for idx, layer in enumerate(self.layers):
            
            src_mask = self.compute_neibor_mask(src_neigh)
            # mask must be tiled to num_heads of the transformer
            bsz, n, n = src_mask.shape
            nhead = layer.nhead
            src_mask = src_mask.unsqueeze(1)
            src_mask = src_mask.repeat(1, nhead, 1, 1)
            src_mask = src_mask.view(bsz * nhead, n, n)
            
            src_pos = self.pos_embed[idx](src_xyz.reshape(-1,3)).reshape(-1,src_mask.shape[1], self.latent_dim*(idx+1))
                
            src_out = layer(src, src_mask=src_mask, 
                             src_key_padding_mask=src_key_padding_mask,
                             src_pos=src_pos.transpose(0,1),
                             sim_matrix = sim_embed[idx])
            
            #print(tgt_out.shape)
            src = torch.cat((src, src_out), dim=2)
            
            if self.return_intermediate:
                src_intermediate.append(src_out.permute(1,2,0))
    
        src = self.mlp(torch.cat(src_intermediate, dim=1))
        
        return src

class LuckTransformerEncoder_select_mask(nn.Module):
    
    def __init__(self, hparams, layer_list, norm=None, return_intermediate=False):
        super().__init__()
        self.hparams = hparams
        self.num_neighs = hparams.num_neighs
        self.latent_dim = hparams.d_feedforward
        self.input_features = hparams.in_features_dim
        self.bb_size = self.hparams.bb_size
        
        self.layers = nn.ModuleList([])
        self.num_layers = len(layer_list)
        self.layer_list = layer_list
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.premlp = nn.Sequential(
                nn.Conv1d(self.input_features, self.bb_size * 4, kernel_size=1, bias=False), nn.BatchNorm1d(self.bb_size * 4), nn.LeakyReLU(negative_slope=0.2),
            )
        self.pos_embed = nn.ModuleList([PositionEmbeddingCoordsSine(3, self.bb_size * (2 ** (i+1)) * 2, scale= 1.0) for i in range(self.num_layers)])

        for i in range(self.num_layers):
            in_features =  self.bb_size * (2 ** (i+1)) * 2
            if self.layer_list[i] == 's':
                self.layers.append(LuckSelfLayer(
                in_features, self.hparams.nhead, self.hparams.d_feedforward, self.hparams.dropout,
                activation=self.hparams.transformer_act,
                normalize_before=self.hparams.pre_norm,
                sa_val_has_pos_emb=self.hparams.sa_val_has_pos_emb,
                ca_val_has_pos_emb=self.hparams.ca_val_has_pos_emb,
                ))
            elif self.layer_list[i] == 'c':
                self.layers.append(LuckCrossLayer(
                in_features, self.hparams.nhead, self.hparams.d_feedforward, self.hparams.dropout,
                activation=self.hparams.transformer_act,
                normalize_before=self.hparams.pre_norm,
                sa_val_has_pos_emb=self.hparams.sa_val_has_pos_emb,
                ca_val_has_pos_emb=self.hparams.ca_val_has_pos_emb,
                ))
            else: 
                assert(self.layer_list[i] in ["s", "c"]), "Please set layer_list only with 's' and 'c' representing 'self_attention_layer' and 'cross_attention_layer' respectively"
            
        last_in_dim = self.bb_size * 2 * sum([2 ** i for i in range(1,self.num_layers + 1,1)])
        self.mlp = nn.Sequential(
                nn.Conv1d(last_in_dim, self.latent_dim, kernel_size=1, bias=False), nn.BatchNorm1d(self.latent_dim), nn.LeakyReLU(negative_slope=0.2),
            )

    def compute_neibor_mask(self, neigh):
        with torch.no_grad():
            B, N, _ = neigh.shape
            mask = torch.full((B,N,N), True).to(neigh.device)
            mask.scatter_(2, neigh.long(), False)
        return mask
    
    def forward(self, src, tgt,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                src_xyz: Optional[Tensor] = None, 
                tgt_xyz: Optional[Tensor] = None,
                src_neigh: Optional[Tensor] = None, 
                tgt_neigh: Optional[Tensor] = None,):
        
        src_intermediate, tgt_intermediate = [], []

        src = self.premlp(src.permute(1,2,0)).permute(2,0,1)
        tgt = self.premlp(tgt.permute(1,2,0)).permute(2,0,1)
        
        for idx, layer in enumerate(self.layers):
            
            src_mask = self.compute_neibor_mask(src_neigh)
            tgt_mask = self.compute_neibor_mask(tgt_neigh)
            # mask must be tiled to num_heads of the transformer
            bsz, n_src, n_src = src_mask.shape
            bsz, n_tgt, n_tgt = tgt_mask.shape
            nhead = layer.nhead
            src_mask = src_mask.unsqueeze(1)
            src_mask = src_mask.repeat(1, nhead, 1, 1)
            src_mask = src_mask.view(bsz * nhead, n_src, n_src)
            tgt_mask = tgt_mask.unsqueeze(1)
            tgt_mask = tgt_mask.repeat(1, nhead, 1, 1)
            tgt_mask = tgt_mask.view(bsz * nhead, n_tgt, n_tgt)
            
            src_pos = self.pos_embed[idx](src_xyz.reshape(-1,3)).reshape(-1,src_mask.shape[1], self.bb_size * (2 ** (idx+1)) * 2)
            tgt_pos = self.pos_embed[idx](tgt_xyz.reshape(-1,3)).reshape(-1,tgt_mask.shape[1], self.bb_size * (2 ** (idx+1)) * 2)
                
            src_out, tgt_out = layer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask,
                             src_key_padding_mask=src_key_padding_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             src_pos=src_pos.transpose(0,1), tgt_pos=tgt_pos.transpose(0,1))
            
            #print(tgt_out.shape)
            src = torch.cat((src, src_out), dim=2)
            tgt = torch.cat((tgt, tgt_out), dim=2)
            
            if self.return_intermediate:
                src_intermediate.append(src_out)
                tgt_intermediate.append(tgt_out)

        # if self.return_intermediate:
        #     return torch.stack(src_intermediate), torch.stack(tgt_intermediate)
        
        src = self.mlp(torch.cat(src_intermediate, dim=2).permute(1,2,0))
        tgt = self.mlp(torch.cat(tgt_intermediate, dim=2).permute(1,2,0))
        
        return src, tgt

    def get_attentions(self):
        """For analysis: Retrieves the attention maps last computed by the individual layers."""

        src_satt_all, tgt_satt_all = [], []
        src_xatt_all, tgt_xatt_all = [], []

        for layer in self.layers:
            src_satt, tgt_satt = layer.satt_weights
            src_xatt, tgt_xatt = layer.xatt_weights

            src_satt_all.append(src_satt)
            tgt_satt_all.append(tgt_satt)
            src_xatt_all.append(src_xatt)
            tgt_xatt_all.append(tgt_xatt)

        src_satt_all = torch.stack(src_satt_all)
        tgt_satt_all = torch.stack(tgt_satt_all)
        src_xatt_all = torch.stack(src_xatt_all)
        tgt_xatt_all = torch.stack(tgt_xatt_all)

        return (src_satt_all, tgt_satt_all), (src_xatt_all, tgt_xatt_all)

class LuckSelfLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 sa_val_has_pos_emb=False,
                 ca_val_has_pos_emb=False,
                 ):
        super().__init__()

        self.nhead = nhead

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)


        # Implementation of Feedforward model
        self.linear = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(d_model)



        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.sa_val_has_pos_emb = sa_val_has_pos_emb
        self.ca_val_has_pos_emb = ca_val_has_pos_emb
        self.satt_weights, self.xatt_weights = None, None  # For analysis

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src, tgt,
                     src_mask: Optional[Tensor] = None,
                     tgt_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     src_pos: Optional[Tensor] = None,
                     tgt_pos: Optional[Tensor] = None,):


        # Self attention
        src_w_pos = self.with_pos_embed(src, src_pos)
        q = k = src_w_pos
        src2, satt_weights_s = self.self_attn(q, k,
                              value=src_w_pos if self.sa_val_has_pos_emb else src,
                              attn_mask=src_mask,
                              key_padding_mask= src_key_padding_mask)
        src = src - self.dropout(src2) # N B C
        src = self.linear(src.permute(1,2,0)) # B C N
        src = self.norm(src) # B C N
        src = self.activation(src) # B C N

        tgt_w_pos = self.with_pos_embed(tgt, tgt_pos)
        q = k = tgt_w_pos
        tgt2, satt_weights_t = self.self_attn(q, k,
                                              value=tgt_w_pos if self.sa_val_has_pos_emb else tgt,
                                              attn_mask=tgt_mask,
                                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt - self.dropout(tgt2)
        tgt = self.linear(tgt.permute(1,2,0))
        tgt = self.norm(tgt)
        tgt = self.activation(tgt)

        # Stores the attention weights for analysis, if required
        self.satt_weights = (satt_weights_s, satt_weights_t)

        return src.permute(2,0,1), tgt.permute(2,0,1)

class LuckCrossLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 sa_val_has_pos_emb=False,
                 ca_val_has_pos_emb=False,
                 ):
        super().__init__()

        self.nhead = nhead

        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)


        # Implementation of Feedforward model
        self.linear = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(d_model)



        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.sa_val_has_pos_emb = sa_val_has_pos_emb
        self.ca_val_has_pos_emb = ca_val_has_pos_emb
        self.satt_weights, self.xatt_weights = None, None  # For analysis

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src, tgt,
                     src_mask: Optional[Tensor] = None,
                     tgt_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     src_pos: Optional[Tensor] = None,
                     tgt_pos: Optional[Tensor] = None,):


        # Cross attention
        src_w_pos = self.with_pos_embed(src, src_pos)
        tgt_w_pos = self.with_pos_embed(tgt, tgt_pos)
        
        src2, satt_weights_s = self.cross_attn(query = src_w_pos, key = tgt_w_pos,
                              value=tgt_w_pos if self.sa_val_has_pos_emb else tgt,
                              key_padding_mask= src_key_padding_mask)

        tgt2, satt_weights_t = self.cross_attn(tgt_w_pos, src_w_pos,
                                        value=src_w_pos if self.sa_val_has_pos_emb else src,
                                        key_padding_mask=tgt_key_padding_mask)
        
        src = src - self.dropout(src2) # N B C
        src = self.linear(src.permute(1,2,0)) # B C N
        src = self.norm(src) # B C N
        src = self.activation(src) # B C N
        

        tgt = tgt - self.dropout(tgt2)
        

        tgt = self.linear(tgt.permute(1,2,0))
        tgt = self.norm(tgt)
        tgt = self.activation(tgt)
        
        # Stores the attention weights for analysis, if required
        self.satt_weights = (satt_weights_s, satt_weights_t)
        
        return src.permute(2,0,1), tgt.permute(2,0,1)

class NovelTransformerEncoder(nn.Module):
    
    def __init__(self, hparams, layer_list, norm=None, return_intermediate=False):
        super().__init__()
        self.hparams = hparams
        self.num_neighs = hparams.num_neighs
        self.latent_dim = hparams.d_feedforward
        self.input_features = hparams.in_features_dim
        self.bb_size = self.hparams.bb_size
        
        self.layers = nn.ModuleList([])
        self.num_layers = len(layer_list)
        self.layer_list = layer_list
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.premlp = nn.Sequential(
                nn.Conv1d(self.input_features, self.bb_size * 4, kernel_size=1, bias=False), nn.BatchNorm1d(self.bb_size * 4), nn.LeakyReLU(negative_slope=0.2),
            )
        self.pos_embed = nn.ModuleList([PositionEmbeddingCoordsSine(3, self.bb_size * (2 ** (i+1)) * 2, scale= 1.0) for i in range(self.num_layers)])

        for i in range(self.num_layers):
            in_features =  self.bb_size * (2 ** (i+1)) * 2
            self.layers.append(LuckSelfLayer(
            in_features, self.hparams.nhead, self.hparams.d_feedforward, self.hparams.dropout,
            activation=self.hparams.transformer_act,
            normalize_before=self.hparams.pre_norm,
            sa_val_has_pos_emb=self.hparams.sa_val_has_pos_emb,
            ca_val_has_pos_emb=self.hparams.ca_val_has_pos_emb,
        ))
        last_in_dim = self.bb_size * 2 * sum([2 ** i for i in range(1,self.num_layers + 1,1)])
        self.mlp = nn.Sequential(
                nn.Conv1d(last_in_dim, self.latent_dim, kernel_size=1, bias=False), nn.BatchNorm1d(self.latent_dim), nn.LeakyReLU(negative_slope=0.2),
            )

    def compute_neibor_mask(self, neigh):
        with torch.no_grad():
            B, N, _ = neigh.shape
            mask = torch.full((B,N,N), True).to(neigh.device)
            mask.scatter_(2, neigh.long(), False)
        return mask
    
    def forward(self, src, tgt,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                src_xyz: Optional[Tensor] = None, 
                tgt_xyz: Optional[Tensor] = None,
                src_neigh: Optional[Tensor] = None, 
                tgt_neigh: Optional[Tensor] = None,):

        src_intermediate, tgt_intermediate = [], []

        src = self.premlp(src.permute(1,2,0)).permute(2,0,1)
        tgt = self.premlp(tgt.permute(1,2,0)).permute(2,0,1)
        
        for idx, layer in enumerate(self.layers):
            src_mask = self.compute_neibor_mask(src_neigh)
            tgt_mask = self.compute_neibor_mask(tgt_neigh)
            # mask must be tiled to num_heads of the transformer
            bsz, n, n = src_mask.shape
            nhead = layer.nhead
            src_mask = src_mask.unsqueeze(1)
            src_mask = src_mask.repeat(1, nhead, 1, 1)
            src_mask = src_mask.view(bsz * nhead, n, n)
            tgt_mask = tgt_mask.unsqueeze(1)
            tgt_mask = tgt_mask.repeat(1, nhead, 1, 1)
            tgt_mask = tgt_mask.view(bsz * nhead, n, n)
            
            src_pos = self.pos_embed[idx](src_xyz.reshape(-1,3)).reshape(-1,src_mask.shape[1], self.bb_size * (2 ** (idx+1)) * 2)
            tgt_pos = self.pos_embed[idx](tgt_xyz.reshape(-1,3)).reshape(-1,tgt_mask.shape[1], self.bb_size * (2 ** (idx+1)) * 2)
                
            src_out, tgt_out = layer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask,
                             src_key_padding_mask=src_key_padding_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             src_pos=src_pos.transpose(0,1), tgt_pos=tgt_pos.transpose(0,1))
            
            src = torch.cat((src, src_out), dim=2)
            tgt = torch.cat((tgt, tgt_out), dim=2)
            
            if self.return_intermediate:
                src_intermediate.append(src_out)
                tgt_intermediate.append(tgt_out)

        # if self.return_intermediate:
        #     return torch.stack(src_intermediate), torch.stack(tgt_intermediate)

        src = self.mlp(torch.cat(src_intermediate, dim=2).permute(1,2,0))
        tgt = self.mlp(torch.cat(tgt_intermediate, dim=2).permute(1,2,0))
        
        return src, tgt

    def get_attentions(self):
        """For analysis: Retrieves the attention maps last computed by the individual layers."""

        src_satt_all, tgt_satt_all = [], []
        src_xatt_all, tgt_xatt_all = [], []

        for layer in self.layers:
            src_satt, tgt_satt = layer.satt_weights
            src_xatt, tgt_xatt = layer.xatt_weights

            src_satt_all.append(src_satt)
            tgt_satt_all.append(tgt_satt)
            src_xatt_all.append(src_xatt)
            tgt_xatt_all.append(tgt_xatt)

        src_satt_all = torch.stack(src_satt_all)
        tgt_satt_all = torch.stack(tgt_satt_all)
        src_xatt_all = torch.stack(src_xatt_all)
        tgt_xatt_all = torch.stack(tgt_xatt_all)

        return (src_satt_all, tgt_satt_all), (src_xatt_all, tgt_xatt_all)

class NovelSelfLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 sa_val_has_pos_emb=False,
                 ca_val_has_pos_emb=False,
                 ):
        super().__init__()

        self.nhead = nhead

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)


        # Implementation of Feedforward model
        self.linear = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(d_model)



        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.sa_val_has_pos_emb = sa_val_has_pos_emb
        self.ca_val_has_pos_emb = ca_val_has_pos_emb
        self.satt_weights, self.xatt_weights = None, None  # For analysis

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src, tgt,
                     src_mask: Optional[Tensor] = None,
                     tgt_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     src_pos: Optional[Tensor] = None,
                     tgt_pos: Optional[Tensor] = None,):


        # Self attention
        src_w_pos = self.with_pos_embed(src, src_pos)
        q = k = src_w_pos
        src2, satt_weights_s = self.self_attn(q, k,
                              value=src_w_pos if self.sa_val_has_pos_emb else src,
                              attn_mask=src_mask,
                              key_padding_mask= src_key_padding_mask)
        src = src - self.dropout(src2) # N B C
        src = self.linear(src.permute(1,2,0)) # B C N
        src = self.norm(src) # B C N
        src = self.activation(src) # B C N

        tgt_w_pos = self.with_pos_embed(tgt, tgt_pos)
        q = k = tgt_w_pos
        tgt2, satt_weights_t = self.self_attn(q, k,
                                              value=tgt_w_pos if self.sa_val_has_pos_emb else tgt,
                                              attn_mask=tgt_mask,
                                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt - self.dropout(tgt2)
        tgt = self.linear(tgt.permute(1,2,0))
        tgt = self.norm(tgt)
        tgt = self.activation(tgt)

        # Stores the attention weights for analysis, if required
        self.satt_weights = (satt_weights_s, satt_weights_t)

        return src.permute(2,0,1), tgt.permute(2,0,1)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _flex_clones(self_layer, cross_layer, layer_list):
    module_list = nn.ModuleList([])
    for i in layer_list:
        if i == "s":
            module_list.append(copy.deepcopy(self_layer))
        elif i == "c":
            module_list.append(copy.deepcopy(cross_layer))
        else:
            assert(i in ["s", "c"]), "Please set layer_list only with 's' and 'c' representing 'self_attention_layer' and 'cross_attention_layer' respectively"
    return module_list

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class Decoder(nn.Module):
    def __init__(self, num_layers,d_model, nhead, dropout=0.0,
                    activation="relu", normalize_before=False):
            super().__init__()
            self.tmp_cross_attention = nn.ModuleList()
            self.tmp_self_attention = nn.ModuleList()
            self.tmp_ffn_attention = nn.ModuleList()
            self.num_layers = num_layers
            for i in range(num_layers):
                # self.tmp_cross_attention.append(
                #     CrossAttentionLayer(
                #         d_model=d_model,
                #         nhead=nhead,
                #         dropout=dropout,
                #         normalize_before=normalize_before
                #     )
                # )
                self.tmp_self_attention.append(
                    SelfAttentionLayer(
                        d_model=d_model,
                        nhead=nhead,
                        dropout=dropout,
                        normalize_before=normalize_before
                    )
                )

                self.tmp_ffn_attention.append(
                    FFNLayer(
                        d_model=d_model,
                        dim_feedforward=d_model,
                        dropout=dropout,
                        normalize_before=normalize_before
                    )
                )

    def forward(self,src_out,pos_src,mask_out,pos_mask): #query 4 1024 512 query_pos1024 4 512
            for i in range(self.num_layers):
                #mask_out = self.tmp_cross_attention[i](mask_out.permute(2,0,1),src_out.permute(2,0,1),pos=pos_src.permute(1,0,2),query_pos=pos_mask.permute(1,0,2))
                mask_out = self.tmp_self_attention[i](mask_out.permute(2,0,1),query_pos=pos_mask.permute(1,0,2))
                mask_out = self.tmp_ffn_attention[i](mask_out).permute((1,2,0))
            return mask_out

class Encoder(nn.Module):
    def __init__(self, num_layers,d_model, nhead, dropout=0.0,
                    activation="relu", normalize_before=False):
            super().__init__()
            self.tmp_cross_attention = nn.ModuleList()
            self.tmp_self_attention = nn.ModuleList()
            self.tmp_ffn_attention = nn.ModuleList()
            self.num_layers = num_layers
            for i in range(num_layers):
                self.tmp_cross_attention.append(
                    CrossAttentionLayer(
                        d_model=d_model,
                        nhead=nhead,
                        dropout=dropout,
                        normalize_before=normalize_before
                    )
                )
                self.tmp_self_attention.append(
                    SelfAttentionLayer(
                        d_model=d_model,
                        nhead=nhead,
                        dropout=dropout,
                        normalize_before=normalize_before
                    )
                )

                self.tmp_ffn_attention.append(
                    FFNLayer(
                        d_model=d_model,
                        dim_feedforward=d_model,
                        dropout=dropout,
                        normalize_before=normalize_before
                    )
                )

    def forward(self,src_out,pos_src,mask_out,pos_mask,mask_num): #query 4 1024 512 query_pos1024 4 512
            # for i in range(self.num_layers):
            #     #mask_out = self.tmp_cross_attention[i](mask_out.permute(2,0,1),src_out.permute(2,0,1),pos=pos_src.permute(1,0,2),query_pos=pos_mask.permute(1,0,2))
            #     mask_out = self.tmp_self_attention[i](mask_out.permute(2,0,1),query_pos=pos_mask.permute(1,0,2))
            #     mask_out = self.tmp_ffn_attention[i](mask_out).permute((1,2,0))
            #     # mask_out = self.tmp_self_attention[i](mask_out.permute(2,0,1),query_pos=pos_mask.permute(1,0,2))
            #     # mask_out = self.tmp_cross_attention[i](mask_out,src_out.permute(2,0,1))
            #     # mask_out = self.tmp_ffn_attention[i](mask_out).permute((1,2,0))
            #print(mask_num)
            for i in range(self.num_layers):
                mask_out1 = self.tmp_cross_attention[i](mask_out[...,mask_num:].permute(2,0,1),src_out.permute(2,0,1),pos=pos_src.permute(1,0,2),query_pos=pos_mask[:,mask_num:,:].permute(1,0,2))
                mask_out = torch.cat([mask_out[...,:mask_num].permute(2,0,1),mask_out1], dim=0).permute((1,2,0))

            for i in range(self.num_layers):
                mask_out = self.tmp_self_attention[i](mask_out.permute(2,0,1),query_pos=pos_mask.permute(1,0,2))  
                mask_out = self.tmp_ffn_attention[i](mask_out).permute((1,2,0))

            return mask_out
        
class AlternateEncoder(nn.Module):
    def __init__(self, num_layers,d_model, nhead, dropout=0.0,
                    activation="relu", normalize_before=False):
            super().__init__()
            self.tmp_cross_attention = nn.ModuleList()
            self.tmp_self_attention = nn.ModuleList()
            self.tmp_ffn_attention = nn.ModuleList()
            self.num_layers = num_layers
            for i in range(num_layers):
                self.tmp_cross_attention.append(
                    CrossAttentionLayer(
                        d_model=d_model,
                        nhead=nhead,
                        dropout=dropout,
                        normalize_before=normalize_before
                    )
                )
                self.tmp_self_attention.append(
                    SelfAttentionLayer(
                        d_model=d_model,
                        nhead=nhead,
                        dropout=dropout,
                        normalize_before=normalize_before
                    )
                )

                self.tmp_ffn_attention.append(
                    FFNLayer(
                        d_model=d_model,
                        dim_feedforward=d_model,
                        dropout=dropout,
                        normalize_before=normalize_before
                    )
                )

    def forward(self,src_out,pos_src,mask_out,pos_mask,mask_num): #query 4 1024 512 query_pos1024 4 512
            # for i in range(self.num_layers):
            #     #mask_out = self.tmp_cross_attention[i](mask_out.permute(2,0,1),src_out.permute(2,0,1),pos=pos_src.permute(1,0,2),query_pos=pos_mask.permute(1,0,2))
            #     mask_out = self.tmp_self_attention[i](mask_out.permute(2,0,1),query_pos=pos_mask.permute(1,0,2))
            #     mask_out = self.tmp_ffn_attention[i](mask_out).permute((1,2,0))
            #     # mask_out = self.tmp_self_attention[i](mask_out.permute(2,0,1),query_pos=pos_mask.permute(1,0,2))
            #     # mask_out = self.tmp_cross_attention[i](mask_out,src_out.permute(2,0,1))
            #     # mask_out = self.tmp_ffn_attention[i](mask_out).permute((1,2,0))
            #print(mask_num)
            for i in range(self.num_layers):
                mask_out1 = self.tmp_cross_attention[i](mask_out[...,mask_num:].permute(2,0,1),src_out.permute(2,0,1),pos=pos_src.permute(1,0,2),query_pos=pos_mask[:,mask_num:,:].permute(1,0,2))
                mask_out = torch.cat([mask_out[...,:mask_num].permute(2,0,1),mask_out1], dim=0).permute((1,2,0))

                mask_out = self.tmp_self_attention[i](mask_out.permute(2,0,1),query_pos=pos_mask.permute(1,0,2))  
                mask_out = self.tmp_ffn_attention[i](mask_out).permute((1,2,0))

            return mask_out
        
class TogetherEncoder(nn.Module):
    def __init__(self, num_layers,d_model, nhead, dropout=0.0,
                    activation="relu", normalize_before=False):
            super().__init__()
            self.tmp_cross_attention = nn.ModuleList()
            self.tmp_self_attention = nn.ModuleList()
            self.tmp_ffn_attention = nn.ModuleList()
            self.num_layers = num_layers
            for i in range(num_layers):
                self.tmp_cross_attention.append(
                    CrossAttentionLayer(
                        d_model=d_model,
                        nhead=nhead,
                        dropout=dropout,
                        normalize_before=normalize_before
                    )
                )
                self.tmp_self_attention.append(
                    SelfAttentionLayer(
                        d_model=d_model,
                        nhead=nhead,
                        dropout=dropout,
                        normalize_before=normalize_before
                    )
                )

                self.tmp_ffn_attention.append(
                    FFNLayer(
                        d_model=d_model,
                        dim_feedforward=d_model,
                        dropout=dropout,
                        normalize_before=normalize_before
                    )
                )

    def forward(self,src_out,pos_src,mask_out,pos_mask,mask_num): #query 4 1024 512 query_pos1024 4 512
            # for i in range(self.num_layers):
            #     #mask_out = self.tmp_cross_attention[i](mask_out.permute(2,0,1),src_out.permute(2,0,1),pos=pos_src.permute(1,0,2),query_pos=pos_mask.permute(1,0,2))
            #     mask_out = self.tmp_self_attention[i](mask_out.permute(2,0,1),query_pos=pos_mask.permute(1,0,2))
            #     mask_out = self.tmp_ffn_attention[i](mask_out).permute((1,2,0))
            #     # mask_out = self.tmp_self_attention[i](mask_out.permute(2,0,1),query_pos=pos_mask.permute(1,0,2))
            #     # mask_out = self.tmp_cross_attention[i](mask_out,src_out.permute(2,0,1))
            #     # mask_out = self.tmp_ffn_attention[i](mask_out).permute((1,2,0))
            #print(mask_num)
            for i in range(self.num_layers):
                mask_out1 = self.tmp_cross_attention[i](mask_out.permute(2,0,1),src_out.permute(2,0,1),pos=pos_src.permute(1,0,2),query_pos=pos_mask.permute(1,0,2))
                mask_out = mask_out1.permute((1,2,0))

            for i in range(self.num_layers):
                mask_out = self.tmp_self_attention[i](mask_out.permute(2,0,1),query_pos=pos_mask.permute(1,0,2))  
                mask_out = self.tmp_ffn_attention[i](mask_out).permute((1,2,0))

            return mask_out

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask = None,
                     tgt_key_padding_mask= None,
                     query_pos = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask= None,
                    tgt_key_padding_mask = None,
                    query_pos = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask = None,
                tgt_key_padding_mask = None,
                query_pos = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask = None,
                     memory_key_padding_mask = None,
                     pos = None,
                     query_pos = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask = None,
                    memory_key_padding_mask = None,
                    pos = None,
                    query_pos = None):
        tgt2 = self.norm(tgt)

        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask = None,
                memory_key_padding_mask = None,
                pos = None,
                query_pos = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)

class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)
