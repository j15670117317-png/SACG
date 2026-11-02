# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .multihead_attention import MultiheadAttentionStream as MultiheadAttention


class ContraNorm1D(nn.Module):
    """ContraNorm for sequence-first Transformer tokens.

    Input/output shape is [T, B, C]. The contrastive normalizer is applied
    across the temporal/token dimension for each batch item.
    """

    def __init__(self, dim, scale=0.1, dual_norm=False, temp=1.0,
                 learnable=False, positive=False, identity=False, pre_norm=False):
        super(ContraNorm1D, self).__init__()
        self.scale = float(scale)
        self.dual_norm = bool(dual_norm)
        self.temp = float(temp)
        self.learnable = bool(learnable)
        self.positive = bool(positive)
        self.identity = bool(identity)
        self.pre_norm = bool(pre_norm)
        if self.learnable and self.scale > 0.0:
            import math
            scale_init = math.log(self.scale) if self.positive else self.scale
            self.scale_param = nn.Parameter(torch.empty(dim).fill_(scale_init))
        else:
            self.register_parameter('scale_param', None)
        self.layernorm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        # x: [T, B, C] -> [B, T, C]
        if self.scale > 0.0:
            xb = x.transpose(0, 1)
            xn = F.normalize(xb, dim=2)
            if self.pre_norm:
                xb = xn
            sim = torch.bmm(xn, xn.transpose(1, 2)) / self.temp
            if self.dual_norm:
                sim = F.softmax(sim, dim=2) + F.softmax(sim, dim=1)
            else:
                sim = F.softmax(sim, dim=2)
            x_neg = torch.bmm(sim, xb)
            if self.scale_param is None:
                if self.identity:
                    xb = (1.0 + self.scale) * xb - self.scale * x_neg
                else:
                    xb = xb - self.scale * x_neg
            else:
                scale = torch.exp(self.scale_param) if self.positive else self.scale_param
                scale = scale.view(1, 1, -1)
                if self.identity:
                    xb = scale * xb - scale * x_neg
                else:
                    xb = xb - scale * x_neg
            x = xb.transpose(0, 1)
        return self.layernorm(x)


def _build_contranorm(dim, position_cfg, position_name, scale=0.1, dual_norm=False,
                      temp=1.0, learnable=False, positive=False, identity=False,
                      pre_norm=False):
    position_cfg = (position_cfg or 'none').lower()
    aliases = {
        'all': {'self', 'cross', 'ffn'},
        'self_cross': {'self', 'cross'},
        'cross_self': {'self', 'cross'},
        'self': {'self'},
        'cross': {'cross'},
        'ffn': {'ffn'},
        'none': set(),
        '': set(),
    }
    active = aliases.get(position_cfg, {p.strip() for p in position_cfg.split(',') if p.strip()})
    if position_name not in active:
        return None
    return ContraNorm1D(
        dim, scale=scale, dual_norm=dual_norm, temp=temp, learnable=learnable,
        positive=positive, identity=identity, pre_norm=pre_norm)



def _normalize_expert_names(experts):
    """Normalize expert names for MultiQueryMoECrossAttention."""
    if experts is None:
        return ['global', 'recent', 'boundary', 'action']
    if isinstance(experts, str):
        experts = [e.strip() for e in experts.replace(';', ',').split(',') if e.strip()]
    experts = [str(e).lower() for e in experts]
    aliases = {
        'g': 'global',
        'ctx': 'global',
        'context': 'global',
        'r': 'recent',
        'local': 'recent',
        'short': 'recent',
        'b': 'boundary',
        'delta': 'boundary',
        'transition': 'boundary',
        'a': 'action',
        'fg': 'action',
        'foreground': 'action',
        'actionness': 'action',
    }
    out = []
    for e in experts:
        e = aliases.get(e, e)
        if e not in {'global', 'recent', 'boundary', 'action'}:
            raise RuntimeError('MQ_MOE_EXPERTS only supports global/recent/boundary/action, got {}'.format(e))
        if e not in out:
            out.append(e)
    if len(out) == 0:
        raise RuntimeError('GEN_MQ_MOE_EXPERTS must contain at least one expert')
    return out


class MultiQueryMoECrossAttention(nn.Module):
    """Query-dependent multi-expert cross attention for TDU.

    The module keeps CMeRT's sequence-first tensor format and returns one fused
    cross-attention output with shape [Lq, B, D]. Each expert receives the same
    query but attends to a different view of the memory:
      - global:   full memory
      - recent:   last K memory tokens
      - boundary: first-order temporal difference memory
      - action:   actionness-gated foreground memory

    A soft query-dependent router fuses expert outputs at every query token.
    This is intentionally dense/soft rather than top-k sparse, because THUMOS is
    small and the goal is stable ablation rather than large-scale MoE training.
    """

    def __init__(self, d_model, nhead, dropout=0.1,
                 attention_type='dotproduct', decay_alpha=1.0,
                 dilate_ratio=[1, 2, 4, 8], assa_tau=1.0, assa_init_w=None,
                 experts=None, recent_k=32, gate_temperature=1.0,
                 boundary_abs=False):
        super(MultiQueryMoECrossAttention, self).__init__()
        self.experts = _normalize_expert_names(experts)
        self.num_experts = len(self.experts)
        self.recent_k = int(recent_k)
        self.gate_temperature = float(gate_temperature)
        self.boundary_abs = bool(boundary_abs)

        self.expert_attns = nn.ModuleDict()
        for name in self.experts:
            self.expert_attns[name] = MultiheadAttention(
                d_model, nhead, dropout=dropout,
                attention_type=attention_type,
                decay_alpha=decay_alpha,
                dilate_ratio=dilate_ratio,
                assa_tau=assa_tau,
                assa_init_w=assa_init_w)

        # Lightweight view-specific adapters. They are identity-initialized as
        # much as possible by using residual-friendly small modules.
        if 'boundary' in self.experts:
            self.boundary_proj = nn.Linear(d_model, d_model)
        else:
            self.boundary_proj = None
        if 'action' in self.experts:
            self.action_gate = nn.Linear(d_model, 1)
        else:
            self.action_gate = None

        self.router = nn.Linear(d_model, self.num_experts)
        # Zero init gives uniform routing at the beginning, preventing early
        # expert collapse and preserving near-baseline behavior.
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)

    def clear_cache(self):
        for attn in self.expert_attns.values():
            if hasattr(attn, 'clear_cache'):
                attn.clear_cache()

    @staticmethod
    def _slice_last(x, start):
        if x is None:
            return None
        return x[..., start:]

    def _memory_view(self, name, memory, memory_mask=None, memory_key_padding_mask=None):
        if name == 'global':
            return memory, memory_mask, memory_key_padding_mask

        if name == 'recent':
            k = min(max(self.recent_k, 1), memory.shape[0])
            start = memory.shape[0] - k
            recent_memory = memory[start:]
            recent_mask = self._slice_last(memory_mask, start)
            recent_key_padding_mask = self._slice_last(memory_key_padding_mask, start)
            return recent_memory, recent_mask, recent_key_padding_mask

        if name == 'boundary':
            delta = torch.zeros_like(memory)
            if memory.shape[0] > 1:
                delta[1:] = memory[1:] - memory[:-1]
            if self.boundary_abs:
                delta = delta.abs()
            delta = self.boundary_proj(delta)
            return delta, memory_mask, memory_key_padding_mask

        if name == 'action':
            action_score = torch.sigmoid(self.action_gate(memory))
            fg_memory = memory * action_score
            return fg_memory, memory_mask, memory_key_padding_mask

        raise RuntimeError('Unsupported expert {}'.format(name))

    def forward(self, tgt, memory, attn_mask=None, key_padding_mask=None,
                knn=False, ratio=0.75):
        expert_outputs = []
        for name in self.experts:
            mem, mask, padding = self._memory_view(
                name, memory, memory_mask=attn_mask,
                memory_key_padding_mask=key_padding_mask)
            out = self.expert_attns[name](
                tgt, mem, mem, attn_mask=mask,
                key_padding_mask=padding, knn=knn, ratio=ratio)[0]
            expert_outputs.append(out)

        # [Lq, B, D, E]
        expert_outputs = torch.stack(expert_outputs, dim=-1)
        gate_logits = self.router(tgt)
        if self.gate_temperature > 0:
            gate_logits = gate_logits / self.gate_temperature
        gate = F.softmax(gate_logits, dim=-1)
        output = (expert_outputs * gate.unsqueeze(2)).sum(dim=-1)
        return output, gate

    def stream_inference(self, tgt, memory, pos=None, attn_mask=None, key_padding_mask=None,
                         cache_num=1, cache_id=0):
        # Streaming fallback. The memory views are recomputed each step; use
        # MODEL.LSTR.INFERENCE_MODE batch for exact high-throughput evaluation.
        return self.forward(tgt, memory, attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask, knn=False, ratio=0.75)


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation='relu', custom_encoder=None, custom_decoder=None):
        super(Transformer, self).__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            encoder_norm = nn.LayerNorm(d_model)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            decoder_norm = nn.LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        if src.size(1) != tgt.size(1):
            raise RuntimeError('the batch number of src and tgt must be equal')

        if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
            raise RuntimeError('the feature number of src and tgt must be equal to d_model')

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()

        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, src_key_padding_mask=None, knn=False, ratio = 0.75):
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=src_mask,
                         src_key_padding_mask=src_key_padding_mask, knn=knn, ratio=ratio)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()

        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def clear_cache(self):
        if len(self.layers) != 1:
            raise RuntimeError('Number of layers cannot larger than 1 for stream inference')

        self.layers[0].clear_cache()

    def stream_inference(self, tgt, memory, pos, tgt_mask=None,
                         memory_mask=None, tgt_key_padding_mask=None,
                         memory_key_padding_mask=None, cache_num=1, cache_id=0):
        output = tgt

        if len(self.layers) != 1:
            raise RuntimeError('Number of layers cannot larger than 1 for stream inference')

        output = self.layers[0].stream_inference(output, memory, pos,
                                                 tgt_mask=tgt_mask,
                                                 memory_mask=memory_mask,
                                                 tgt_key_padding_mask=tgt_key_padding_mask,
                                                 memory_key_padding_mask=memory_key_padding_mask,
                                                 cache_num=cache_num, cache_id=cache_id)

        if self.norm is not None:
            output = self.norm(output)

        return output

    def forward(self, tgt, memory, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None, knn=False, ratio = 0.75):
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask,
                         knn=knn, ratio=ratio)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu'):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, knn=False, ratio = 0.75):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask, knn=knn, ratio=ratio)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation='relu', attention_type='dotproduct',
                 decay_alpha=1.0, dilate_ratio = [1, 2, 4, 8], assa_tau=1.0,
                 assa_init_w=None, self_attention_type=None, cross_attention_type=None,
                 self_assa_init_w=None, cross_assa_init_w=None,
                 contranorm_pos='none', contranorm_scale=0.0,
                 contranorm_dual_norm=False, contranorm_temp=1.0,
                 contranorm_learnable=False, contranorm_positive=False,
                 contranorm_identity=False, contranorm_pre_norm=False,
                 mq_moe_cross=False, mq_moe_experts=None, mq_moe_recent_k=32,
                 mq_moe_gate_temperature=1.0, mq_moe_boundary_abs=False,assa_v2_hidden=8,
        assa_v2_bound=0.1):
        super(TransformerDecoderLayer, self).__init__()

        # Backward-compatible default: if the caller only sets attention_type,
        # both self-attention and cross-attention use that type. New callers can
        # set them independently so causal self-attention can stay dense while
        # cross-memory retrieval uses ASSA.
        self_attention_type = self_attention_type or attention_type
        cross_attention_type = cross_attention_type or attention_type
        self_assa_init_w = self_assa_init_w if self_assa_init_w is not None else assa_init_w
        cross_assa_init_w = cross_assa_init_w if cross_assa_init_w is not None else assa_init_w
        if cross_attention_type is None:
            cross_attention_type = attention_type
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout,
                                            attention_type=self_attention_type, dilate_ratio=dilate_ratio,
                                            assa_tau=assa_tau, assa_init_w=self_assa_init_w)
        self.use_mq_moe_cross = bool(mq_moe_cross)
        if self.use_mq_moe_cross:
            self.multihead_attn = MultiQueryMoECrossAttention(
                d_model, nhead, dropout=dropout,
                attention_type=cross_attention_type,
                decay_alpha=decay_alpha, dilate_ratio=dilate_ratio,
                assa_tau=assa_tau, assa_init_w=cross_assa_init_w,
                experts=mq_moe_experts, recent_k=mq_moe_recent_k,
                gate_temperature=mq_moe_gate_temperature,
                boundary_abs=mq_moe_boundary_abs,assa_v2_hidden=assa_v2_hidden,
            assa_v2_bound=assa_v2_bound,)
        else:
            self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout,
                                                     attention_type=cross_attention_type,
                                                     decay_alpha=decay_alpha, dilate_ratio=dilate_ratio,
                                                     assa_tau=assa_tau, assa_init_w=cross_assa_init_w,assa_v2_hidden=assa_v2_hidden,
            assa_v2_bound=assa_v2_bound,)

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

        self.contranorm_self = _build_contranorm(
            d_model, contranorm_pos, 'self', scale=contranorm_scale,
            dual_norm=contranorm_dual_norm, temp=contranorm_temp,
            learnable=contranorm_learnable, positive=contranorm_positive,
            identity=contranorm_identity, pre_norm=contranorm_pre_norm)
        self.contranorm_cross = _build_contranorm(
            d_model, contranorm_pos, 'cross', scale=contranorm_scale,
            dual_norm=contranorm_dual_norm, temp=contranorm_temp,
            learnable=contranorm_learnable, positive=contranorm_positive,
            identity=contranorm_identity, pre_norm=contranorm_pre_norm)
        self.contranorm_ffn = _build_contranorm(
            d_model, contranorm_pos, 'ffn', scale=contranorm_scale,
            dual_norm=contranorm_dual_norm, temp=contranorm_temp,
            learnable=contranorm_learnable, positive=contranorm_positive,
            identity=contranorm_identity, pre_norm=contranorm_pre_norm)

        self.activation = _get_activation_fn(activation)

        ############################
        # Cache for stream inference
        ############################
        self.tgt_cache = None

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def clear_cache(self):
        self.tgt_cache = None
        self.multihead_attn.clear_cache()

    def stream_inference(self, tgt, memory, pos, tgt_mask=None, memory_mask=None,
                         tgt_key_padding_mask=None, memory_key_padding_mask=None,
                         cache_num=1, cache_id=0):
        if self.tgt_cache is None:
            tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
            if self.contranorm_self is not None:
                tgt = self.contranorm_self(tgt)
            self.tgt_cache = tgt
        else:
            tgt = self.tgt_cache
        if self.use_mq_moe_cross:
            tgt2 = self.multihead_attn.stream_inference(
                tgt, memory, pos, attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                cache_num=cache_num, cache_id=cache_id)[0]
        else:
            tgt2 = self.multihead_attn.stream_inference(tgt, memory, memory, pos, attn_mask=memory_mask,
                                                        key_padding_mask=memory_key_padding_mask,
                                                        cache_num=cache_num, cache_id=cache_id)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        if self.contranorm_cross is not None:
            tgt = self.contranorm_cross(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        if self.contranorm_ffn is not None:
            tgt = self.contranorm_ffn(tgt)
        return tgt

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                knn=False, ratio = 0.75):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask, knn=False)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        if self.contranorm_self is not None:
            tgt = self.contranorm_self(tgt)
        if self.use_mq_moe_cross:
            tgt2 = self.multihead_attn(
                tgt, memory, attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask, knn=knn, ratio=ratio)[0]
        else:
            tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask, knn=knn, ratio=ratio)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        if self.contranorm_cross is not None:
            tgt = self.contranorm_cross(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        if self.contranorm_ffn is not None:
            tgt = self.contranorm_ffn(tgt)
        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return F.gelu

    raise RuntimeError('activation should be relu/gelu, not {}'.format(activation))
