"""DepMamba implementation.
Authors
-------
* Jiaxin Ye 2024
"""

import warnings
from dataclasses import dataclass
from typing import List, Optional

import abc
import torch
import torch.nn as nn
import torch.nn.functional as F

import speechbrain as sb
from speechbrain.nnet.activations import Swish
from speechbrain.nnet.attention import (
    MultiheadAttention,
    PositionalwiseFeedForward,
    RelPosMHAXL,
)
from speechbrain.nnet.hypermixing import HyperMixing
from speechbrain.nnet.normalization import LayerNorm
from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig

# Mamba
from mamba_ssm import Mamba
from .mamba.bimamba import Mamba as BiMamba 
from .mamba.mm_bimamba import Mamba as MMBiMamba 
from .base import BaseNet
from .dynamic_temporal import DynamicTemporalRefinement, bounded_gamma_parameter



class MMMambaEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        d_ffn,
        activation='Swish',
        dropout=0.0,
        causal=False,
        mamba_config=None
    ):
        super().__init__()
        assert mamba_config != None

        if activation == 'Swish':
            activation = Swish
        elif activation == "GELU":
            activation = torch.nn.GELU
        else:
            activation = Swish

        bidirectional = mamba_config.pop('bidirectional')
        if causal or (not bidirectional):
            self.mamba = Mamba(
                d_model=d_model,
                **mamba_config
            )
        else:
            self.mamba = MMBiMamba(
                d_model=d_model,
                bimamba_type='v2',
                **mamba_config
            )
        mamba_config['bidirectional'] = bidirectional

        self.norm1 = LayerNorm(d_model, eps=1e-6)
        self.norm2 = LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)

        self.a_downsample = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=16, stride=2, padding=8),
            nn.BatchNorm1d(d_model),
        )

    def forward(
        self,
        a_x, v_x, 
        a_inference_params = None,
        v_inference_params = None
    ):
        
        a_out1, v_out1 = self.mamba(a_x, v_x,a_inference_params,v_inference_params)
        a_out = a_x + self.norm1(a_out1)
        v_out = v_x + self.norm2(v_out1)

        return a_out, v_out

class MMCNNEncoderLayer(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        dropout=0.0,
        causal=False,
        dilation=1,
    ):
        super().__init__()

        self.a_conv = nn.Conv1d(input_size, output_size, 3, padding=1, dilation=dilation, bias=False)
        self.a_bn = nn.BatchNorm1d(output_size)

        self.v_conv = nn.Conv1d(input_size, output_size, 3, padding=1, dilation=dilation, bias=False)
        self.v_bn = nn.BatchNorm1d(output_size)

        self.relu = nn.ReLU()

        self.a_drop = nn.Dropout(dropout)
        self.v_drop = nn.Dropout(dropout)

        self.a_net = nn.Sequential(self.a_conv, self.a_bn, self.relu, self.a_drop)
        self.v_net = nn.Sequential(self.v_conv, self.v_bn, self.relu, self.v_drop)

        if input_size != output_size:
            self.a_skipconv = nn.Conv1d(input_size, output_size, 1, padding=0, dilation=dilation, bias=False)
            self.v_skipconv = nn.Conv1d(input_size, output_size, 1, padding=0, dilation=dilation, bias=False)
        else:
            self.a_skipconv = None
            self.v_skipconv = None

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.a_conv.weight.data)
        nn.init.xavier_uniform_(self.v_conv.weight.data)
        # nn.init.xavier_uniform_(self.conv2.weight.data)

    def forward(self, xa, xv):
        a_out = self.a_net(xa)
        v_out = self.v_net(xv)
        if self.a_skipconv is not None:
            xa = self.a_skipconv(xa)
        if self.v_skipconv is not None:
            xv = self.v_skipconv(xv)
        a_out = a_out+xa
        v_out = v_out+xv
        return a_out, v_out

class MambaEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        d_ffn,
        activation='Swish',
        dropout=0.0,
        causal=False,
        mamba_config=None
    ):
        super().__init__()
        assert mamba_config != None

        if activation == 'Swish':
            activation = Swish
        elif activation == "GELU":
            activation = torch.nn.GELU
        else:
            activation = Swish

        bidirectional = mamba_config.pop('bidirectional')
        if causal or (not bidirectional):
            self.mamba = Mamba(
                d_model=d_model,
                **mamba_config
            )
        else:
            self.mamba = BiMamba(
                d_model=d_model,
                bimamba_type='v2',
                **mamba_config
            )
        mamba_config['bidirectional'] = bidirectional

        self.norm1 = LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)


    def forward(
        self,
        x, inference_params = None
    ):
        out = x + self.norm1(self.mamba(x, inference_params))
        return out

class CNNEncoderLayer(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        dropout=0.0,
        causal=False,
        dilation=1,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(input_size, output_size, 3, padding=1, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(output_size)
        self.relu1 = nn.ReLU()
        # self.conv2 = nn.Conv1d(output_size, output_size, 5, padding=2, dilation=dilation, bias=False)
        # self.bn2 = nn.BatchNorm1d(output_size)
        # self.relu2 = nn.ReLU()

        self.drop = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.bn1, self.relu1, self.drop)

        if input_size != output_size:
            self.conv = nn.Conv1d(input_size, output_size, 1, padding=0, dilation=dilation, bias=False)
        else:
            self.conv = None
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.conv1.weight.data)
        # nn.init.xavier_uniform_(self.conv2.weight.data)

    def forward(self, x):
        out = self.net(x)
        if self.conv is not None:
            x = self.conv(x)
        out = out+x
        return out

class CoSSM(nn.Module):
    """This class implements the CoSSM encoder.
    """
    def __init__(
        self,
        num_layers,
        input_size,
        output_sizes=[256,512,512],
        d_ffn=1024,
        activation='Swish',
        dropout=0.0,
        kernel_size = 3,
        causal=False,
        mamba_config=None
    ):
        super().__init__()
        prev_input_size = input_size

        cnn_list = []
        mamba_list = []
        # print(output_sizes)
        for i in range(len(output_sizes)):
            cnn_list.append(MMCNNEncoderLayer(
                    input_size = input_size if i<1 else output_sizes[i-1],
                    output_size = output_sizes[i],
                    dropout=dropout
                ))
            mamba_list.append(MMMambaEncoderLayer(
                    d_model=output_sizes[i],
                    d_ffn=d_ffn,
                    dropout=dropout,
                    activation=activation,
                    causal=causal,
                    mamba_config=mamba_config,
                ))

        self.mamba_layers = torch.nn.ModuleList(mamba_list)
        self.cnn_layers = torch.nn.ModuleList(cnn_list)


    def forward(
        self,
        a_x, v_x, 
        a_inference_params = None,
        v_inference_params = None
    ):
        a_out = a_x
        v_out = v_x

        for cnn_layer, mamba_layer in zip(self.cnn_layers, self.mamba_layers):
            a_out, v_out  = cnn_layer(a_out.permute(0,2,1), v_out.permute(0,2,1))
            a_out = a_out.permute(0,2,1)
            v_out = v_out.permute(0,2,1)
            a_out, v_out = mamba_layer(
                a_out, v_out,
                a_inference_params = a_inference_params,
                v_inference_params = v_inference_params
            )
            
        return a_out, v_out

class EnSSM(nn.Module):
    """This class implements the EnSSM encoder.
    """
    def __init__(
        self,
        num_layers,
        input_size,
        output_sizes=[256,512,512],
        d_ffn=1024,
        activation='Swish',
        dropout=0.0,
        causal=False,
        mamba_config=None
    ):
        super().__init__()
        prev_input_size = input_size

        cnn_list = []
        mamba_list = []
        # print(output_sizes)
        for i in range(len(output_sizes)):
            cnn_list.append(CNNEncoderLayer(
                    input_size = input_size if i<1 else output_sizes[i-1],
                    output_size = output_sizes[i],
                    dropout=dropout
                ))
            mamba_list.append(MambaEncoderLayer(
                    d_model=output_sizes[i],
                    d_ffn=d_ffn,
                    dropout=dropout,
                    activation=activation,
                    causal=causal,
                    mamba_config=mamba_config,
                ))

        self.mamba_layers = torch.nn.ModuleList(mamba_list)
        self.cnn_layers = torch.nn.ModuleList(cnn_list)


    def forward(
        self,
        x,
        inference_params = None,
    ):
        out = x

        for cnn_layer, mamba_layer in zip(self.cnn_layers, self.mamba_layers):
            out  = cnn_layer(out.permute(0,2,1))
            out = out.permute(0,2,1)
            out = mamba_layer(
                out,
                inference_params = inference_params,
            )

        return out


def masked_temporal_mean(x, padding_mask=None, eps=1e-8):
    if padding_mask is None:
        return x.mean(dim=1)
    mask = padding_mask.unsqueeze(-1).to(dtype=x.dtype, device=x.device)
    denom = mask.sum(dim=1).clamp_min(eps)
    return (x * mask).sum(dim=1) / denom


def rbf_mmd_loss(x, y, kernel_mul=2.0, kernel_num=5, fixed_sigma=None, eps=1e-8):
    batch_size = x.size(0)
    if batch_size == 0:
        return x.new_tensor(0.0)

    total = torch.cat([x, y], dim=0)
    sq_dist = torch.cdist(total, total, p=2).pow(2)

    if fixed_sigma is None:
        denom = max(total.size(0) * total.size(0) - total.size(0), 1)
        bandwidth = sq_dist.detach().sum() / denom
    else:
        bandwidth = x.new_tensor(float(fixed_sigma))
    bandwidth = bandwidth.clamp_min(eps)
    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))

    kernels = 0.0
    for i in range(kernel_num):
        kernels = kernels + torch.exp(
            -sq_dist / ((bandwidth * (kernel_mul ** i)).clamp_min(eps))
        )

    xx = kernels[:batch_size, :batch_size]
    yy = kernels[batch_size:, batch_size:]
    xy = kernels[:batch_size, batch_size:]
    yx = kernels[batch_size:, :batch_size]
    return xx.mean() + yy.mean() - xy.mean() - yx.mean()


class DepMamba(BaseNet):

    def __init__(self, audio_input_size=161, video_input_size=161, mm_input_size=128, mm_output_sizes=[256,64], d_ffn=1024, num_layers=8, dropout=0.1, activation='Swish', causal=False, mamba_config=None, use_global_alignment=False, pooling_mode="mean", pooling_num_segments=8, pooling_scales=None, pooling_scale_fusion="mean", pooling_dropout=0.1, pooling_residual=True, pooling_residual_alpha=0.1, pooling_motion_emb_dim=32, pooling_motion_cue_mode="full", pooling_reliability_beta=0.2, pooling_gate_emb_dim=16, pooling_gamma_init=0.001, pooling_gamma_max=0.3, gate_cue_mode="real", use_dynamic_temporal_refinement=False, dynamic_refinement_layers=1, dynamic_refinement_kernel_size=5, dynamic_refinement_gate_activation="sigmoid", dynamic_refinement_dyn_type="k", dynamic_refinement_dropout=0.1, dynamic_refinement_residual_scale=0.1, mmd_kernel_mul=2.0, mmd_kernel_num=5, mmd_fixed_sigma=None):
        super().__init__()
        self.use_global_alignment = use_global_alignment
        self.pooling_mode = str(pooling_mode).lower()
        self.pooling_num_segments = pooling_num_segments
        self.pooling_scales = self._normalize_pooling_scales(
            pooling_scales,
            fallback=pooling_num_segments,
        )
        self.pooling_scale_fusion = str(pooling_scale_fusion).lower()
        self.pooling_residual = pooling_residual
        self.pooling_residual_alpha = pooling_residual_alpha
        self.pooling_motion_cue_mode = str(pooling_motion_cue_mode).lower()
        self.pooling_reliability_beta = pooling_reliability_beta
        self.pooling_gamma_max = float(pooling_gamma_max)
        self.gate_cue_mode = str(gate_cue_mode).lower()
        self.use_dynamic_temporal_refinement = bool(use_dynamic_temporal_refinement)
        self.mmd_kernel_mul = mmd_kernel_mul
        self.mmd_kernel_num = mmd_kernel_num
        self.mmd_fixed_sigma = mmd_fixed_sigma
        self.aux_losses = {}
        self.pooling_weights = None
        self.motion_cues = None
        self.gate_cues = None
        self.reliability_gates = None
        self.scale_pooling_weights = None
        self.scale_motion_cues = None
        self.scale_reliability_gates = None
        self.pooling_scales_used = None
        self.current_pooling_gamma = None
        self.last_dtr_aux = None

        if self.pooling_mode not in (
            "mean",
            "motion_segment_attention",
            "motion_reliability_attention",
            "multi_scale_motion",
            "gate_segment_attention",
            "gate_motion_segment_attention",
        ):
            raise ValueError(
                "pooling_mode must be 'mean', 'motion_segment_attention', "
                "'motion_reliability_attention', 'multi_scale_motion', "
                "'gate_segment_attention', or 'gate_motion_segment_attention'"
            )
        if self.gate_cue_mode not in ("real", "batch_shuffle", "segment_shuffle"):
            raise ValueError(
                "gate_cue_mode must be 'real', 'batch_shuffle', or 'segment_shuffle'"
            )
        if self.pooling_scale_fusion != "mean":
            raise ValueError("pooling_scale_fusion currently only supports 'mean'")
        valid_cue_modes = (
            "full",
            "visual",
            "audio",
            "mean_only",
            "visual_mean",
            "audio_mean",
        )
        if self.pooling_motion_cue_mode not in valid_cue_modes:
            raise ValueError(
                "pooling_motion_cue_mode must be one of: "
                + ", ".join(valid_cue_modes)
            )

        self.cossm_encoder = CoSSM(num_layers,
                                         mm_input_size,
                                    mm_output_sizes,
                                    d_ffn,
                                    activation=activation,
                                    dropout=dropout,
                                    causal=causal,
                                    mamba_config=mamba_config)

        self.conv_audio = nn.Conv1d(audio_input_size, mm_input_size, 1, padding=0, dilation=1, bias=False)
        self.conv_video = nn.Conv1d(video_input_size, mm_input_size, 1, padding=0, dilation=1, bias=False)
        
        self.enssm_encoder = EnSSM(num_layers,
                                    mm_output_sizes[-1]*2,
                                    [mm_output_sizes[-1]*2],
                                    d_ffn,
                                    activation=activation,
                                    dropout=dropout,
                                    causal=causal,
                                    mamba_config=mamba_config)
        
        self.pool = nn.AdaptiveMaxPool1d(1)

        self.output = nn.Linear(mm_output_sizes[-1]*2, 1)
        self.m = nn.Sigmoid()
        temporal_dim = mm_output_sizes[-1] * 2
        self.dynamic_temporal_refinement = None
        if self.use_dynamic_temporal_refinement:
            self.dynamic_temporal_refinement = DynamicTemporalRefinement(
                temporal_dim,
                num_layers=dynamic_refinement_layers,
                kernel_size=dynamic_refinement_kernel_size,
                gate_activation=dynamic_refinement_gate_activation,
                dyn_type=dynamic_refinement_dyn_type,
                dropout=dynamic_refinement_dropout,
                residual_scale=dynamic_refinement_residual_scale,
            )
        self.motion_encoder = None
        self.motion_segment_score = None
        self.motion_reliability_gate = None
        self.gate_encoder = None
        self.gate_segment_score = None
        self.pooling_gamma_raw = None
        if self.pooling_mode in (
            "motion_segment_attention",
            "motion_reliability_attention",
            "multi_scale_motion",
            "gate_motion_segment_attention",
        ):
            motion_emb_dim = int(pooling_motion_emb_dim)
            motion_cue_dim = self._motion_cue_dim()
            score_dim = temporal_dim + motion_emb_dim
            self.motion_encoder = nn.Sequential(
                nn.LayerNorm(motion_cue_dim),
                nn.Linear(motion_cue_dim, motion_emb_dim),
                nn.ReLU(),
                nn.Dropout(pooling_dropout),
                nn.Linear(motion_emb_dim, motion_emb_dim),
                nn.ReLU(),
            )
            self.motion_segment_score = nn.Sequential(
                nn.LayerNorm(score_dim),
                nn.Dropout(pooling_dropout),
                nn.Linear(score_dim, 1),
            )
        if self.pooling_mode == "motion_reliability_attention":
            self.motion_reliability_gate = nn.Sequential(
                nn.LayerNorm(score_dim),
                nn.Dropout(pooling_dropout),
                nn.Linear(score_dim, 1),
            )
            nn.init.constant_(self.motion_reliability_gate[-1].bias, 1.0)
        if self.pooling_mode in (
            "gate_segment_attention",
            "gate_motion_segment_attention",
        ):
            gate_emb_dim = int(pooling_gate_emb_dim)
            self.gate_encoder = nn.Sequential(
                nn.LayerNorm(5),
                nn.Linear(5, gate_emb_dim),
                nn.ReLU(),
                nn.Dropout(pooling_dropout),
                nn.Linear(gate_emb_dim, gate_emb_dim),
                nn.ReLU(),
            )
            gate_score_dim = temporal_dim + gate_emb_dim
            if self.pooling_mode == "gate_motion_segment_attention":
                gate_score_dim += int(pooling_motion_emb_dim)
            self.gate_segment_score = nn.Sequential(
                nn.LayerNorm(gate_score_dim),
                nn.Dropout(pooling_dropout),
                nn.Linear(gate_score_dim, 1),
            )
            if self.pooling_residual:
                self.pooling_gamma_raw = bounded_gamma_parameter(
                    pooling_gamma_init,
                    pooling_gamma_max,
                )

        nn.init.xavier_uniform_(self.conv_audio.weight.data)
        nn.init.xavier_uniform_(self.conv_video.weight.data)
        
    def _normalize_pooling_scales(self, pooling_scales, fallback):
        if pooling_scales is None:
            values = [fallback]
        elif isinstance(pooling_scales, str):
            values = [
                item.strip()
                for item in pooling_scales.replace(";", ",").split(",")
                if item.strip()
            ]
        else:
            values = list(pooling_scales)

        scales = []
        for value in values:
            scale = int(value)
            if scale < 1:
                raise ValueError("pooling scales must be positive integers")
            if scale not in scales:
                scales.append(scale)
        if not scales:
            raise ValueError("pooling_scales must contain at least one scale")
        return scales

    def _resolve_num_segments(self, num_segments=None):
        if num_segments is None:
            num_segments = self.pooling_num_segments
        return max(int(num_segments), 1)

    def _clear_pooling_diagnostics(self):
        self.pooling_weights = None
        self.motion_cues = None
        self.gate_cues = None
        self.reliability_gates = None
        self.scale_pooling_weights = None
        self.scale_motion_cues = None
        self.scale_reliability_gates = None
        self.pooling_scales_used = None
        self.current_pooling_gamma = None

    def temporal_stream_extractor(self, x, padding_mask=None, a_inference_params = None, v_inference_params = None):
        xa = x[:, :, 136:]
        xv = x[:, :, :136]
        xa = self.conv_audio(xa.permute(0,2,1)).permute(0,2,1)
        xv = self.conv_video(xv.permute(0,2,1)).permute(0,2,1)

        zero = x.new_tensor(0.0)
        self.aux_losses = {
            "global_align_loss": zero,
        }
        if self.use_global_alignment:
            za = masked_temporal_mean(xa, padding_mask)
            zv = masked_temporal_mean(xv, padding_mask)
            self.aux_losses["global_align_loss"] = rbf_mmd_loss(
                za,
                zv,
                kernel_mul=self.mmd_kernel_mul,
                kernel_num=self.mmd_kernel_num,
                fixed_sigma=self.mmd_fixed_sigma,
            )

        xa, xv = self.cossm_encoder(xa, xv, a_inference_params, v_inference_params)

        x = torch.cat([xa,xv],dim=-1)
        self.last_dtr_aux = None
        if self.dynamic_temporal_refinement is not None:
            x, self.last_dtr_aux = self.dynamic_temporal_refinement(
                x,
                padding_mask,
            )
        x = self.enssm_encoder(x)
        return xa, xv, x

    def temporal_feature_extractor(self, x, padding_mask=None, a_inference_params = None, v_inference_params = None):
        _, _, x = self.temporal_stream_extractor(
            x,
            padding_mask,
            a_inference_params=a_inference_params,
            v_inference_params=v_inference_params,
        )
        return x

    def mean_pool_temporal_features(self, x, padding_mask=None):
        if padding_mask is not None:
            x = x * (padding_mask.unsqueeze(-1).float())
            x = x.sum(dim=1) / (padding_mask.unsqueeze(-1).float()
                                ).sum(dim=1, keepdim=False)  # Compute average
        else:
            x = self.pool(x.permute(0,2,1)).squeeze(-1)
        return x

    def _segment_representations(
        self, x, padding_mask=None, num_segments=None, eps=1e-8
    ):
        batch_size, steps, dim = x.shape
        num_segments = self._resolve_num_segments(num_segments)

        if padding_mask is None:
            padding_mask = torch.ones(
                batch_size, steps, dtype=torch.bool, device=x.device
            )
        else:
            padding_mask = padding_mask.to(dtype=torch.bool, device=x.device)

        valid_lengths = padding_mask.sum(dim=1).clamp_min(1)
        positions = torch.arange(steps, device=x.device).unsqueeze(0)
        segment_reprs = []
        segment_masks = []

        for segment_idx in range(num_segments):
            starts = (segment_idx * valid_lengths) // num_segments
            ends = ((segment_idx + 1) * valid_lengths) // num_segments
            segment_mask = (
                padding_mask
                & (positions >= starts.unsqueeze(1))
                & (positions < ends.unsqueeze(1))
            )
            weights = segment_mask.unsqueeze(-1).to(dtype=x.dtype, device=x.device)
            denom = weights.sum(dim=1).clamp_min(eps)
            segment_reprs.append((x * weights).sum(dim=1) / denom)
            segment_masks.append(segment_mask.any(dim=1))

        segment_reprs = torch.stack(segment_reprs, dim=1)
        segment_masks = torch.stack(segment_masks, dim=1)
        return segment_reprs, segment_masks

    def _motion_cue_dim(self):
        if self.pooling_motion_cue_mode == "full":
            return 4
        if self.pooling_motion_cue_mode in ("visual", "audio", "mean_only"):
            return 2
        return 1

    def _select_motion_cues(self, cues):
        if self.pooling_motion_cue_mode == "full":
            return cues
        if self.pooling_motion_cue_mode == "visual":
            return cues[:, :, [0, 1]]
        if self.pooling_motion_cue_mode == "audio":
            return cues[:, :, [2, 3]]
        if self.pooling_motion_cue_mode == "mean_only":
            return cues[:, :, [0, 2]]
        if self.pooling_motion_cue_mode == "visual_mean":
            return cues[:, :, [0]]
        if self.pooling_motion_cue_mode == "audio_mean":
            return cues[:, :, [2]]
        return cues

    def _segment_motion_cues(
        self, raw_input, padding_mask=None, num_segments=None, eps=1e-8
    ):
        batch_size, steps, _ = raw_input.shape
        num_segments = self._resolve_num_segments(num_segments)

        if padding_mask is None:
            padding_mask = torch.ones(
                batch_size, steps, dtype=torch.bool, device=raw_input.device
            )
        else:
            padding_mask = padding_mask.to(dtype=torch.bool, device=raw_input.device)

        visual = raw_input[:, :, :136]
        audio = raw_input[:, :, 136:]
        visual_delta = torch.zeros(
            batch_size, steps, dtype=raw_input.dtype, device=raw_input.device
        )
        audio_delta = torch.zeros_like(visual_delta)
        if steps > 1:
            transition_mask = padding_mask[:, 1:] & padding_mask[:, :-1]
            visual_delta[:, 1:] = torch.linalg.vector_norm(
                visual[:, 1:] - visual[:, :-1], dim=-1
            ) * transition_mask.to(dtype=raw_input.dtype)
            audio_delta[:, 1:] = torch.linalg.vector_norm(
                audio[:, 1:] - audio[:, :-1], dim=-1
            ) * transition_mask.to(dtype=raw_input.dtype)

        valid_lengths = padding_mask.sum(dim=1).clamp_min(1)
        positions = torch.arange(steps, device=raw_input.device).unsqueeze(0)
        cues = []
        for segment_idx in range(num_segments):
            starts = (segment_idx * valid_lengths) // num_segments
            ends = ((segment_idx + 1) * valid_lengths) // num_segments
            segment_mask = (
                padding_mask
                & (positions >= starts.unsqueeze(1))
                & (positions < ends.unsqueeze(1))
            )
            weights = segment_mask.to(dtype=raw_input.dtype, device=raw_input.device)
            denom = weights.sum(dim=1).clamp_min(eps)
            visual_motion_mean = (visual_delta * weights).sum(dim=1) / denom
            audio_change_mean = (audio_delta * weights).sum(dim=1) / denom
            visual_motion_var = (
                ((visual_delta - visual_motion_mean.unsqueeze(1)) ** 2) * weights
            ).sum(dim=1) / denom
            audio_change_var = (
                ((audio_delta - audio_change_mean.unsqueeze(1)) ** 2) * weights
            ).sum(dim=1) / denom
            visual_motion_std = visual_motion_var.clamp_min(0.0).sqrt()
            audio_change_std = audio_change_var.clamp_min(0.0).sqrt()
            cues.append(torch.stack([
                visual_motion_mean,
                visual_motion_std,
                audio_change_mean,
                audio_change_std,
            ], dim=-1))

        cues = torch.stack(cues, dim=1)
        cue_mean = cues.mean(dim=1, keepdim=True)
        cue_std = cues.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
        cues = (cues - cue_mean) / cue_std
        return self._select_motion_cues(cues)

    def _segment_gate_cues(
        self, gate_k, padding_mask=None, num_segments=None, eps=1e-8
    ):
        if gate_k is None:
            raise ValueError(
                "DTR gate cues are required for gate-aware pooling. "
                "Set use_dynamic_temporal_refinement=True."
            )

        batch_size, kernel_size, steps = gate_k.shape
        num_segments = self._resolve_num_segments(num_segments)
        if padding_mask is None:
            padding_mask = torch.ones(
                batch_size, steps, dtype=torch.bool, device=gate_k.device
            )
        else:
            padding_mask = padding_mask.to(dtype=torch.bool, device=gate_k.device)

        valid_lengths = padding_mask.sum(dim=1).clamp_min(1)
        positions = torch.arange(steps, device=gate_k.device).unsqueeze(0)
        center_idx = kernel_size // 2
        neighbor_indices = [
            idx for idx in range(kernel_size) if idx != center_idx
        ]
        cues = []

        for segment_idx in range(num_segments):
            starts = (segment_idx * valid_lengths) // num_segments
            ends = ((segment_idx + 1) * valid_lengths) // num_segments
            segment_mask = (
                padding_mask
                & (positions >= starts.unsqueeze(1))
                & (positions < ends.unsqueeze(1))
            )
            time_weights = segment_mask.to(dtype=gate_k.dtype)
            denom_time = time_weights.sum(dim=1).clamp_min(eps)
            weights = time_weights.unsqueeze(1)
            denom_all = (denom_time * kernel_size).clamp_min(eps)

            gate_sum = (gate_k * weights).sum(dim=(1, 2))
            gate_mean = gate_sum / denom_all
            gate_var = (
                ((gate_k - gate_mean.view(batch_size, 1, 1)) ** 2) * weights
            ).sum(dim=(1, 2)) / denom_all
            gate_std = gate_var.clamp_min(0.0).sqrt()

            masked_gate = gate_k.masked_fill(~segment_mask.unsqueeze(1), -1e9)
            gate_max = masked_gate.amax(dim=(1, 2))
            has_segment = segment_mask.any(dim=1)
            gate_max = torch.where(
                has_segment,
                gate_max,
                torch.zeros_like(gate_max),
            )

            center_strength = (
                gate_k[:, center_idx, :] * time_weights
            ).sum(dim=1) / denom_time
            if neighbor_indices:
                neighbor_strength = (
                    gate_k[:, neighbor_indices, :] * weights
                ).sum(dim=(1, 2)) / (
                    denom_time * len(neighbor_indices)
                ).clamp_min(eps)
            else:
                neighbor_strength = torch.zeros_like(center_strength)
            strength_total = center_strength + neighbor_strength + eps
            center_ratio = center_strength / strength_total
            neighbor_ratio = neighbor_strength / strength_total

            cues.append(torch.stack([
                gate_mean,
                gate_std,
                gate_max,
                center_ratio,
                neighbor_ratio,
            ], dim=-1))

        gate_cues = torch.stack(cues, dim=1)
        if self.gate_cue_mode == "batch_shuffle" and batch_size > 1:
            gate_cues = gate_cues[torch.randperm(batch_size, device=gate_k.device)]
        elif self.gate_cue_mode == "segment_shuffle" and num_segments > 1:
            gate_cues = gate_cues[
                :,
                torch.randperm(num_segments, device=gate_k.device),
                :,
            ]
        return gate_cues

    def motion_pool_single_scale(
        self,
        x,
        padding_mask=None,
        raw_input=None,
        num_segments=None,
        use_reliability=False,
        eps=1e-8,
    ):
        if raw_input is None:
            raise ValueError("raw_input is required for motion-aware pooling")

        segment_reprs, segment_masks = self._segment_representations(
            x, padding_mask, num_segments=num_segments, eps=eps
        )
        motion_cues = self._segment_motion_cues(
            raw_input, padding_mask, num_segments=num_segments, eps=eps
        )
        motion_emb = self.motion_encoder(motion_cues)
        score_input = torch.cat([segment_reprs, motion_emb], dim=-1)
        scores = self.motion_segment_score(score_input).squeeze(-1)
        gates = None
        if use_reliability:
            gates = torch.sigmoid(
                self.motion_reliability_gate(score_input).squeeze(-1)
            )
            beta = float(self.pooling_reliability_beta)
            scores = scores + beta * torch.log(gates.clamp_min(eps))
        scores = scores.masked_fill(~segment_masks, -1e9)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(weights.unsqueeze(-1) * segment_reprs, dim=1)
        aux = {
            "weights": weights.detach(),
            "motion_cues": motion_cues.detach(),
            "reliability_gates": gates.detach() if gates is not None else None,
        }
        return pooled, aux

    def motion_segment_attention_pool_temporal_features(
        self, x, padding_mask=None, raw_input=None, eps=1e-8
    ):
        pooled, aux = self.motion_pool_single_scale(
            x,
            padding_mask=padding_mask,
            raw_input=raw_input,
            num_segments=self.pooling_num_segments,
            use_reliability=self.pooling_mode == "motion_reliability_attention",
            eps=eps,
        )
        self.pooling_weights = aux["weights"]
        self.motion_cues = aux["motion_cues"]
        self.reliability_gates = aux["reliability_gates"]
        self.scale_pooling_weights = None
        self.scale_motion_cues = None
        self.scale_reliability_gates = None
        self.pooling_scales_used = None
        self.gate_cues = None
        self.current_pooling_gamma = None
        return pooled

    def multi_scale_motion_pool_temporal_features(
        self, x, padding_mask=None, raw_input=None
    ):
        pools = []
        scale_weights = {}
        scale_motion_cues = {}
        for scale in self.pooling_scales:
            pooled, aux = self.motion_pool_single_scale(
                x,
                padding_mask=padding_mask,
                raw_input=raw_input,
                num_segments=scale,
                use_reliability=False,
            )
            pools.append(pooled)
            scale_weights[int(scale)] = aux["weights"]
            scale_motion_cues[int(scale)] = aux["motion_cues"]

        if self.pooling_scale_fusion == "mean":
            pooled = torch.stack(pools, dim=1).mean(dim=1)
        else:
            raise ValueError("pooling_scale_fusion currently only supports 'mean'")

        self.pooling_weights = None
        self.motion_cues = None
        self.reliability_gates = None
        self.gate_cues = None
        self.scale_pooling_weights = scale_weights
        self.scale_motion_cues = scale_motion_cues
        self.scale_reliability_gates = None
        self.pooling_scales_used = list(self.pooling_scales)
        self.current_pooling_gamma = None
        return pooled

    def gate_segment_attention_pool_temporal_features(
        self, x, padding_mask=None, raw_input=None, dtr_aux=None, eps=1e-8
    ):
        if dtr_aux is None:
            dtr_aux = self.last_dtr_aux
        gate_k = None if dtr_aux is None else dtr_aux.get("gate_k")

        segment_reprs, segment_masks = self._segment_representations(
            x,
            padding_mask,
            num_segments=self.pooling_num_segments,
            eps=eps,
        )
        gate_cues = self._segment_gate_cues(
            gate_k,
            padding_mask,
            num_segments=self.pooling_num_segments,
            eps=eps,
        )
        gate_emb = self.gate_encoder(gate_cues)
        score_parts = [segment_reprs]

        motion_cues = None
        if self.pooling_mode == "gate_motion_segment_attention":
            if raw_input is None:
                raise ValueError("raw_input is required for gate-motion pooling")
            motion_cues = self._segment_motion_cues(
                raw_input,
                padding_mask,
                num_segments=self.pooling_num_segments,
                eps=eps,
            )
            score_parts.append(self.motion_encoder(motion_cues))

        score_parts.append(gate_emb)
        scores = self.gate_segment_score(torch.cat(score_parts, dim=-1)).squeeze(-1)
        scores = scores.masked_fill(~segment_masks, -1e9)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(weights.unsqueeze(-1) * segment_reprs, dim=1)

        self.pooling_weights = weights.detach()
        self.motion_cues = motion_cues.detach() if motion_cues is not None else None
        self.gate_cues = gate_cues.detach()
        self.reliability_gates = None
        self.scale_pooling_weights = None
        self.scale_motion_cues = None
        self.scale_reliability_gates = None
        self.pooling_scales_used = None
        return pooled

    def _bounded_pooling_gamma(self):
        gamma = self.pooling_gamma_max * torch.sigmoid(self.pooling_gamma_raw)
        self.current_pooling_gamma = gamma.detach()
        return gamma

    def pool_temporal_features(self, x, padding_mask=None, raw_input=None, dtr_aux=None):
        mean_pooled = self.mean_pool_temporal_features(x, padding_mask)
        if self.pooling_mode == "mean":
            self._clear_pooling_diagnostics()
            return mean_pooled

        if self.pooling_mode in (
            "gate_segment_attention",
            "gate_motion_segment_attention",
        ):
            attention_pooled = self.gate_segment_attention_pool_temporal_features(
                x,
                padding_mask,
                raw_input=raw_input,
                dtr_aux=dtr_aux,
            )
            if self.pooling_residual:
                gamma = self._bounded_pooling_gamma()
                return mean_pooled + gamma * (attention_pooled - mean_pooled)
            return attention_pooled

        if self.pooling_mode == "multi_scale_motion":
            attention_pooled = self.multi_scale_motion_pool_temporal_features(
                x, padding_mask, raw_input=raw_input
            )
        else:
            attention_pooled = self.motion_segment_attention_pool_temporal_features(
                x, padding_mask, raw_input=raw_input
            )
        if self.pooling_residual:
            alpha = float(self.pooling_residual_alpha)
            return (1.0 - alpha) * mean_pooled + alpha * attention_pooled
        return attention_pooled

    def feature_extractor(self, x, padding_mask=None, a_inference_params = None, v_inference_params = None):
        raw_input = x
        x = self.temporal_feature_extractor(
            x,
            padding_mask,
            a_inference_params=a_inference_params,
            v_inference_params=v_inference_params,
        )
        return self.pool_temporal_features(x, padding_mask, raw_input=raw_input)

    def classifier(self, x):
        return self.output(x)

