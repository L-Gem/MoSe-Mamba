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
    def __init__(
        self,
        audio_input_size=161,
        video_input_size=161,
        mm_input_size=128,
        mm_output_sizes=[256, 64],
        d_ffn=1024,
        num_layers=8,
        dropout=0.1,
        activation='Swish',
        causal=False,
        mamba_config=None,
        motion_num_segments=8,
        motion_alpha=0.1,
        motion_hidden_size=64,
        **kwargs,
    ):
        super().__init__()

        self.cossm_encoder = CoSSM(
            num_layers,
            mm_input_size,
            mm_output_sizes,
            d_ffn,
            activation=activation,
            dropout=dropout,
            causal=causal,
            mamba_config=mamba_config,
        )

        self.conv_audio = nn.Conv1d(audio_input_size, mm_input_size, 1, padding=0, dilation=1, bias=False)
        self.conv_video = nn.Conv1d(video_input_size, mm_input_size, 1, padding=0, dilation=1, bias=False)

        self.enssm_encoder = EnSSM(
            num_layers,
            mm_output_sizes[-1] * 2,
            [mm_output_sizes[-1] * 2],
            d_ffn,
            activation=activation,
            dropout=dropout,
            causal=causal,
            mamba_config=mamba_config,
        )

        encoder_dim = mm_output_sizes[-1] * 2

        self.motion_num_segments = kwargs.get("pooling_num_segments", motion_num_segments)
        self.motion_alpha = kwargs.get("pooling_residual_alpha", motion_alpha)

        # This argument is now used as the real ablation control.
        self.ablation_mode = str(
            kwargs.get("pooling_motion_cue_mode", "full")
        ).lower()

        allowed_ablation_modes = {
            "full",
            "baseline",
            "no_motion",
            "motion_only",
            "visual_only",
            "audio_only",
            "no_residual",
        }
        if self.ablation_mode not in allowed_ablation_modes:
            raise ValueError(
                f"Unsupported ablation mode: {self.ablation_mode}. "
                f"Expected one of {sorted(allowed_ablation_modes)}"
            )

        self.motion_mlp = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, motion_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(motion_hidden_size, motion_hidden_size),
            nn.GELU(),
        )

        self.segment_scorer = nn.Sequential(
            nn.Linear(encoder_dim + motion_hidden_size, motion_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(motion_hidden_size, 1),
        )

        self.pool = nn.AdaptiveMaxPool1d(1)
        self.output = nn.Linear(encoder_dim, 1)
        self.m = nn.Sigmoid()

        nn.init.xavier_uniform_(self.conv_audio.weight.data)
        nn.init.xavier_uniform_(self.conv_video.weight.data)

        print(
            f"[Motion Pooling] K={self.motion_num_segments}, "
            f"alpha={self.motion_alpha}, mode={self.ablation_mode}"
        )

    def _masked_mean_pool(self, z, padding_mask=None):
        if padding_mask is None:
            return z.mean(dim=1)

        mask = padding_mask.to(device=z.device).float()
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (z * mask.unsqueeze(-1)).sum(dim=1) / denom

    def _segment_mean_and_std(self, values, segment_mask):
        mask = segment_mask.float()
        denom = mask.sum(dim=1).clamp_min(1.0)

        mean = (values * mask).sum(dim=1) / denom
        var = (((values - mean.unsqueeze(1)) ** 2) * mask).sum(dim=1) / denom
        std = torch.sqrt(var + 1e-6)

        return mean, std

    def motion_aware_segment_pooling(
        self, z, raw_x, padding_mask=None, return_analysis=False
    ):
        batch_size, seq_len, hidden_dim = z.shape
        device = z.device
        dtype = z.dtype
        num_segments = self.motion_num_segments

        if padding_mask is None:
            valid_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.bool)
        else:
            valid_mask = padding_mask.to(device=device).bool()

        valid_lengths = valid_mask.long().sum(dim=1).clamp_min(1)
        time_index = torch.arange(seq_len, device=device).unsqueeze(0)

        visual_x = raw_x[:, :, :136]
        audio_x = raw_x[:, :, 136:]

        visual_motion = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)
        audio_change = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)

        if seq_len > 1:
            visual_motion[:, 1:] = (visual_x[:, 1:] - visual_x[:, :-1]).abs().mean(dim=-1)
            audio_change[:, 1:] = (audio_x[:, 1:] - audio_x[:, :-1]).abs().mean(dim=-1)

        segment_reprs = []
        motion_cues = []
        segment_valids = []

        for seg_idx in range(num_segments):
            start = torch.div(valid_lengths * seg_idx, num_segments, rounding_mode='floor')
            end = torch.div(valid_lengths * (seg_idx + 1), num_segments, rounding_mode='floor')

            segment_mask = (
                (time_index >= start.unsqueeze(1))
                & (time_index < end.unsqueeze(1))
                & valid_mask
            )

            segment_valid = segment_mask.any(dim=1)
            segment_valids.append(segment_valid)

            segment_mask_float = segment_mask.float()
            denom = segment_mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)

            segment_repr = (z * segment_mask_float.unsqueeze(-1)).sum(dim=1) / denom
            segment_reprs.append(segment_repr)

            visual_motion_mean, visual_motion_std = self._segment_mean_and_std(
                visual_motion, segment_mask
            )
            audio_change_mean, audio_change_std = self._segment_mean_and_std(
                audio_change, segment_mask
            )

            cue = torch.stack(
                [
                    visual_motion_mean,
                    visual_motion_std,
                    audio_change_mean,
                    audio_change_std,
                ],
                dim=-1,
            )
            motion_cues.append(cue)

        segment_reprs = torch.stack(segment_reprs, dim=1)
        motion_cues = torch.stack(motion_cues, dim=1)
        segment_valids = torch.stack(segment_valids, dim=1)

        # Select the dynamic cues used by each ablation variant.
        cues_for_scoring = motion_cues

        if self.ablation_mode == "visual_only":
            cues_for_scoring = motion_cues.clone()
            cues_for_scoring[:, :, 2:4] = 0.0
        elif self.ablation_mode == "audio_only":
            cues_for_scoring = motion_cues.clone()
            cues_for_scoring[:, :, 0:2] = 0.0

        motion_embedding = self.motion_mlp(cues_for_scoring)

        semantic_for_scoring = segment_reprs
        motion_for_scoring = motion_embedding

        # no_motion: segment weights depend only on high-level semantics.
        if self.ablation_mode == "no_motion":
            motion_for_scoring = torch.zeros_like(motion_embedding)

        # motion_only: segment weights depend only on raw dynamic cues.
        elif self.ablation_mode == "motion_only":
            semantic_for_scoring = torch.zeros_like(segment_reprs)

        score_input = torch.cat(
            [semantic_for_scoring, motion_for_scoring],
            dim=-1,
        )

        segment_scores = self.segment_scorer(score_input).squeeze(-1)
        segment_scores = segment_scores.masked_fill(
            ~segment_valids,
            torch.finfo(segment_scores.dtype).min,
        )

        has_valid_segment = segment_valids.any(dim=1, keepdim=True)
        segment_scores = torch.where(
            has_valid_segment,
            segment_scores,
            torch.zeros_like(segment_scores),
        )

        segment_weights = torch.softmax(segment_scores, dim=1)
        segment_weights = segment_weights * segment_valids.float()
        segment_weights = segment_weights / segment_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

        motion_pool = (segment_reprs * segment_weights.unsqueeze(-1)).sum(dim=1)
        if return_analysis:
            analysis = {
                "segment_weights": segment_weights,
                "motion_cues": {
                    "visual_motion_mean": motion_cues[:, :, 0],
                    "visual_motion_std": motion_cues[:, :, 1],
                    "audio_change_mean": motion_cues[:, :, 2],
                    "audio_change_std": motion_cues[:, :, 3],
                },
                "motion_cues_tensor": motion_cues,
                "valid_segment_mask": segment_valids,
                "segment_representations": segment_reprs,
                "segment_scores": segment_scores,
            }
            return motion_pool, analysis
        return motion_pool

    def feature_extractor(
        self,
        x,
        padding_mask=None,
        a_inference_params=None,
        v_inference_params=None,
        return_analysis=False,
    ):
        raw_x = x

        xa = x[:, :, 136:]
        xv = x[:, :, :136]

        xa = self.conv_audio(xa.permute(0, 2, 1)).permute(0, 2, 1)
        xv = self.conv_video(xv.permute(0, 2, 1)).permute(0, 2, 1)

        xa, xv = self.cossm_encoder(
            xa,
            xv,
            a_inference_params,
            v_inference_params,
        )

        z = torch.cat([xa, xv], dim=-1)
        z = self.enssm_encoder(z)

        mean_pool = self._masked_mean_pool(z, padding_mask)
        if return_analysis:
            motion_pool, analysis = self.motion_aware_segment_pooling(
                z, raw_x, padding_mask, return_analysis=True
            )
        else:
            motion_pool = self.motion_aware_segment_pooling(z, raw_x, padding_mask)

        if self.ablation_mode == "baseline":
            # Pure global masked mean pooling.
            final_pool = mean_pool
        elif self.ablation_mode == "no_residual":
            # Use only the weighted segment representation.
            final_pool = motion_pool
        else:
            # Full residual aggregation and cue/semantic ablations.
            final_pool = (
                (1.0 - self.motion_alpha) * mean_pool
                + self.motion_alpha * motion_pool
            )
        if return_analysis:
            analysis.update(
                {
                    "z": z,
                    "h_mean": mean_pool,
                    "h_motion": motion_pool,
                    "h_final": final_pool,
                    "motion_alpha": self.motion_alpha,
                }
            )
            return final_pool, analysis
        return final_pool

    def classifier(self, x):
        return self.output(x)

    def forward(self, x, mask=None, return_analysis=False, labels=None):
        if not return_analysis:
            return super().forward(x, mask)

        features, analysis = self.feature_extractor(
            x, mask, return_analysis=True
        )
        logits = self.classifier(features)
        analysis["logits"] = logits
        analysis["predicted_probabilities"] = torch.sigmoid(logits)
        if labels is not None:
            analysis["true_labels"] = labels
        return logits, analysis
