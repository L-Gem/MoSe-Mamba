import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalOffsetGate(nn.Module):
    def __init__(self, dim, kernel_size=5, activation="sigmoid"):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for temporal offset gates")
        self.kernel_size = int(kernel_size)
        self.activation = str(activation).lower()
        self.gate_conv = nn.Conv1d(
            dim,
            self.kernel_size,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
        )

    def _activate(self, gate):
        if self.activation == "sigmoid":
            return torch.sigmoid(gate)
        if self.activation == "retanh":
            return torch.tanh(gate).clamp_min(0.0)
        if self.activation == "relu":
            return torch.relu(gate)
        raise ValueError(
            "dynamic_refinement_gate_activation must be sigmoid, retanh, or relu"
        )

    def forward(self, x, padding_mask=None):
        # x: [B, T, D], returns gate_k: [B, K, T]
        gate = self.gate_conv(x.transpose(1, 2))
        gate = self._activate(gate)
        if padding_mask is not None:
            gate = gate * padding_mask.unsqueeze(1).to(dtype=gate.dtype)
        return gate


class DynamicTemporalAggregation(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size=5,
        gate_activation="sigmoid",
        dyn_type="k",
        dropout=0.1,
    ):
        super().__init__()
        if str(dyn_type).lower() != "k":
            raise NotImplementedError(
                "Only dyn_type='k' is implemented because it gives interpretable "
                "temporal-offset gates for segment evidence aggregation."
            )
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        self.dim = int(dim)
        self.kernel_size = int(kernel_size)
        self.radius = self.kernel_size // 2
        self.gate = TemporalOffsetGate(
            dim, kernel_size=self.kernel_size, activation=gate_activation
        )
        self.proj = nn.Linear(self.dim * self.kernel_size, self.dim)
        self.norm = nn.LayerNorm(self.dim)
        self.dropout = nn.Dropout(dropout)

    def _shift_neighbors(self, x):
        # x: [B, T, D], output: [B, T, K, D]
        batch_size, steps, dim = x.shape
        x_ch = x.transpose(1, 2)
        padded = F.pad(x_ch, (self.radius, self.radius))
        neighbors = []
        for idx in range(self.kernel_size):
            neighbors.append(padded[:, :, idx:idx + steps].transpose(1, 2))
        return torch.stack(neighbors, dim=2)

    def forward(self, x, padding_mask=None):
        if padding_mask is not None:
            x = x * padding_mask.unsqueeze(-1).to(dtype=x.dtype)

        gate_k = self.gate(x, padding_mask=padding_mask)
        shifted = self._shift_neighbors(x)
        gate_for_data = gate_k.transpose(1, 2).unsqueeze(-1)
        gated = shifted * gate_for_data
        out = self.proj(gated.reshape(x.size(0), x.size(1), -1))
        out = self.dropout(self.norm(out))

        if padding_mask is not None:
            out = out * padding_mask.unsqueeze(-1).to(dtype=out.dtype)

        return out, {"gate_k": gate_k}


class DynamicTemporalRefinementBlock(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size=5,
        gate_activation="sigmoid",
        dyn_type="k",
        dropout=0.1,
        ffn_mult=2,
        residual_scale=0.1,
    ):
        super().__init__()
        self.residual_scale = float(residual_scale)
        hidden_dim = int(dim * ffn_mult)
        self.norm = nn.LayerNorm(dim)
        self.dynamic_agg = DynamicTemporalAggregation(
            dim,
            kernel_size=kernel_size,
            gate_activation=gate_activation,
            dyn_type=dyn_type,
            dropout=dropout,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, padding_mask=None):
        residual, aux = self.dynamic_agg(self.norm(x), padding_mask=padding_mask)
        x = x + self.residual_scale * residual
        if padding_mask is not None:
            x = x * padding_mask.unsqueeze(-1).to(dtype=x.dtype)

        x = x + self.residual_scale * self.ffn(x)
        if padding_mask is not None:
            x = x * padding_mask.unsqueeze(-1).to(dtype=x.dtype)
        return x, aux


class DynamicTemporalRefinement(nn.Module):
    def __init__(
        self,
        dim,
        num_layers=1,
        kernel_size=5,
        gate_activation="sigmoid",
        dyn_type="k",
        dropout=0.1,
        ffn_mult=2,
        residual_scale=0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            DynamicTemporalRefinementBlock(
                dim,
                kernel_size=kernel_size,
                gate_activation=gate_activation,
                dyn_type=dyn_type,
                dropout=dropout,
                ffn_mult=ffn_mult,
                residual_scale=residual_scale,
            )
            for _ in range(int(num_layers))
        ])

    def forward(self, x, padding_mask=None):
        layer_aux = []
        last_aux = None
        for layer in self.layers:
            x, last_aux = layer(x, padding_mask=padding_mask)
            layer_aux.append(last_aux)
        if last_aux is None:
            last_aux = {"gate_k": None}
        return x, {
            "gate_k": last_aux.get("gate_k"),
            "layers": layer_aux,
        }


def bounded_gamma_parameter(gamma_init, gamma_max):
    gamma_max = float(gamma_max)
    gamma_init = float(gamma_init)
    if not (0.0 < gamma_init < gamma_max):
        raise ValueError("pooling_gamma_init must be between 0 and pooling_gamma_max")
    ratio = gamma_init / gamma_max
    raw = math.log(ratio / (1.0 - ratio))
    return nn.Parameter(torch.tensor(raw, dtype=torch.float32))


