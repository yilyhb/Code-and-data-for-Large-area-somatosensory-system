from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import nn


OUTPUT_NAMES = ("ext_map1", "ext_map2", "int_map1", "int_map2")


def sinusoidal_position_2d(side: int, d_model: int) -> torch.Tensor:
    """Return a fixed 2D encoding with shape ``1 x side**2 x d_model``."""

    if side <= 0:
        raise ValueError("side must be positive")
    if d_model % 4:
        raise ValueError("d_model must be divisible by four")
    y, x = torch.meshgrid(
        torch.arange(side, dtype=torch.float32),
        torch.arange(side, dtype=torch.float32),
        indexing="ij",
    )
    x = x.reshape(-1)
    y = y.reshape(-1)
    quarter = d_model // 4
    frequency = torch.exp(
        torch.arange(quarter, dtype=torch.float32)
        * (-math.log(10_000.0) / max(quarter, 1))
    )
    encoding = torch.zeros(1, side * side, d_model, dtype=torch.float32)
    encoding[0, :, :quarter] = torch.sin(x[:, None] * frequency)
    encoding[0, :, quarter : 2 * quarter] = torch.cos(x[:, None] * frequency)
    encoding[0, :, 2 * quarter : 3 * quarter] = torch.sin(y[:, None] * frequency)
    encoding[0, :, 3 * quarter :] = torch.cos(y[:, None] * frequency)
    return encoding


class TactileEncoder(nn.Module):
    """Shallow, layer-specific encoder that keeps one token per tactile pixel."""

    def __init__(self, d_model: int, side: int) -> None:
        super().__init__()
        self.side = int(side)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, d_model, kernel_size=3, padding=1),
        )
        self.register_buffer(
            "position_encoding",
            sinusoidal_position_2d(self.side, d_model),
            persistent=True,
        )

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        if field.ndim != 4 or field.shape[1:] != (3, self.side, self.side):
            raise ValueError(
                f"tactile field must be Bx3x{self.side}x{self.side}, "
                f"got {tuple(field.shape)}"
            )
        feature = self.stem(field)
        tokens = feature.flatten(2).transpose(1, 2)
        return tokens + self.position_encoding.to(dtype=tokens.dtype)


class AngleConditioning(nn.Module):
    """Map independent angle3 values to early FiLM-like token modulation."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.pose_embedding = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.map1_modulation = nn.Linear(d_model, 2 * d_model)
        self.map2_modulation = nn.Linear(d_model, 2 * d_model)
        nn.init.normal_(self.map1_modulation.weight, std=0.01)
        nn.init.normal_(self.map2_modulation.weight, std=0.01)
        nn.init.zeros_(self.map1_modulation.bias)
        nn.init.zeros_(self.map2_modulation.bias)

    @staticmethod
    def _modulate(tokens: torch.Tensor, modulation: torch.Tensor) -> torch.Tensor:
        gamma, beta = modulation.chunk(2, dim=-1)
        gamma = 0.1 * torch.tanh(gamma).unsqueeze(1)
        beta = beta.unsqueeze(1)
        return tokens * (1.0 + gamma) + beta

    def forward(
        self,
        map1_tokens: torch.Tensor,
        map2_tokens: torch.Tensor,
        angle3: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if angle3.ndim != 2 or angle3.shape[1] != 3:
            raise ValueError(f"angle3 must be Bx3, got {tuple(angle3.shape)}")
        if angle3.shape[0] != map1_tokens.shape[0]:
            raise ValueError("angle3 batch does not match tactile fields")
        pose = self.pose_embedding(angle3)
        return (
            self._modulate(map1_tokens, self.map1_modulation(pose)),
            self._modulate(map2_tokens, self.map2_modulation(pose)),
        )


class CrossStreamUpdate(nn.Module):
    """One direction of cross-attention, learned gating, and FFN update."""

    def __init__(self, d_model: int, nhead: int, hidden_dim: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.context_norm = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=0.0,
            batch_first=True,
        )
        self.gate_norm = nn.LayerNorm(2 * d_model)
        self.gate_projection = nn.Linear(2 * d_model, d_model)
        nn.init.normal_(self.gate_projection.weight, std=0.01)
        nn.init.constant_(self.gate_projection.bias, -1.0)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, local: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        cross, _ = self.cross_attention(
            self.query_norm(local),
            self.context_norm(other),
            self.context_norm(other),
            need_weights=False,
        )
        gate_input = self.gate_norm(torch.cat((local, cross), dim=-1))
        gate = torch.sigmoid(self.gate_projection(gate_input))
        updated = local + gate * cross
        return updated + self.ffn(self.ffn_norm(updated))


class SourceSpecificDecoder(nn.Module):
    """One semantic source decoder producing its Map1 and Map2 fields."""

    def __init__(self, d_model: int, side: int) -> None:
        super().__init__()
        self.side = int(side)
        self.token_fusion = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
        )
        self.token_refinement = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.spatial_refinement = nn.Sequential(
            nn.GroupNorm(8, d_model),
            nn.GELU(),
            nn.Conv2d(d_model, 96, kernel_size=3, padding=1),
            nn.GroupNorm(8, 96),
            nn.GELU(),
            nn.Conv2d(96, d_model, kernel_size=1),
        )
        self.map1_head = nn.Sequential(
            nn.Conv2d(2 * d_model, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
        )
        self.map2_head = nn.Sequential(
            nn.Conv2d(2 * d_model, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
        )

    def _to_map(self, tokens: torch.Tensor) -> torch.Tensor:
        batch = tokens.shape[0]
        return tokens.transpose(1, 2).reshape(
            batch, tokens.shape[-1], self.side, self.side
        )

    def forward(
        self,
        map1_tokens: torch.Tensor,
        map2_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fused = self.token_fusion(torch.cat((map1_tokens, map2_tokens), dim=-1))
        fused = fused + self.token_refinement(fused)
        fused_map = self._to_map(fused)
        fused_map = fused_map + self.spatial_refinement(fused_map)
        map1_local = self._to_map(map1_tokens)
        map2_local = self._to_map(map2_tokens)
        return (
            self.map1_head(torch.cat((fused_map, map1_local), dim=1)),
            self.map2_head(torch.cat((fused_map, map2_local), dim=1)),
        )


@dataclass(frozen=True)
class Architecture:
    name: str = "MPDTransformerOracle"
    input_contract: str = "Map1[3x32x32] + Map2[3x32x32] + angle3[3]"
    outputs: tuple[str, ...] = OUTPUT_NAMES
    side: int = 32
    d_model: int = 128
    nhead: int = 4
    hidden_dim: int = 512
    token_count_per_map: int = 1024
    angle_source: str = "independently estimated from raw6; raw6 is not a model input"
    pose_head: bool = False
    coupling_output: bool = False


class MPDTransformerOracle(nn.Module):
    """Final angle3-conditioned bilayer MPD-Transformer.

    The model accepts only two normalized tactile maps and normalized angle3.
    It performs a single bidirectional Map1/Map2 interaction, then splits the
    fused bilayer representation into independent external and internal source
    decoders.  There is no raw6 argument, pose head, or fifth residual output.
    """

    def __init__(
        self,
        *,
        side: int = 32,
        d_model: int = 128,
        nhead: int = 4,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        if side != 32:
            raise ValueError("The finalized Oracle contract requires side=32")
        if d_model != 128 or nhead != 4 or hidden_dim != 512:
            raise ValueError(
                "The finalized contract requires d_model=128, nhead=4, hidden_dim=512"
            )
        self.architecture = Architecture(
            side=side,
            d_model=d_model,
            nhead=nhead,
            hidden_dim=hidden_dim,
            token_count_per_map=side * side,
        )
        self.map1_encoder = TactileEncoder(d_model, side)
        self.map2_encoder = TactileEncoder(d_model, side)
        self.angle_conditioning = AngleConditioning(d_model)
        self.map1_queries_map2 = CrossStreamUpdate(d_model, nhead, hidden_dim)
        self.map2_queries_map1 = CrossStreamUpdate(d_model, nhead, hidden_dim)
        self.external_decoder = SourceSpecificDecoder(d_model, side)
        self.internal_decoder = SourceSpecificDecoder(d_model, side)

    def forward(
        self,
        map1: torch.Tensor,
        map2: torch.Tensor,
        angle3: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        map1_tokens = self.map1_encoder(map1)
        map2_tokens = self.map2_encoder(map2)
        map1_tokens, map2_tokens = self.angle_conditioning(
            map1_tokens, map2_tokens, angle3
        )
        # Both directions use the same pre-update conditioned state.
        updated_map1 = self.map1_queries_map2(map1_tokens, map2_tokens)
        updated_map2 = self.map2_queries_map1(map2_tokens, map1_tokens)
        ext_map1, ext_map2 = self.external_decoder(updated_map1, updated_map2)
        int_map1, int_map2 = self.internal_decoder(updated_map1, updated_map2)
        return {
            "ext_map1": ext_map1,
            "ext_map2": ext_map2,
            "int_map1": int_map1,
            "int_map2": int_map2,
        }

    def descriptor(self) -> dict[str, Any]:
        return asdict(self.architecture)


def parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }
