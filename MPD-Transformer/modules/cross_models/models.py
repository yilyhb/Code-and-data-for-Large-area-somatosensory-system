from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


OUTPUT_NAMES = ("ext_map1", "ext_map2", "int_map1", "int_map2")


class SinusoidalPositionalEncoding2D(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        if d_model % 4:
            raise ValueError("d_model must be divisible by four")
        self.d_model = d_model

    def forward(
        self, height: int, width: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        y, x = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing="ij",
        )
        x = x.reshape(-1)
        y = y.reshape(-1)
        quarter = self.d_model // 4
        div_term = torch.exp(
            torch.arange(quarter, device=device, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0, device=device)) / quarter)
        )
        encoding = torch.zeros(1, height * width, self.d_model, device=device)
        encoding[0, :, :quarter] = torch.sin(x[:, None] * div_term)
        encoding[0, :, quarter : 2 * quarter] = torch.cos(x[:, None] * div_term)
        encoding[0, :, 2 * quarter : 3 * quarter] = torch.sin(y[:, None] * div_term)
        encoding[0, :, 3 * quarter :] = torch.cos(y[:, None] * div_term)
        return encoding.to(dtype=dtype)


class TactileEncoder(nn.Module):
    def __init__(self, in_channels: int, d_model: int, token_grid: int, use_positional_encoding: bool = True) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, d_model, kernel_size=3, padding=1),
        )
        self.token_grid = token_grid
        self.positional_encoding = SinusoidalPositionalEncoding2D(d_model)
        self.use_positional_encoding = bool(use_positional_encoding)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        output_hw = (int(x.shape[-2]), int(x.shape[-1]))
        features = self.backbone(x)
        if features.shape[-2:] != (self.token_grid, self.token_grid):
            raise ValueError(f"Expected native {self.token_grid}x{self.token_grid} grid, got {features.shape[-2:]}")
        height, width = features.shape[-2:]
        tokens = features.flatten(2).transpose(1, 2)
        if self.use_positional_encoding:
            tokens = tokens + self.positional_encoding(height, width, x.device, x.dtype)
        return tokens, tokens.mean(dim=1), output_hw


class GatedCrossAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, learnable_gate: bool = True) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        if learnable_gate:
            self.gate = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer("gate", torch.ones(1), persistent=True)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        output, _ = self.attention(query, key_value, key_value, need_weights=False)
        return output * self.gate


class DecouplingDecoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, mlp_ratio: int, learnable_gate: bool = True) -> None:
        super().__init__()
        self.cross1 = GatedCrossAttention(d_model, nhead, learnable_gate)
        self.cross2 = GatedCrossAttention(d_model, nhead, learnable_gate)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model),
        )

    def forward(self, query: torch.Tensor, source1: torch.Tensor, source2: torch.Tensor) -> torch.Tensor:
        value = query + self.cross1(self.norm1(query), source1)
        value = value + self.cross2(self.norm2(value), source2)
        return value + self.mlp(value)


class SimpleFusionDecoder(nn.Module):
    """Token-aligned concat fusion with attention-like parameter capacity."""

    def __init__(self, d_model: int, mlp_ratio: int) -> None:
        super().__init__()
        self.norm_query = nn.LayerNorm(d_model)
        self.norm_source1 = nn.LayerNorm(d_model)
        self.norm_source2 = nn.LayerNorm(d_model)
        self.fusion = nn.Sequential(
            nn.Linear(3 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model),
        )

    def forward(self, query: torch.Tensor, source1: torch.Tensor, source2: torch.Tensor) -> torch.Tensor:
        if query.shape[1] != source1.shape[1] or query.shape[1] != source2.shape[1]:
            raise ValueError("Simple fusion requires aligned native-grid tokens")
        fused = self.fusion(
            torch.cat(
                [self.norm_query(query), self.norm_source1(source1), self.norm_source2(source2)],
                dim=-1,
            )
        )
        value = query + fused
        return value + self.mlp(value)


class MPDProposed(nn.Module):
    VALID_MODES = {"full", "no_cross", "simple_fusion", "no_gating", "shared_decoder", "no_positional_encoding"}

    def __init__(
        self,
        mode: str = "full",
        d_model: int = 128,
        nhead: int = 4,
        mlp_ratio: int = 4,
        token_grid: int = 32,
        cls_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(mode)
        self.mode = mode
        self.d_model = d_model
        self.token_grid = token_grid
        use_positional_encoding = mode != "no_positional_encoding"

        # These names intentionally match the frozen E7 model so Full can be
        # checked and loaded bit-for-bit from the E7-C checkpoints.
        self.map1_encoder = TactileEncoder(3, d_model, token_grid, use_positional_encoding)
        self.map2_encoder = TactileEncoder(3, d_model, token_grid, use_positional_encoding)
        self.imu_projection = nn.Linear(6, d_model)
        self.raw6_pose_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 3)
        )
        self.tactile_pose_head = nn.Sequential(
            nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 3)
        )
        self.condition_map1 = nn.Sequential(nn.Linear(3, d_model), nn.GELU())
        self.condition_map2 = nn.Sequential(nn.Linear(3, d_model), nn.GELU())
        self.cls_dropout = nn.Dropout(cls_dropout)

        if mode == "simple_fusion":
            self.decoders = nn.ModuleDict({name: SimpleFusionDecoder(d_model, mlp_ratio) for name in OUTPUT_NAMES})
        elif mode == "shared_decoder":
            self.shared_decoder = DecouplingDecoder(d_model, nhead, mlp_ratio, True)
            self.task_embeddings = nn.Parameter(torch.zeros(len(OUTPUT_NAMES), 1, d_model))
            nn.init.normal_(self.task_embeddings, std=0.02)
            self.output_heads = nn.ModuleDict({name: nn.Linear(d_model, 3) for name in OUTPUT_NAMES})
        else:
            learnable_gate = mode != "no_gating"
            self.decoders = nn.ModuleDict(
                {name: DecouplingDecoder(d_model, nhead, mlp_ratio, learnable_gate) for name in OUTPUT_NAMES}
            )
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 3)
        )

    def _tokens_to_map(self, tokens: torch.Tensor, output_hw: tuple[int, int], name: str) -> torch.Tensor:
        batch = tokens.shape[0]
        if self.mode == "shared_decoder":
            field = self.output_heads[name](tokens)
        else:
            field = self.output_head(tokens)
        field = field.transpose(1, 2).reshape(batch, 3, self.token_grid, self.token_grid)
        if field.shape[-2:] != output_hw:
            raise ValueError("Decoder output does not match the native sensor grid")
        return field

    @staticmethod
    def _routing(
        name: str, map1_tokens: torch.Tensor, map2_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if name == "ext_map1":
            return map1_tokens, map2_tokens
        if name == "int_map1":
            return map2_tokens, map1_tokens
        if name == "ext_map2":
            return map1_tokens, map2_tokens
        if name == "int_map2":
            return map2_tokens, map1_tokens
        raise KeyError(name)

    def forward(
        self, map1: torch.Tensor, map2: torch.Tensor, imu: torch.Tensor, angle3: torch.Tensor
    ) -> dict[str, torch.Tensor | None]:
        del imu  # E7 selected angle3; raw6 is deliberately inactive in E8.
        map1_tokens, _, output_hw = self.map1_encoder(map1)
        map2_tokens, _, _ = self.map2_encoder(map2)
        query_map1 = map1_tokens + self.condition_map1(angle3).unsqueeze(1)
        query_map2 = map2_tokens + self.condition_map2(angle3).unsqueeze(1)
        queries = {
            "ext_map1": query_map1,
            "int_map1": query_map1,
            "ext_map2": query_map2,
            "int_map2": query_map2,
        }

        result: dict[str, torch.Tensor | None] = {"pose": None}
        for task_index, name in enumerate(OUTPUT_NAMES):
            source1, source2 = self._routing(name, map1_tokens, map2_tokens)
            if self.mode == "no_cross":
                local = map1_tokens if name.endswith("map1") else map2_tokens
                source1 = local
                source2 = local
            query = queries[name]
            if self.mode == "shared_decoder":
                query = query + self.task_embeddings[task_index].unsqueeze(0)
                decoded = self.shared_decoder(query, source1, source2)
            else:
                decoded = self.decoders[name](query, source1, source2)
            result[name] = self._tokens_to_map(decoded, output_hw, name)
        return result

    def active_parameter_count(self) -> int:
        unused = ["imu_projection.", "raw6_pose_head.", "tactile_pose_head."]
        if self.mode == "shared_decoder":
            unused.append("output_head.")
        return sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if not any(name.startswith(prefix) for prefix in unused)
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout else nn.Identity(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.block(value)


class AngleFiLM(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, channels), nn.GELU(), nn.Linear(channels, 2 * channels))

    def forward(self, feature: torch.Tensor, angle3: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.net(angle3).chunk(2, dim=-1)
        return feature * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]


def _split_twelve(value: torch.Tensor) -> dict[str, torch.Tensor | None]:
    ext_map1, ext_map2, int_map1, int_map2 = value.chunk(4, dim=1)
    return {
        "pose": None,
        "ext_map1": ext_map1,
        "ext_map2": ext_map2,
        "int_map1": int_map1,
        "int_map2": int_map2,
    }


class StrongResCNN(nn.Module):
    def __init__(self, base: int = 48, bottleneck: int = 160, blocks: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        mid = max(base * 2, bottleneck // 2)
        self.stem = nn.Conv2d(6, base, 3, padding=1)
        self.enc0 = nn.Sequential(*[ResidualBlock(base, dropout) for _ in range(blocks)])
        self.down1 = nn.Conv2d(base, mid, 4, stride=2, padding=1)
        self.enc1 = nn.Sequential(*[ResidualBlock(mid, dropout) for _ in range(blocks)])
        self.down2 = nn.Conv2d(mid, bottleneck, 4, stride=2, padding=1)
        self.bottleneck = nn.Sequential(*[ResidualBlock(bottleneck, dropout) for _ in range(blocks)])
        self.film = AngleFiLM(bottleneck)
        self.up1 = nn.Conv2d(bottleneck, mid, 3, padding=1)
        self.dec1 = ResidualBlock(mid, dropout)
        self.up0 = nn.Conv2d(mid, base, 3, padding=1)
        self.dec0 = ResidualBlock(base, dropout)
        self.head = nn.Conv2d(base, 12, 1)

    def forward(self, map1: torch.Tensor, map2: torch.Tensor, imu: torch.Tensor, angle3: torch.Tensor) -> dict[str, torch.Tensor | None]:
        del imu
        skip0 = self.enc0(self.stem(torch.cat([map1, map2], dim=1)))
        skip1 = self.enc1(self.down1(skip0))
        value = self.film(self.bottleneck(self.down2(skip1)), angle3)
        value = F.interpolate(value, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
        value = self.dec1(self.up1(value))
        value = F.interpolate(value, size=skip0.shape[-2:], mode="bilinear", align_corners=False)
        value = self.dec0(self.up0(value))
        return _split_twelve(self.head(value))


class ResUNet(nn.Module):
    def __init__(self, base: int = 32, depth: int = 3, blocks: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        if depth not in (2, 3):
            raise ValueError("ResUNet depth must be 2 or 3 for a 32x32 grid")
        channels = [base * (2**level) for level in range(depth + 1)]
        self.stem = nn.Conv2d(6, channels[0], 3, padding=1)
        self.encoder_blocks = nn.ModuleList()
        self.down = nn.ModuleList()
        for level in range(depth):
            self.encoder_blocks.append(nn.Sequential(*[ResidualBlock(channels[level], dropout) for _ in range(blocks)]))
            self.down.append(nn.Conv2d(channels[level], channels[level + 1], 4, stride=2, padding=1))
        self.center = nn.Sequential(*[ResidualBlock(channels[-1], dropout) for _ in range(blocks + 1)])
        self.film = AngleFiLM(channels[-1])
        self.up = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for level in reversed(range(depth)):
            self.up.append(nn.ConvTranspose2d(channels[level + 1], channels[level], 2, stride=2))
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.Conv2d(2 * channels[level], channels[level], 3, padding=1),
                    *[ResidualBlock(channels[level], dropout) for _ in range(blocks)],
                )
            )
        self.head = nn.Conv2d(channels[0], 12, 1)

    def forward(self, map1: torch.Tensor, map2: torch.Tensor, imu: torch.Tensor, angle3: torch.Tensor) -> dict[str, torch.Tensor | None]:
        del imu
        value = self.stem(torch.cat([map1, map2], dim=1))
        skips: list[torch.Tensor] = []
        for block, down in zip(self.encoder_blocks, self.down):
            value = block(value)
            skips.append(value)
            value = down(value)
        value = self.film(self.center(value), angle3)
        for up, block, skip in zip(self.up, self.decoder_blocks, reversed(skips)):
            value = up(value)
            value = block(torch.cat([value, skip], dim=1))
        return _split_twelve(self.head(value))


class MixerBlock(nn.Module):
    def __init__(self, tokens: int, channels: int, token_mlp: int, channel_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden = int(round(channels * channel_ratio))
        self.norm1 = nn.LayerNorm(channels)
        self.token_mixing = nn.Sequential(
            nn.Linear(tokens, token_mlp), nn.GELU(), nn.Dropout(dropout), nn.Linear(token_mlp, tokens)
        )
        self.norm2 = nn.LayerNorm(channels)
        self.channel_mixing = nn.Sequential(
            nn.Linear(channels, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, channels)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value + self.token_mixing(self.norm1(value).transpose(1, 2)).transpose(1, 2)
        return value + self.channel_mixing(self.norm2(value))


class MLPMixer(nn.Module):
    def __init__(
        self,
        patch_size: int = 4,
        d_model: int = 160,
        depth: int = 6,
        token_mlp: int = 64,
        channel_ratio: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if 32 % patch_size:
            raise ValueError("patch_size must divide 32")
        self.patch_size = patch_size
        grid = 32 // patch_size
        tokens = grid * grid
        self.patch_embed = nn.Conv2d(6, d_model, patch_size, stride=patch_size)
        self.angle_projection = nn.Sequential(nn.Linear(3, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.blocks = nn.Sequential(
            *[MixerBlock(tokens, d_model, token_mlp, channel_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 12 * patch_size * patch_size)

    def forward(self, map1: torch.Tensor, map2: torch.Tensor, imu: torch.Tensor, angle3: torch.Tensor) -> dict[str, torch.Tensor | None]:
        del imu
        value = self.patch_embed(torch.cat([map1, map2], dim=1)).flatten(2).transpose(1, 2)
        value = value + self.angle_projection(angle3).unsqueeze(1)
        patches = self.head(self.norm(self.blocks(value))).transpose(1, 2)
        output = F.fold(patches, output_size=(32, 32), kernel_size=self.patch_size, stride=self.patch_size)
        return _split_twelve(output)


class PlainTransformer(nn.Module):
    def __init__(
        self,
        patch_size: int = 4,
        d_model: int = 128,
        nhead: int = 4,
        depth: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if 32 % patch_size or d_model % 4:
            raise ValueError("Invalid patch_size or d_model")
        self.patch_size = patch_size
        self.grid = 32 // patch_size
        self.map1_embed = nn.Conv2d(3, d_model, patch_size, stride=patch_size)
        self.map2_embed = nn.Conv2d(3, d_model, patch_size, stride=patch_size)
        self.positional_encoding = SinusoidalPositionalEncoding2D(d_model)
        self.modality_embedding = nn.Parameter(torch.zeros(2, 1, d_model))
        nn.init.normal_(self.modality_embedding, std=0.02)
        self.angle_projection = nn.Sequential(nn.Linear(3, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.map1_head = nn.Linear(d_model, 6 * patch_size * patch_size)
        self.map2_head = nn.Linear(d_model, 6 * patch_size * patch_size)

    def _fold(self, value: torch.Tensor) -> torch.Tensor:
        return F.fold(
            value.transpose(1, 2),
            output_size=(32, 32),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, map1: torch.Tensor, map2: torch.Tensor, imu: torch.Tensor, angle3: torch.Tensor) -> dict[str, torch.Tensor | None]:
        del imu
        tokens1 = self.map1_embed(map1).flatten(2).transpose(1, 2)
        tokens2 = self.map2_embed(map2).flatten(2).transpose(1, 2)
        position = self.positional_encoding(self.grid, self.grid, map1.device, map1.dtype)
        condition = self.angle_projection(angle3).unsqueeze(1)
        tokens1 = tokens1 + position + self.modality_embedding[0].unsqueeze(0) + condition
        tokens2 = tokens2 + position + self.modality_embedding[1].unsqueeze(0) + condition
        encoded = self.norm(self.encoder(torch.cat([tokens1, tokens2], dim=1)))
        count = tokens1.shape[1]
        fields1 = self._fold(self.map1_head(encoded[:, :count]))
        fields2 = self._fold(self.map2_head(encoded[:, count:]))
        ext_map1, int_map1 = fields1.chunk(2, dim=1)
        ext_map2, int_map2 = fields2.chunk(2, dim=1)
        return {
            "pose": None,
            "ext_map1": ext_map1,
            "ext_map2": ext_map2,
            "int_map1": int_map1,
            "int_map2": int_map2,
        }


@dataclass(frozen=True)
class ModelDescriptor:
    family: str
    kwargs: dict[str, Any]


def build_model(descriptor: dict[str, Any]) -> nn.Module:
    family = str(descriptor["family"])
    kwargs = dict(descriptor.get("kwargs", {}))
    registry: dict[str, type[nn.Module]] = {
        "proposed": MPDProposed,
        "legacy_aligned": MPDProposed,
        "strong_rescnn": StrongResCNN,
        "resunet": ResUNet,
        "mlp_mixer": MLPMixer,
        "plain_transformer": PlainTransformer,
    }
    if family not in registry:
        if family in {
            "static_first", "static_mean", "tcn", "temporal_tcn", "lstm",
            "temporal_lstm", "temporal_mpd_transformer",
        }:
            from .temporal import build_temporal_model

            return build_temporal_model(descriptor)
        raise KeyError(f"Unknown model family: {family}")
    if family == "legacy_aligned":
        kwargs.setdefault("mode", "full")
    return registry[family](**kwargs)


def canonical_model_signature(descriptor: dict[str, Any]) -> str:
    family = str(descriptor["family"])
    kwargs = dict(descriptor.get("kwargs", {}))
    if family in {"proposed", "legacy_aligned"}:
        defaults = {
            "mode": "full",
            "d_model": 128,
            "nhead": 4,
            "mlp_ratio": 4,
            "token_grid": 32,
            "cls_dropout": 0.3,
        }
        defaults.update(kwargs)
        normalized = {"implementation": "mpd_proposed", "kwargs": defaults}
    else:
        normalized = {"implementation": family, "kwargs": kwargs}
    encoded = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def canonical_run_signature(
    descriptor: dict[str, Any],
    loss_variant: str,
    dataset_contract_sha256: str,
    split_sha256: str,
) -> str:
    value = {
        "model_signature": canonical_model_signature(descriptor),
        "loss_variant": str(loss_variant),
        "dataset_contract_sha256": str(dataset_contract_sha256),
        "split_sha256": str(split_sha256),
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def parameter_counts(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    active = model.active_parameter_count() if isinstance(model, MPDProposed) else total
    return {"total": total, "trainable": trainable, "active": int(active)}


def trace_active_parameters(model: nn.Module, device: torch.device | str = "cpu") -> dict[str, Any]:
    """Audit parameters structurally reachable from the four formal outputs."""
    target = torch.device(device)
    model = model.to(target)
    model.train()
    model.zero_grad(set_to_none=True)
    generator = torch.Generator(device=target).manual_seed(20260715)
    map1 = torch.randn(1, 3, 32, 32, generator=generator, device=target)
    map2 = torch.randn(1, 3, 32, 32, generator=generator, device=target)
    imu = torch.randn(1, 6, generator=generator, device=target)
    angle3 = torch.randn(1, 3, generator=generator, device=target)
    outputs = model(map1, map2, imu, angle3)
    scalar = sum(outputs[name].sum() for name in OUTPUT_NAMES if outputs[name] is not None)
    scalar.backward()
    active_names = [name for name, parameter in model.named_parameters() if parameter.grad is not None]
    inactive_names = [name for name, parameter in model.named_parameters() if parameter.grad is None]
    active = sum(parameter.numel() for name, parameter in model.named_parameters() if name in active_names)
    model.zero_grad(set_to_none=True)
    return {
        "active": int(active),
        "active_names": active_names,
        "inactive_names": inactive_names,
    }
