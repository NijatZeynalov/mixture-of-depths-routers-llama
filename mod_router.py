import json
import os
from dataclasses import asdict, dataclass
from typing import List, Optional

import torch
import torch.nn as nn


@dataclass
class RouterConfig:
    """Lightweight config for Mixture-of-Depths routers."""

    router_hidden_size: int = 256
    dropout: float = 0.0
    pattern: str = "all"  # "all" | "every_other"
    start_layer: int = 0
    stop_layer: Optional[int] = None
    hard_threshold: Optional[float] = None  # enables inference-time hard skip
    layer_norm_input: bool = False

    def to_dict(self):
        return asdict(self)

    def save_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


class RouterUsageRecorder:
    """Tracks gate usage for compute penalty."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._gates: List[torch.Tensor] = []

    def add(self, gates: torch.Tensor):
        # gates: [batch, seq, 1]
        self._gates.append(gates)

    def average(self) -> Optional[torch.Tensor]:
        if not self._gates:
            return None
        means = [g.mean() for g in self._gates]
        return torch.stack(means).mean()

    def per_layer(self) -> List[torch.Tensor]:
        return [g.mean() for g in self._gates]


class TokenRouter(nn.Module):
    """Tiny router: token-level gate in [0,1]."""

    def __init__(self, hidden_size: int, router_hidden_size: int = 256, dropout: float = 0.0, layer_norm_input: bool = False):
        super().__init__()
        inner = []
        if layer_norm_input:
            self.ln = nn.LayerNorm(hidden_size)
        else:
            self.ln = None
        if router_hidden_size > 0:
            inner.extend(
                [
                    nn.Linear(hidden_size, router_hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(router_hidden_size, 1),
                ]
            )
        else:
            inner.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*inner)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states
        if self.ln is not None:
            x = self.ln(x)
        gate = torch.sigmoid(self.net(x))  # [batch, seq, 1]
        return gate


class RouterDecoderLayer(nn.Module):
    """Wraps a decoder layer with soft/hard gating."""

    def __init__(
        self,
        base_layer: nn.Module,
        router: nn.Module,
        recorder: Optional[RouterUsageRecorder] = None,
        hard_threshold: Optional[float] = None,
        layer_index: int = 0,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.router = router
        self.recorder = recorder
        self.hard_threshold = hard_threshold
        self.layer_index = layer_index
        if freeze_base:
            for p in self.base_layer.parameters():
                p.requires_grad = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        gates = self.router(hidden_states)
        if self.recorder is not None:
            self.recorder.add(gates)

        # Inference-time hard skip (only when not using cache for safety)
        if (not self.training) and self.hard_threshold is not None and not use_cache:
            gate_mean = gates.mean()
            if gate_mean < self.hard_threshold:
                outputs = (hidden_states,)
                if output_attentions:
                    outputs = outputs + (None,)
                return outputs

        base_outputs = self.base_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )

        layer_hidden = base_outputs[0]
        mixed = gates * layer_hidden + (1 - gates) * hidden_states

        outputs = (mixed,) + base_outputs[1:]
        return outputs


def freeze_base_model(model: nn.Module):
    """Freeze everything; routers will be re-enabled after attach."""
    for _, param in model.named_parameters():
        param.requires_grad = False


def _enable_router_grads(model: nn.Module):
    for module in model.modules():
        if isinstance(module, (TokenRouter,)):
            for p in module.parameters():
                p.requires_grad = True


def attach_mod_routers(model: nn.Module, cfg: RouterConfig, recorder: Optional[RouterUsageRecorder] = None):
    """Inject routers into selected layers and freeze base weights."""
    if recorder is None:
        recorder = RouterUsageRecorder()
    layers = model.model.layers
    total_layers = len(layers)
    stop_layer = cfg.stop_layer if cfg.stop_layer is not None else total_layers
    freeze_base_model(model)

    for idx, layer in enumerate(layers):
        within_range = cfg.start_layer <= idx < stop_layer
        pattern_ok = cfg.pattern == "all" or (cfg.pattern == "every_other" and idx % 2 == 0)
        if not (within_range and pattern_ok):
            continue
        router = TokenRouter(
            hidden_size=model.config.hidden_size,
            router_hidden_size=cfg.router_hidden_size,
            dropout=cfg.dropout,
            layer_norm_input=cfg.layer_norm_input,
        )
        wrapped = RouterDecoderLayer(
            base_layer=layer,
            router=router,
            recorder=recorder,
            hard_threshold=cfg.hard_threshold,
            layer_index=idx,
            freeze_base=True,
        )
        layers[idx] = wrapped

    _enable_router_grads(model)
    model.router_usage_recorder = recorder
    model.router_config = cfg
    return model


def set_router_mode(model: nn.Module, hard_threshold: Optional[float] = None):
    """Enable or disable hard skipping at inference."""
    for layer in model.model.layers:
        if isinstance(layer, RouterDecoderLayer):
            layer.hard_threshold = hard_threshold


def load_with_routers(
    model_name_or_path: str,
    router_config_path: Optional[str] = None,
    base_model_name_or_path: Optional[str] = None,
    device_map=None,
    hard_threshold: Optional[float] = None,
):
    """Load a base LLaMA, attach routers, and hydrate router weights."""
    from transformers import AutoModelForCausalLM

    config_path = router_config_path or os.path.join(model_name_or_path, "router_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Router config not found at {config_path}")
    cfg = RouterConfig.from_json(config_path)

    base_path = base_model_name_or_path or model_name_or_path
    model = AutoModelForCausalLM.from_pretrained(base_path, device_map=device_map)
    attach_mod_routers(model, cfg)
    set_router_mode(model, hard_threshold=hard_threshold if hard_threshold is not None else cfg.hard_threshold)

    # Load router weights (and any unfrozen params) with strict=False
    state_dict_path = None
    if os.path.exists(os.path.join(model_name_or_path, "pytorch_model.bin")):
        state_dict_path = os.path.join(model_name_or_path, "pytorch_model.bin")
        state_dict = torch.load(state_dict_path, map_location="cpu")
    elif os.path.exists(os.path.join(model_name_or_path, "model.safetensors")):
        from safetensors.torch import load_file

        state_dict_path = os.path.join(model_name_or_path, "model.safetensors")
        state_dict = load_file(state_dict_path)
    else:
        raise FileNotFoundError(f"No model weights found in {model_name_or_path}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"[warn] Unexpected keys when loading: {unexpected}")
    if missing:
        print(f"[warn] Missing keys when loading: {missing}")
    return model
