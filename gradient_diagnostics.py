#!/usr/bin/env python3
"""Diagnose how ASTA and CFN alter backpropagation in CMeRT/SACG.

This standalone script reuses the repository's configuration, data loader,
model, criterion and optimizer. It runs matched short training traces for four
variants (Baseline, ASTA, CFN and SACG), then exports module-level gradients,
local input/output gradient ratios, task-gradient conflict, ASTA gate dynamics,
loss traces and diagnostic figures.

The current loss implementation follows the non-EK100 path in main.py:
    total_loss = detant_loss + future_loss
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

torch = None
F = None
get_cfg = None
build_criterion = None
build_data_loader = None
build_model = None
build_optimizer = None
build_scheduler = None
setup_environment = None
setup_random_seed = None
assert_and_infer_cfg = None


VARIANT_ORDER = ("baseline", "asta", "cfn", "sacg")
VARIANT_LABELS = {
    "baseline": "Baseline",
    "asta": "ASTA",
    "cfn": "CFN",
    "sacg": "SACG",
}
PREFERRED_MODULES = (
    "long_compressor",
    "short_decoder",
    "generator",
    "asta",
    "cfn",
    "refinement",
    "classifier",
    "other",
    "global",
)


def load_runtime() -> None:
    global torch, F, get_cfg, build_criterion, build_data_loader, build_model
    global build_optimizer, build_scheduler, setup_environment, setup_random_seed
    global assert_and_infer_cfg
    if torch is not None:
        return
    try:
        import torch as torch_module
        import torch.nn.functional as functional
        from rekognition_online_action_detection.config.defaults import get_cfg as cfg_factory
        from rekognition_online_action_detection.criterions import (
            build_criterion as criterion_factory,
        )
        from rekognition_online_action_detection.datasets import (
            build_data_loader as data_loader_factory,
        )
        from rekognition_online_action_detection.models import build_model as model_factory
        from rekognition_online_action_detection.optimizers import (
            build_optimizer as optimizer_factory,
        )
        from rekognition_online_action_detection.optimizers import (
            build_scheduler as scheduler_factory,
        )
        from rekognition_online_action_detection.utils.env import (
            setup_environment as environment_factory,
        )
        from rekognition_online_action_detection.utils.env import (
            setup_random_seed as random_seed_factory,
        )
        from rekognition_online_action_detection.utils.parser import (
            assert_and_infer_cfg as infer_cfg,
        )
    except ImportError as exc:
        raise RuntimeError(
            "The full diagnostic requires the CMeRT PyTorch environment. Activate "
            "the same conda environment used to run main.py."
        ) from exc
    torch = torch_module
    F = functional
    get_cfg = cfg_factory
    build_criterion = criterion_factory
    build_data_loader = data_loader_factory
    build_model = model_factory
    build_optimizer = optimizer_factory
    build_scheduler = scheduler_factory
    setup_environment = environment_factory
    setup_random_seed = random_seed_factory
    assert_and_infer_cfg = infer_cfg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Matched backpropagation diagnostics for Baseline/ASTA/CFN/SACG."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/THUMOS/cmert_long256_work4_kinetics_1x.yaml"),
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANT_ORDER,
        default=list(VARIANT_ORDER),
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--conflict-every", type=int, default=25)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("gradient_diagnostics"))
    parser.add_argument("--assa-init", type=float, nargs=2, default=[0.0, 0.0])
    parser.add_argument("--cfn-scale", type=float, default=0.05)
    parser.add_argument("--cfn-temperature", type=float, default=1.0)
    parser.add_argument("--cfn-position", default="cross")
    parser.add_argument(
        "--extra-opt",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional YACS override; repeat for multiple values.",
    )
    parser.add_argument(
        "--track-updates",
        action="store_true",
        help="Measure actual parameter-update ratios on logging steps (slower).",
    )
    parser.add_argument(
        "--no-update",
        action="store_true",
        help="Run backward passes without optimizer updates.",
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_extra_opts(items: Sequence[str]) -> list[str]:
    opts: list[str] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"--extra-opt must use KEY=VALUE syntax: {item!r}")
        key, value = item.split("=", 1)
        if not key or not value:
            raise ValueError(f"Invalid --extra-opt: {item!r}")
        opts.extend((key, value))
    return opts


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def config_cli_path(config_path: Path) -> str:
    try:
        return config_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return config_path.as_posix()


def apply_variant(cfg: Any, variant: str, args: argparse.Namespace) -> None:
    use_asta = variant in ("asta", "sacg")
    use_cfn = variant in ("cfn", "sacg")
    cfg.MODEL.LSTR.GEN_CROSS_ATTN_TYPE = "assa" if use_asta else "dotproduct"
    cfg.MODEL.LSTR.GEN_CROSS_ASSA_INIT_W = [float(value) for value in args.assa_init]
    cfg.MODEL.LSTR.GEN_CONTRANORM = use_cfn
    cfg.MODEL.LSTR.GEN_CONTRANORM_POS = args.cfn_position if use_cfn else "none"
    cfg.MODEL.LSTR.GEN_CONTRANORM_SCALE = float(args.cfn_scale)
    cfg.MODEL.LSTR.GEN_CONTRANORM_TEMP = float(args.cfn_temperature)


def build_cfg(args: argparse.Namespace, variant: str):
    config_path = resolve_path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(str(config_path))
    extra_opts = parse_extra_opts(args.extra_opt)
    if extra_opts:
        cfg.merge_from_list(extra_opts)
    cfg.SEED = args.seed
    cfg.GPU = str(args.gpu)
    cfg.DATA_LOADER.NUM_WORKERS = args.workers
    cfg.DATA_LOADER.PIN_MEMORY = False
    if args.batch_size is not None:
        cfg.DATA_LOADER.BATCH_SIZE = args.batch_size
    cfg.OUTPUT_DIR = str(resolve_path(args.output_dir) / "unused_checkpoints" / variant)
    cfg.MODEL.CHECKPOINT = None
    apply_variant(cfg, variant, args)
    parser_args = SimpleNamespace(
        gpu=str(args.gpu),
        config_file=config_cli_path(config_path),
    )
    assert_and_infer_cfg(cfg, parser_args)
    cfg.TEST = 0
    if cfg.DATA.DATA_NAME == "EK100":
        raise NotImplementedError(
            "This diagnostic currently targets the non-EK100 loss path used by "
            "THUMOS, TVSeries and CrossTask."
        )
    return cfg


def variant_settings(args: argparse.Namespace, variant: str) -> dict[str, object]:
    return {
        "variant": variant,
        "cross_attention": "assa" if variant in ("asta", "sacg") else "dotproduct",
        "assa_init": [float(value) for value in args.assa_init],
        "cfn_enabled": variant in ("cfn", "sacg"),
        "cfn_position": args.cfn_position if variant in ("cfn", "sacg") else "none",
        "cfn_scale": float(args.cfn_scale),
        "cfn_temperature": float(args.cfn_temperature),
    }


def first_tensor(value: Any, require_grad: bool = False) -> torch.Tensor | None:
    if torch.is_tensor(value):
        if not require_grad or value.requires_grad:
            return value
        return None
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = first_tensor(item, require_grad=require_grad)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = first_tensor(item, require_grad=require_grad)
            if tensor is not None:
                return tensor
    return None


def tensor_l2(tensor: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(tensor.detach().float()).item())


def tensor_finite_fraction(tensor: torch.Tensor) -> float:
    detached = tensor.detach()
    return float(torch.isfinite(detached).float().mean().item())


def token_cosine(tensor: torch.Tensor) -> float:
    value = tensor.detach().float()
    if value.ndim == 3:
        value = value.transpose(0, 1)
    elif value.ndim == 2:
        value = value.unsqueeze(0)
    else:
        return float("nan")
    token_count = value.shape[1]
    if token_count < 2:
        return float("nan")
    normalized = F.normalize(value, dim=-1)
    similarity = torch.bmm(normalized, normalized.transpose(1, 2))
    diagonal = torch.diagonal(similarity, dim1=1, dim2=2).sum()
    denominator = similarity.shape[0] * token_count * (token_count - 1)
    return float(((similarity.sum() - diagonal) / denominator).item())


def aggregate_numbers(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.mean(finite)) if finite else float("nan")


class BackpropRecorder:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.handles: list[Any] = []
        self.asta_modules: list[tuple[str, torch.nn.Module]] = []
        self.active = False
        self.gradient_stats: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
        self.activation_stats: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
        self.asta_forward_stats: list[dict[str, float]] = []
        self._register_modules()

    def _register_modules(self) -> None:
        for name, module in self.model.named_modules():
            group = self._module_group(name, module)
            if group is None:
                continue
            class_name = module.__class__.__name__
            if group == "asta":
                self.asta_modules.append((name, module))
            self.handles.append(
                module.register_forward_hook(self._make_forward_hook(name, group, class_name))
            )

    @staticmethod
    def _module_group(name: str, module: torch.nn.Module) -> str | None:
        class_name = module.__class__.__name__
        if class_name == "ASSADotProductAttention" and "gen_layer" in name:
            return "asta"
        if class_name == "ContraNorm1D" and "gen_layer" in name:
            return "cfn"
        if name == "gen_query":
            return "generator_query"
        if name == "gen_layer":
            return "generator"
        if name == "dec_modules":
            return "short_decoder"
        if name == "classifier":
            return "classifier"
        if name.startswith("enc_modules.") and name.count(".") == 1:
            return "long_compressor"
        if name.startswith("work_fusions.") and name.count(".") == 1:
            return "refinement"
        return None

    def begin_step(self, active: bool) -> None:
        self.active = active
        self.gradient_stats.clear()
        self.activation_stats.clear()
        self.asta_forward_stats.clear()

    def _record_gradient(self, group: str, signal: str, gradient: torch.Tensor) -> None:
        self.gradient_stats[(group, signal)].append(
            {
                "grad_l2": tensor_l2(gradient),
                "finite_fraction": tensor_finite_fraction(gradient),
                "max_abs": float(gradient.detach().abs().max().item()),
            }
        )

    def _attach_gradient_hook(
        self, group: str, signal: str, tensor: torch.Tensor | None
    ) -> None:
        if tensor is None or not tensor.requires_grad:
            return
        tensor.register_hook(
            lambda gradient, g=group, s=signal: self._record_gradient(g, s, gradient)
        )

    def _make_forward_hook(self, name: str, group: str, class_name: str):
        def hook(module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            if not self.active:
                return
            input_tensor = first_tensor(inputs, require_grad=True)
            output_tensor = first_tensor(output, require_grad=True)
            self._attach_gradient_hook(group, "input", input_tensor)
            self._attach_gradient_hook(group, "output", output_tensor)
            for signal, tensor in (("input", input_tensor), ("output", output_tensor)):
                if tensor is None:
                    continue
                self.activation_stats[(group, signal)].append(
                    {
                        "activation_l2": tensor_l2(tensor),
                        "activation_finite_fraction": tensor_finite_fraction(tensor),
                        "token_cosine": token_cosine(tensor)
                        if group in ("cfn", "generator")
                        else float("nan"),
                    }
                )
            if class_name == "ASSADotProductAttention" and len(inputs) >= 2:
                q, k = inputs[0], inputs[1]
                with torch.no_grad():
                    scores = torch.bmm(q.detach(), k.detach().transpose(1, 2))
                    self.asta_forward_stats.append(
                        {
                            "positive_score_fraction": float((scores > 0).float().mean().item()),
                            "fallback_row_fraction": float(
                                (scores.amax(dim=-1) <= 0).float().mean().item()
                            ),
                        }
                    )

        return hook

    def activation_rows(
        self, variant: str, seed: int, step: int, epoch: int, batch: int
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        keys = sorted(set(self.gradient_stats) | set(self.activation_stats))
        for group, signal in keys:
            gradients = self.gradient_stats.get((group, signal), [])
            activations = self.activation_stats.get((group, signal), [])
            grad_l2 = math.sqrt(sum(item["grad_l2"] ** 2 for item in gradients))
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "step": step,
                    "epoch": epoch,
                    "batch": batch,
                    "module": group,
                    "signal": signal,
                    "grad_l2": grad_l2 if gradients else float("nan"),
                    "grad_finite_fraction": aggregate_numbers(
                        item["finite_fraction"] for item in gradients
                    ),
                    "grad_max_abs": max(
                        (item["max_abs"] for item in gradients), default=float("nan")
                    ),
                    "activation_l2": aggregate_numbers(
                        item["activation_l2"] for item in activations
                    ),
                    "activation_finite_fraction": aggregate_numbers(
                        item["activation_finite_fraction"] for item in activations
                    ),
                    "token_cosine": aggregate_numbers(
                        item["token_cosine"] for item in activations
                    ),
                }
            )
        return rows

    def asta_rows(
        self, variant: str, seed: int, step: int, epoch: int, batch: int
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        positive = aggregate_numbers(
            item["positive_score_fraction"] for item in self.asta_forward_stats
        )
        fallback = aggregate_numbers(
            item["fallback_row_fraction"] for item in self.asta_forward_stats
        )
        for name, module in self.asta_modules:
            gate = module._branch_weights().detach().float().cpu()
            gradient = module.w.grad
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "step": step,
                    "epoch": epoch,
                    "batch": batch,
                    "module_name": name,
                    "dense_weight": float(gate[0].item()),
                    "sparse_weight": float(gate[1].item()),
                    "w_grad_l2": tensor_l2(gradient)
                    if gradient is not None
                    else float("nan"),
                    "positive_score_fraction": positive,
                    "fallback_row_fraction": fallback,
                }
            )
        return rows

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def parameter_group(name: str) -> str:
    if "gen_layer" in name and ".attention.w" in name:
        return "asta"
    if "gen_layer" in name and "contranorm" in name:
        return "cfn"
    if name.startswith(("gen_query", "gen_layer", "final_query")):
        return "generator"
    if name.startswith("work_fusions"):
        return "refinement"
    if name.startswith(("enc_queries", "enc_modules", "feature_head_long")):
        return "long_compressor"
    if name.startswith(("dec_modules", "feature_head_work")):
        return "short_decoder"
    if name.startswith("classifier"):
        return "classifier"
    return "other"


def collect_parameter_rows(
    model: torch.nn.Module,
    variant: str,
    seed: int,
    step: int,
    epoch: int,
    batch: int,
    learning_rate: float,
) -> list[dict[str, object]]:
    grouped: dict[str, list[tuple[str, torch.nn.Parameter]]] = defaultdict(list)
    all_parameters: list[tuple[str, torch.nn.Parameter]] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        grouped[parameter_group(name)].append((name, parameter))
        all_parameters.append((name, parameter))
    grouped["global"] = all_parameters

    rows: list[dict[str, object]] = []
    for group in PREFERRED_MODULES:
        parameters = grouped.get(group, [])
        if not parameters:
            continue
        parameter_elements = sum(parameter.numel() for _, parameter in parameters)
        parameter_sq = 0.0
        gradient_sq = 0.0
        gradient_elements = 0
        missing_elements = 0
        zero_elements = 0
        finite_elements = 0
        max_abs = 0.0
        for _, parameter in parameters:
            detached = parameter.detach().float()
            parameter_sq += float(detached.square().sum().item())
            if parameter.grad is None:
                missing_elements += parameter.numel()
                continue
            gradient = parameter.grad.detach().float()
            gradient_elements += gradient.numel()
            gradient_sq += float(gradient.square().sum().item())
            zero_elements += int((gradient == 0).sum().item())
            finite_elements += int(torch.isfinite(gradient).sum().item())
            max_abs = max(max_abs, float(gradient.abs().max().item()))
        parameter_l2 = math.sqrt(parameter_sq)
        gradient_l2 = math.sqrt(gradient_sq)
        rows.append(
            {
                "variant": variant,
                "seed": seed,
                "step": step,
                "epoch": epoch,
                "batch": batch,
                "module": group,
                "parameter_count": parameter_elements,
                "gradient_l2": gradient_l2,
                "parameter_l2": parameter_l2,
                "gradient_parameter_ratio": gradient_l2 / (parameter_l2 + 1e-12),
                "lr_gradient_parameter_ratio": learning_rate
                * gradient_l2
                / (parameter_l2 + 1e-12),
                "zero_gradient_fraction": zero_elements / max(gradient_elements, 1),
                "missing_gradient_fraction": missing_elements / max(parameter_elements, 1),
                "finite_gradient_fraction": finite_elements / max(gradient_elements, 1),
                "maximum_absolute_gradient": max_abs,
                "learning_rate": learning_rate,
                "update_parameter_ratio": float("nan"),
            }
        )
    return rows


def snapshot_parameters(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def update_ratios(
    model: torch.nn.Module, before: dict[str, torch.Tensor]
) -> dict[str, float]:
    delta_sq: dict[str, float] = defaultdict(float)
    parameter_sq: dict[str, float] = defaultdict(float)
    for name, parameter in model.named_parameters():
        if name not in before:
            continue
        group = parameter_group(name)
        old = before[name].float()
        new = parameter.detach().cpu().float()
        delta = new - old
        for key in (group, "global"):
            delta_sq[key] += float(delta.square().sum().item())
            parameter_sq[key] += float(old.square().sum().item())
    return {
        group: math.sqrt(delta_sq[group]) / (math.sqrt(parameter_sq[group]) + 1e-12)
        for group in delta_sq
    }


def load_compatible_checkpoint(model: torch.nn.Module, checkpoint: Path) -> dict[str, int]:
    payload = torch.load(str(checkpoint), map_location="cpu")
    source = payload.get("model_state_dict", payload)
    target = model.state_dict()
    compatible = {
        key: value
        for key, value in source.items()
        if key in target and tuple(value.shape) == tuple(target[key].shape)
    }
    result = model.load_state_dict(compatible, strict=False)
    return {
        "checkpoint_keys": len(source),
        "loaded_keys": len(compatible),
        "missing_keys": len(result.missing_keys),
        "unexpected_keys": len(result.unexpected_keys),
    }


def normalize_scores(scores: Any) -> list[torch.Tensor]:
    if torch.is_tensor(scores):
        return [scores]
    return list(scores)


def compute_losses(cfg: Any, criterion: Any, outputs: Any, targets: Any):
    scores, future_scores = outputs
    scores = normalize_scores(scores)
    future_scores = normalize_scores(future_scores)
    det_target, ant_target, future_target = targets
    det_target = det_target.to(scores[0].device)
    ant_target = ant_target.to(scores[0].device)
    future_target = future_target.to(scores[0].device)
    total_target = det_target
    if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
        total_target = torch.cat((det_target, ant_target), dim=1)

    loss_name = list(zip(*cfg.MODEL.CRITERIONS))[0][0]
    stage1_weight = cfg.SOLVER.get("S1_W", 0.2)
    stage2_weight = cfg.SOLVER.get("S2_W", 1.0)
    future_weight = cfg.SOLVER.get("F_W", 0.5)

    detant_loss = None
    for index, score in enumerate(scores):
        weight = stage1_weight if index == 0 else stage2_weight
        value = weight * criterion[loss_name](
            score.reshape(-1, cfg.DATA.NUM_CLASSES),
            total_target.reshape(-1, cfg.DATA.NUM_CLASSES),
        )
        detant_loss = value if detant_loss is None else detant_loss + value
    if detant_loss is None:
        raise RuntimeError("The model returned no detection/anticipation scores.")

    future_loss = torch.zeros((), device=detant_loss.device)
    for score in future_scores:
        future_loss = future_loss + future_weight * criterion["MCE"](
            score.reshape(-1, cfg.DATA.NUM_CLASSES),
            future_target.reshape(-1, cfg.DATA.NUM_CLASSES),
        )
    return detant_loss, future_loss, detant_loss + future_loss


def task_gradient_conflict(
    detant_loss: torch.Tensor,
    future_loss: torch.Tensor,
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
) -> dict[str, float]:
    parameters = [parameter for _, parameter in named_parameters]
    if not parameters or not future_loss.requires_grad:
        return {
            "gradient_cosine": float("nan"),
            "detant_gradient_l2": float("nan"),
            "future_gradient_l2": float("nan"),
            "future_detant_norm_ratio": float("nan"),
            "shared_parameter_fraction": float("nan"),
        }
    detant_gradients = torch.autograd.grad(
        detant_loss, parameters, retain_graph=True, allow_unused=True
    )
    future_gradients = torch.autograd.grad(
        future_loss, parameters, retain_graph=True, allow_unused=True
    )
    dot = 0.0
    detant_sq = 0.0
    future_sq = 0.0
    shared = 0
    for detant_gradient, future_gradient in zip(detant_gradients, future_gradients):
        if detant_gradient is None or future_gradient is None:
            continue
        shared += 1
        first = detant_gradient.detach().float()
        second = future_gradient.detach().float()
        dot += float((first * second).sum().item())
        detant_sq += float(first.square().sum().item())
        future_sq += float(second.square().sum().item())
    detant_norm = math.sqrt(detant_sq)
    future_norm = math.sqrt(future_sq)
    return {
        "gradient_cosine": dot / (detant_norm * future_norm + 1e-12),
        "detant_gradient_l2": detant_norm,
        "future_gradient_l2": future_norm,
        "future_detant_norm_ratio": future_norm / (detant_norm + 1e-12),
        "shared_parameter_fraction": shared / max(len(parameters), 1),
    }


def conflict_parameters(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    selected = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter_group(name) in ("generator", "asta", "cfn"):
            selected.append((name, parameter))
    return selected


def move_inputs_to_device(data: Any, device: torch.device) -> list[torch.Tensor]:
    return [value.to(device) for value in data[:-1]]


def append_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def reset_outputs(output_dir: Path) -> None:
    for name in (
        "parameter_gradients.csv",
        "activation_gradients.csv",
        "task_gradient_conflict.csv",
        "asta_gate_diagnostics.csv",
        "loss_trace.csv",
        "local_gradient_ratios.csv",
        "diagnostic_summary.csv",
    ):
        path = output_dir / name
        if path.exists():
            path.unlink()


def run_variant(
    args: argparse.Namespace, variant: str, output_dir: Path
) -> dict[str, object]:
    cfg = build_cfg(args, variant)
    device = setup_environment(cfg)
    data_loader = build_data_loader(cfg, "train")
    model = build_model(cfg, device)
    checkpoint_report: dict[str, int] = {}
    if args.checkpoint is not None:
        checkpoint = resolve_path(args.checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        checkpoint_report = load_compatible_checkpoint(model, checkpoint)
    criterion = build_criterion(cfg, device)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer, len(data_loader))
    recorder = BackpropRecorder(model)
    shared_parameters = conflict_parameters(model)

    setup_random_seed(args.seed)
    model.train()
    global_step = 0
    epoch = 0
    stop = False
    while not stop:
        epoch += 1
        for batch_index, data in enumerate(data_loader, start=1):
            global_step += 1
            should_log = global_step == 1 or global_step % args.log_every == 0
            should_probe_conflict = (
                global_step == 1 or global_step % args.conflict_every == 0
            )
            recorder.begin_step(should_log)
            optimizer.zero_grad()
            outputs = model(*move_inputs_to_device(data, device))
            detant_loss, future_loss, total_loss = compute_losses(
                cfg, criterion, outputs, data[-1]
            )

            if should_probe_conflict:
                conflict = task_gradient_conflict(
                    detant_loss, future_loss, shared_parameters
                )
                append_csv(
                    output_dir / "task_gradient_conflict.csv",
                    [
                        {
                            "variant": variant,
                            "seed": args.seed,
                            "step": global_step,
                            "epoch": epoch,
                            "batch": batch_index,
                            **conflict,
                        }
                    ],
                )

            total_loss.backward()
            learning_rate = float(optimizer.param_groups[0]["lr"])
            parameter_rows: list[dict[str, object]] = []
            activation_rows: list[dict[str, object]] = []
            asta_rows: list[dict[str, object]] = []
            before = None
            if should_log:
                parameter_rows = collect_parameter_rows(
                    model,
                    variant,
                    args.seed,
                    global_step,
                    epoch,
                    batch_index,
                    learning_rate,
                )
                activation_rows = recorder.activation_rows(
                    variant, args.seed, global_step, epoch, batch_index
                )
                asta_rows = recorder.asta_rows(
                    variant, args.seed, global_step, epoch, batch_index
                )
                if args.track_updates and not args.no_update:
                    before = snapshot_parameters(model)

            if not args.no_update:
                optimizer.step()
                scheduler.step()
            if before is not None:
                ratios = update_ratios(model, before)
                for row in parameter_rows:
                    row["update_parameter_ratio"] = ratios.get(
                        str(row["module"]), float("nan")
                    )

            append_csv(
                output_dir / "loss_trace.csv",
                [
                    {
                        "variant": variant,
                        "seed": args.seed,
                        "step": global_step,
                        "epoch": epoch,
                        "batch": batch_index,
                        "detant_loss": float(detant_loss.detach().item()),
                        "future_loss": float(future_loss.detach().item()),
                        "total_loss": float(total_loss.detach().item()),
                        "learning_rate": learning_rate,
                    }
                ],
            )
            if should_log:
                append_csv(output_dir / "parameter_gradients.csv", parameter_rows)
                append_csv(output_dir / "activation_gradients.csv", activation_rows)
                append_csv(output_dir / "asta_gate_diagnostics.csv", asta_rows)
                global_row = next(
                    (row for row in parameter_rows if row["module"] == "global"), None
                )
                global_gradient = (
                    float(global_row["gradient_l2"]) if global_row is not None else float("nan")
                )
                print(
                    f"[{VARIANT_LABELS[variant]}] step {global_step:4d}/{args.steps} "
                    f"loss={total_loss.detach().item():.4f} "
                    f"global_grad={global_gradient:.4g}",
                    flush=True,
                )
            if global_step >= args.steps:
                stop = True
                break
        if len(data_loader) == 0:
            raise RuntimeError("The training data loader is empty.")

    recorder.close()
    del model, optimizer, scheduler, data_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        **variant_settings(args, variant),
        **checkpoint_report,
        "steps_completed": global_step,
        "device": str(device),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def float_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def build_local_gradient_ratios(output_dir: Path) -> list[dict[str, object]]:
    rows = read_csv(output_dir / "activation_gradients.csv")
    grouped: dict[tuple[str, str, str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = (row["variant"], row["seed"], row["step"], row["module"])
        grouped[key][row["signal"]] = float_value(row["grad_l2"])
    ratios = []
    for (variant, seed, step, module), signals in sorted(grouped.items()):
        input_gradient = signals.get("input", float("nan"))
        output_gradient = signals.get("output", float("nan"))
        ratios.append(
            {
                "variant": variant,
                "seed": seed,
                "step": step,
                "module": module,
                "input_gradient_l2": input_gradient,
                "output_gradient_l2": output_gradient,
                "input_output_gradient_ratio": input_gradient
                / (output_gradient + 1e-12),
            }
        )
    path = output_dir / "local_gradient_ratios.csv"
    if path.exists():
        path.unlink()
    append_csv(path, ratios)
    return ratios


def build_summary(output_dir: Path) -> list[dict[str, object]]:
    rows = read_csv(output_dir / "parameter_gradients.csv")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["variant"], row["module"])].append(row)
    summary = []
    for (variant, module), items in sorted(grouped.items()):
        gradient_norms = [float_value(item["gradient_l2"]) for item in items]
        finite_fractions = [float_value(item["finite_gradient_fraction"]) for item in items]
        missing_fractions = [float_value(item["missing_gradient_fraction"]) for item in items]
        zero_fractions = [float_value(item["zero_gradient_fraction"]) for item in items]
        valid_norms = [value for value in gradient_norms if math.isfinite(value)]
        summary.append(
            {
                "variant": variant,
                "module": module,
                "observations": len(items),
                "median_gradient_l2": statistics.median(valid_norms)
                if valid_norms
                else float("nan"),
                "minimum_finite_gradient_fraction": min(finite_fractions),
                "maximum_missing_gradient_fraction": max(missing_fractions),
                "median_zero_gradient_fraction": statistics.median(zero_fractions),
            }
        )
    path = output_dir / "diagnostic_summary.csv"
    if path.exists():
        path.unlink()
    append_csv(path, summary)
    return summary


def import_plotting():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib and numpy.") from exc
    return np, plt


def configure_plot_style(plt: Any) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def save_figure(fig: Any, stem: Path) -> None:
    for suffix, dpi in ((".svg", 300), (".pdf", 300), (".png", 300), (".tiff", 600)):
        fig.savefig(stem.with_suffix(suffix), dpi=dpi, bbox_inches="tight", facecolor="white")


def clean_axis(axis: Any) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.7)
    axis.tick_params(direction="out", length=2.5, width=0.7)
    axis.set_axisbelow(True)


def plot_diagnostics(output_dir: Path) -> None:
    np, plt = import_plotting()
    configure_plot_style(plt)
    colors = {
        "baseline": "#6F7782",
        "asta": "#D55E00",
        "cfn": "#0072B2",
        "sacg": "#00796B",
    }
    parameter_rows = read_csv(output_dir / "parameter_gradients.csv")
    modules = [
        module
        for module in PREFERRED_MODULES
        if module != "global" and any(row["module"] == module for row in parameter_rows)
    ]
    if parameter_rows and modules:
        columns = 3
        rows_count = math.ceil(len(modules) / columns)
        fig, axes = plt.subplots(
            rows_count,
            columns,
            figsize=(7.15, max(2.1, rows_count * 1.9)),
            squeeze=False,
        )
        module_titles = {"asta": "ASTA", "cfn": "CFN"}
        for axis, module in zip(axes.flat, modules):
            for variant in VARIANT_ORDER:
                selected = [
                    row
                    for row in parameter_rows
                    if row["variant"] == variant and row["module"] == module
                ]
                if not selected:
                    continue
                selected.sort(key=lambda row: int(row["step"]))
                axis.plot(
                    [int(row["step"]) for row in selected],
                    [float_value(row["gradient_l2"]) for row in selected],
                    marker="o",
                    markersize=2.8,
                    color=colors[variant],
                    label=VARIANT_LABELS[variant],
                )
            axis.set_title(module_titles.get(module, module.replace("_", " ").title()))
            axis.set_xlabel("Training step")
            axis.set_ylabel("Gradient L2 norm")
            axis.set_yscale("log")
            clean_axis(axis)
        for axis in axes.flat[len(modules) :]:
            axis.set_visible(False)
        handle_map: dict[str, Any] = {}
        for axis in axes.flat:
            axis_handles, axis_labels = axis.get_legend_handles_labels()
            handle_map.update(zip(axis_labels, axis_handles))
        labels = [
            VARIANT_LABELS[variant]
            for variant in VARIANT_ORDER
            if VARIANT_LABELS[variant] in handle_map
        ]
        handles = [handle_map[label] for label in labels]
        if handles:
            fig.legend(
                handles,
                labels,
                frameon=False,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.995),
                ncol=4,
            )
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        save_figure(fig, output_dir / "module_gradient_norms")
        plt.close(fig)

    conflict_rows = read_csv(output_dir / "task_gradient_conflict.csv")
    if conflict_rows:
        fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.35))
        for variant in VARIANT_ORDER:
            selected = [row for row in conflict_rows if row["variant"] == variant]
            if not selected:
                continue
            selected.sort(key=lambda row: int(row["step"]))
            steps = [int(row["step"]) for row in selected]
            axes[0].plot(
                steps,
                [float_value(row["gradient_cosine"]) for row in selected],
                marker="o",
                color=colors[variant],
                label=VARIANT_LABELS[variant],
            )
            axes[1].plot(
                steps,
                [float_value(row["future_detant_norm_ratio"]) for row in selected],
                marker="o",
                color=colors[variant],
                label=VARIANT_LABELS[variant],
            )
        axes[0].axhline(0, color="#777777", lw=0.8, ls=(0, (3, 2)))
        axes[0].set_ylabel("Gradient cosine")
        axes[0].set_title("Task-gradient alignment")
        axes[1].axhline(1, color="#777777", lw=0.8, ls=(0, (3, 2)))
        axes[1].set_ylabel("Future/detection gradient norm")
        axes[1].set_title("Task-gradient magnitude")
        for axis in axes:
            axis.set_xlabel("Training step")
            clean_axis(axis)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                frameon=False,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.995),
                ncol=4,
            )
        fig.tight_layout(rect=(0, 0, 1, 0.86))
        save_figure(fig, output_dir / "task_gradient_conflict")
        plt.close(fig)

    gate_rows = read_csv(output_dir / "asta_gate_diagnostics.csv")
    local_rows = read_csv(output_dir / "local_gradient_ratios.csv")
    cfn_rows = [row for row in local_rows if row["module"] == "cfn"]
    if gate_rows or cfn_rows:
        fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.35), constrained_layout=True)
        for variant in ("asta", "sacg"):
            selected = [row for row in gate_rows if row["variant"] == variant]
            if not selected:
                continue
            grouped_steps: dict[int, list[dict[str, str]]] = defaultdict(list)
            for row in selected:
                grouped_steps[int(row["step"])].append(row)
            steps = sorted(grouped_steps)
            dense = [
                statistics.mean(float_value(row["dense_weight"]) for row in grouped_steps[step])
                for step in steps
            ]
            axes[0].plot(
                steps,
                dense,
                marker="o",
                color=colors[variant],
                label=f"{VARIANT_LABELS[variant]} dense",
            )
            axes[0].plot(
                steps,
                [1.0 - value for value in dense],
                marker="s",
                ls="--",
                color=colors[variant],
                label=f"{VARIANT_LABELS[variant]} sparse",
            )
        axes[0].set_ylim(-0.02, 1.02)
        axes[0].set_title("ASTA branch weights")
        axes[0].set_ylabel("Softmax gate weight")
        for variant in ("cfn", "sacg"):
            selected = [row for row in cfn_rows if row["variant"] == variant]
            selected.sort(key=lambda row: int(row["step"]))
            if selected:
                axes[1].plot(
                    [int(row["step"]) for row in selected],
                    [float_value(row["input_output_gradient_ratio"]) for row in selected],
                    marker="o",
                    color=colors[variant],
                    label=VARIANT_LABELS[variant],
                )
        axes[1].axhline(1, color="#777777", lw=0.8, ls=(0, (3, 2)))
        axes[1].set_yscale("log")
        axes[1].set_title("CFN local gradient transfer")
        axes[1].set_ylabel("Input/output gradient norm")
        for axis in axes:
            axis.set_xlabel("Training step")
            clean_axis(axis)
            axis.legend(frameon=False)
        save_figure(fig, output_dir / "asta_cfn_backprop")
        plt.close(fig)


def run(args: argparse.Namespace) -> int:
    if args.steps < 1:
        raise ValueError("--steps must be at least 1.")
    if args.log_every < 1 or args.conflict_every < 1:
        raise ValueError("--log-every and --conflict-every must be at least 1.")
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        build_local_gradient_ratios(output_dir)
        build_summary(output_dir)
        if not args.no_plot:
            plot_diagnostics(output_dir)
        print(f"Diagnostics plotted from existing CSV files: {output_dir}")
        return 0

    settings = [variant_settings(args, variant) for variant in args.variants]
    if args.dry_run:
        print(json.dumps({"output_dir": str(output_dir), "variants": settings}, indent=2))
        return 0

    load_runtime()
    reset_outputs(output_dir)
    metadata: dict[str, object] = {
        "config": str(resolve_path(args.config)),
        "checkpoint": str(resolve_path(args.checkpoint)) if args.checkpoint else None,
        "seed": args.seed,
        "steps": args.steps,
        "log_every": args.log_every,
        "conflict_every": args.conflict_every,
        "track_updates": args.track_updates,
        "optimizer_updates": not args.no_update,
        "variants": [],
    }
    for variant in args.variants:
        print(f"\nRunning {VARIANT_LABELS[variant]} diagnostics", flush=True)
        report = run_variant(args, variant, output_dir)
        metadata["variants"].append(report)
    (output_dir / "diagnostic_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    build_local_gradient_ratios(output_dir)
    build_summary(output_dir)
    if not args.no_plot:
        try:
            plot_diagnostics(output_dir)
        except RuntimeError as exc:
            print(f"Warning: {exc} CSV diagnostics are still complete.", file=sys.stderr)
    print(f"\nGradient diagnostics written to: {output_dir}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        os.chdir(PROJECT_ROOT)
        return run(args)
    except (OSError, RuntimeError, ValueError, NotImplementedError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
