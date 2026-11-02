#!/usr/bin/env python3
"""Measure CMeRT and SACG GFLOPs, latency, FPS, and parameter count.

The benchmark uses synthetic tensors with the exact THUMOS'14 ResNet protocol
shapes, so it does not require the dataset or checkpoints. Feature extraction,
data loading, and metric aggregation are intentionally excluded.

GFLOPs follow the MAC-style convention used by many vision papers: one fused
multiply-add is counted as one operation. PyTorch profiler reports two FLOPs
per multiply-add, so its supported-operator total is divided by two.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import platform
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from rekognition_online_action_detection.config.defaults import get_cfg
    from rekognition_online_action_detection.models import build_model
    from rekognition_online_action_detection.models.feature_head import FEATURE_SIZES
except Exception as exc:
    raise RuntimeError(
        "Cannot import CMeRT. Put this script in the CMeRT-main root and run it "
        "from that directory, using the same conda environment as main.py. "
        f"Original import error: {exc}"
    ) from exc


VARIANTS = ("CMeRT", "SACG")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark CMeRT and SACG on THUMOS'14 ResNet input shapes."
    )
    parser.add_argument(
        "--config",
        default="configs/THUMOS/cmert_long256_work4_kinetics_1x.yaml",
        help="THUMOS'14 YAML config, relative to the project root by default.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=22)
    parser.add_argument(
        "--allow-tf32",
        action="store_true",
        help="Allow TF32 matrix multiplication. The default is strict FP32.",
    )
    parser.add_argument(
        "--output-csv", default="th14_efficiency_results.csv"
    )
    parser.add_argument(
        "--output-json", default="th14_efficiency_results.json"
    )
    return parser.parse_args()


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def infer_temporal_fields(cfg, fps: int) -> None:
    lstr = cfg.MODEL.LSTR
    cfg.DATA.FPS = int(fps)
    lstr.LONG_MEMORY_LENGTH = int(lstr.LONG_MEMORY_SECONDS * fps)
    lstr.WORK_MEMORY_LENGTH = int(lstr.WORK_MEMORY_SECONDS * fps)
    lstr.TOTAL_MEMORY_LENGTH = (
        lstr.LONG_MEMORY_LENGTH + lstr.WORK_MEMORY_LENGTH
    )
    lstr.ANTICIPATION_LENGTH = int(lstr.ANTICIPATION_SECONDS * fps)
    lstr.FUTURE_LENGTH = int(lstr.FUTURE_SECONDS * fps)

    divisibility_checks = (
        ("LONG_MEMORY_LENGTH", lstr.LONG_MEMORY_LENGTH, lstr.LONG_MEMORY_SAMPLE_RATE),
        ("WORK_MEMORY_LENGTH", lstr.WORK_MEMORY_LENGTH, lstr.WORK_MEMORY_SAMPLE_RATE),
        ("ANTICIPATION_LENGTH", lstr.ANTICIPATION_LENGTH, lstr.ANTICIPATION_SAMPLE_RATE),
        ("FUTURE_LENGTH", lstr.FUTURE_LENGTH, lstr.FUTURE_SAMPLE_RATE),
    )
    for name, length, rate in divisibility_checks:
        if length % rate != 0:
            raise ValueError(f"{name}={length} is not divisible by sample rate {rate}.")

    lstr.LONG_MEMORY_NUM_SAMPLES = (
        lstr.LONG_MEMORY_LENGTH // lstr.LONG_MEMORY_SAMPLE_RATE
    )
    lstr.WORK_MEMORY_NUM_SAMPLES = (
        lstr.WORK_MEMORY_LENGTH // lstr.WORK_MEMORY_SAMPLE_RATE
    )
    lstr.TOTAL_MEMORY_NUM_SAMPLES = (
        lstr.LONG_MEMORY_NUM_SAMPLES + lstr.WORK_MEMORY_NUM_SAMPLES
    )
    lstr.ANTICIPATION_NUM_SAMPLES = (
        lstr.ANTICIPATION_LENGTH // lstr.ANTICIPATION_SAMPLE_RATE
    )
    lstr.FUTURE_NUM_SAMPLES = lstr.FUTURE_LENGTH // lstr.FUTURE_SAMPLE_RATE


def build_variant_cfg(args: argparse.Namespace, method: str):
    config_path = resolve_project_path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(str(config_path))
    cfg.DATA.NUM_CLASSES = int(args.num_classes)
    if cfg.INPUT.MODALITY == "twostream":
        cfg.INPUT.MODALITY = "visual+motion"
    infer_temporal_fields(cfg, args.fps)

    lstr = cfg.MODEL.LSTR
    lstr.INFERENCE_MODE = "batch"
    lstr.GEN_CROSS_ASSA_INIT_W = [0.0, 0.0]
    lstr.GEN_CONTRANORM_SCALE = 0.05
    lstr.GEN_CONTRANORM_TEMP = 1.0

    if method == "CMeRT":
        lstr.GEN_CROSS_ATTN_TYPE = "dotproduct"
        lstr.GEN_CONTRANORM = False
        lstr.GEN_CONTRANORM_POS = "none"
    elif method == "SACG":
        lstr.GEN_CROSS_ATTN_TYPE = "assa"
        lstr.GEN_CONTRANORM = True
        lstr.GEN_CONTRANORM_POS = "cross"
    else:
        raise ValueError(f"Unknown method: {method}")

    cfg.TEST = 1
    return cfg


def make_inputs(cfg, device: torch.device) -> Tuple[torch.Tensor, ...]:
    lstr = cfg.MODEL.LSTR
    context_samples = (
        int(lstr.CONTEXT_SECONDS * cfg.DATA.FPS)
        // int(lstr.CONTEXT_MEMORY_SAMPLE_RATE)
    )
    input_samples = (
        int(lstr.LONG_MEMORY_NUM_SAMPLES)
        + context_samples
        + int(lstr.WORK_MEMORY_NUM_SAMPLES)
    )
    visual_dim = FEATURE_SIZES[cfg.INPUT.VISUAL_FEATURE]
    motion_dim = FEATURE_SIZES[cfg.INPUT.MOTION_FEATURE]

    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    visual = torch.randn(
        1, input_samples, visual_dim, generator=generator,
        dtype=torch.float32, device=device,
    )
    motion = torch.randn(
        1, input_samples, motion_dim, generator=generator,
        dtype=torch.float32, device=device,
    )
    memory_mask = torch.zeros(
        1, int(lstr.LONG_MEMORY_NUM_SAMPLES),
        dtype=torch.float32, device=device,
    )
    return visual, motion, memory_mask


def profile_gflops(
    model: torch.nn.Module,
    inputs: Tuple[torch.Tensor, ...],
    device: torch.device,
) -> Tuple[float, float, Dict[str, float]]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.inference_mode():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
            with_flops=True,
        ) as profiler:
            model(*inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    by_operator_raw: Dict[str, float] = {}
    raw_flops = 0.0
    for event in profiler.key_averages():
        event_flops = float(event.flops or 0.0)
        if event_flops > 0.0:
            raw_flops += event_flops
            by_operator_raw[event.key] = event_flops

    if raw_flops <= 0.0:
        raise RuntimeError(
            "PyTorch profiler returned zero FLOPs. Check that this PyTorch build "
            "supports with_flops=True."
        )

    mac_style_flops = raw_flops / 2.0
    by_operator_gflops = {
        name: value / 2.0 / 1e9
        for name, value in sorted(
            by_operator_raw.items(), key=lambda item: item[1], reverse=True
        )
    }
    return mac_style_flops / 1e9, raw_flops, by_operator_gflops


def benchmark_cuda(
    model: torch.nn.Module,
    inputs: Tuple[torch.Tensor, ...],
    device: torch.device,
    warmup: int,
    iterations: int,
    trials: int,
) -> Tuple[List[float], List[float]]:
    if device.type != "cuda":
        raise RuntimeError("FPS must be measured on CUDA for this experiment.")

    with torch.inference_mode():
        for _ in range(warmup):
            model(*inputs)
        torch.cuda.synchronize(device)

        latency_ms_trials: List[float] = []
        fps_trials: List[float] = []
        for trial in range(trials):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                model(*inputs)
            end.record()
            torch.cuda.synchronize(device)

            elapsed_ms = float(start.elapsed_time(end))
            latency_ms = elapsed_ms / iterations
            measured_fps = 1000.0 / latency_ms
            latency_ms_trials.append(latency_ms)
            fps_trials.append(measured_fps)
            print(
                f"  trial {trial + 1}/{trials}: "
                f"latency={latency_ms:.4f} ms, FPS={measured_fps:.2f}"
            )

    return latency_ms_trials, fps_trials


def mean_and_std(values: List[float]) -> Tuple[float, float]:
    return statistics.fmean(values), statistics.pstdev(values)


def benchmark_variant(
    args: argparse.Namespace,
    method: str,
    device: torch.device,
) -> Dict[str, Any]:
    print(f"\n{'=' * 64}\nBenchmarking {method}\n{'=' * 64}")
    cfg = build_variant_cfg(args, method)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = build_model(cfg, device=device)
    model.eval()
    inputs = make_inputs(cfg, device)

    with torch.inference_mode():
        output = model(*inputs)
    if output is None:
        raise RuntimeError(f"{method} returned no output.")
    torch.cuda.synchronize(device)

    params = sum(parameter.numel() for parameter in model.parameters())
    gflops, profiler_raw_flops, by_operator = profile_gflops(
        model, inputs, device
    )
    latency_trials, fps_trials = benchmark_cuda(
        model, inputs, device, args.warmup, args.iterations, args.trials
    )
    latency_mean, latency_std = mean_and_std(latency_trials)
    fps_mean, fps_std = mean_and_std(fps_trials)

    lstr = cfg.MODEL.LSTR
    context_samples = (
        int(lstr.CONTEXT_SECONDS * cfg.DATA.FPS)
        // int(lstr.CONTEXT_MEMORY_SAMPLE_RATE)
    )
    result: Dict[str, Any] = {
        "method": method,
        "params": int(params),
        "params_m": params / 1e6,
        "gflops": gflops,
        "latency_ms_mean": latency_mean,
        "latency_ms_std": latency_std,
        "fps_mean": fps_mean,
        "fps_std": fps_std,
        "latency_ms_trials": latency_trials,
        "fps_trials": fps_trials,
        "profiler_raw_flops_fma2": profiler_raw_flops,
        "gflops_by_operator": by_operator,
        "input_shape": {
            "batch_size": 1,
            "long_term_tokens": int(lstr.LONG_MEMORY_NUM_SAMPLES),
            "near_past_tokens": context_samples,
            "short_term_tokens": int(lstr.WORK_MEMORY_NUM_SAMPLES),
            "total_input_tokens": int(inputs[0].shape[1]),
            "visual_dimension": int(inputs[0].shape[2]),
            "motion_dimension": int(inputs[1].shape[2]),
        },
        "model_options": {
            "gen_cross_attention": str(lstr.GEN_CROSS_ATTN_TYPE),
            "gen_cross_assa_init_w": list(lstr.GEN_CROSS_ASSA_INIT_W),
            "gen_contranorm": bool(lstr.GEN_CONTRANORM),
            "gen_contranorm_position": str(lstr.GEN_CONTRANORM_POS),
            "gen_contranorm_scale": float(lstr.GEN_CONTRANORM_SCALE),
            "gen_contranorm_temperature": float(lstr.GEN_CONTRANORM_TEMP),
        },
    }

    del output, inputs, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def write_outputs(
    args: argparse.Namespace,
    results: List[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> Tuple[Path, Path]:
    csv_path = resolve_project_path(args.output_csv)
    json_path = resolve_project_path(args.output_json)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    columns = (
        "method",
        "params_m",
        "gflops",
        "latency_ms_mean",
        "latency_ms_std",
        "fps_mean",
        "fps_std",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    key: result[key] if key == "method" else f"{result[key]:.6f}"
                    for key in columns
                }
            )

    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(
            {"metadata": metadata, "results": results},
            stream,
            indent=2,
            ensure_ascii=True,
        )
        stream.write("\n")
    return csv_path, json_path


def main() -> None:
    args = parse_args()
    if args.warmup < 1 or args.iterations < 1 or args.trials < 1:
        raise ValueError("warmup, iterations, and trials must all be positive.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run this benchmark on the RTX 4090.")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if args.allow_tf32 else "highest")

    results = [benchmark_variant(args, method, device) for method in VARIANTS]
    baseline, sacg = results
    metadata: Dict[str, Any] = {
        "protocol": "THUMOS'14 ResNet temporal-model-only forward pass",
        "gflops_convention": (
            "PyTorch supported operator FLOPs divided by two; one "
            "multiply-add is counted as one MAC-style FLOP"
        ),
        "prediction_unit": "one online prediction per batch-size-1 forward pass",
        "feature_extraction_included": False,
        "data_loading_included": False,
        "precision": "FP32 with TF32 allowed" if args.allow_tf32 else "strict FP32",
        "warmup_iterations": args.warmup,
        "timed_iterations_per_trial": args.iterations,
        "trials": args.trials,
        "seed": args.seed,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "config": str(resolve_project_path(args.config)),
    }

    csv_path, json_path = write_outputs(args, results, metadata)
    gflops_delta = sacg["gflops"] - baseline["gflops"]
    gflops_pct = gflops_delta / baseline["gflops"] * 100.0
    fps_delta = sacg["fps_mean"] - baseline["fps_mean"]
    fps_pct = fps_delta / baseline["fps_mean"] * 100.0

    print("\nFinal paper-ready values")
    print("Method   Params(M)   GFLOPs   FPS(mean +/- std)   Latency(ms)")
    for result in results:
        print(
            f"{result['method']:<8} "
            f"{result['params_m']:>9.3f} "
            f"{result['gflops']:>8.3f} "
            f"{result['fps_mean']:>9.2f} +/- {result['fps_std']:.2f} "
            f"{result['latency_ms_mean']:>10.4f}"
        )
    print(
        f"SACG delta: GFLOPs {gflops_delta:+.4f} ({gflops_pct:+.3f}%), "
        f"FPS {fps_delta:+.2f} ({fps_pct:+.3f}%)."
    )
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
