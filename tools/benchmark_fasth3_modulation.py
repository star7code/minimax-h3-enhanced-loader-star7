from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import statistics
import sys

import torch


def _load_fusion():
    path = Path(__file__).resolve().parents[1] / "fasth3_modulation.py"
    spec = importlib.util.spec_from_file_location("star7_fasth3_modulation_bench", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _median_ms(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def _error(reference, actual):
    delta = actual.float() - reference.float()
    relative_rmse = delta.square().mean().sqrt() / reference.float().square().mean().sqrt().clamp_min(1e-12)
    return float(delta.abs().max()), float(relative_rmse)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=8773)
    parser.add_argument("--hidden", type=int, default=5376)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    fusion = _load_fusion()

    x = torch.randn(args.sequence, args.hidden, device="cuda", dtype=torch.float32)
    branch = torch.randn_like(x)
    weight = torch.randn(args.hidden, device="cuda", dtype=torch.float32)
    scale = torch.randn(4, args.hidden, device="cuda", dtype=torch.float16)
    shift = torch.randn_like(scale)
    gate = torch.randn_like(scale)
    cuts = [0, min(128, args.sequence), min(512, args.sequence), min(1024, args.sequence), args.sequence]
    segments = [(cuts[i], cuts[i + 1], i) for i in range(4) if cuts[i] < cuts[i + 1]]
    eps = 1e-5

    def eager_norm():
        output = torch.nn.functional.rms_norm(x, (args.hidden,), weight, eps)
        for start, stop, row in segments:
            output[start:stop].mul_(1.0 + scale[row].float()).add_(shift[row].float())
        return output

    def fused_norm():
        return fusion.rmsnorm_modulate(x, weight, scale, shift, segments, eps)

    def eager_residual():
        hidden = x.clone()
        for start, stop, row in segments:
            hidden[start:stop].addcmul_(branch[start:stop], gate[row].float())
        output = torch.nn.functional.rms_norm(hidden, (args.hidden,), weight, eps)
        for start, stop, row in segments:
            output[start:stop].mul_(1.0 + scale[row].float()).add_(shift[row].float())
        return hidden, output

    def fused_residual():
        return fusion.residual_gate_rmsnorm_modulate(
            x, branch, gate, weight, scale, shift, segments, eps
        )

    with torch.inference_mode():
        norm_reference = eager_norm()
        norm_actual = fused_norm()
        residual_reference = eager_residual()
        residual_actual = fused_residual()
        norm_error = _error(norm_reference, norm_actual)
        hidden_error = _error(residual_reference[0], residual_actual[0])
        residual_error = _error(residual_reference[1], residual_actual[1])
        eager_norm_ms = _median_ms(eager_norm, args.warmup, args.repeats)
        fused_norm_ms = _median_ms(fused_norm, args.warmup, args.repeats)
        eager_residual_ms = _median_ms(eager_residual, args.warmup, args.repeats)
        fused_residual_ms = _median_ms(fused_residual, args.warmup, args.repeats)

    print(f"device={torch.cuda.get_device_name()} S={args.sequence} H={args.hidden}")
    print(
        f"rmsnorm+modulate eager={eager_norm_ms:.3f}ms fused={fused_norm_ms:.3f}ms "
        f"speedup={eager_norm_ms / fused_norm_ms:.2f}x error={norm_error[0]:.3g}/{norm_error[1]:.3g}"
    )
    print(
        f"residual+gate+rmsnorm+modulate eager={eager_residual_ms:.3f}ms "
        f"fused={fused_residual_ms:.3f}ms speedup={eager_residual_ms / fused_residual_ms:.2f}x "
        f"hidden_error={hidden_error[0]:.3g}/{hidden_error[1]:.3g} "
        f"output_error={residual_error[0]:.3g}/{residual_error[1]:.3g}"
    )


if __name__ == "__main__":
    main()
