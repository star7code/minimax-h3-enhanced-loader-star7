# SPDX-License-Identifier: Apache-2.0
"""FastH3 modulation kernels adapted to ComfyUI's contiguous segments."""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _rmsnorm_modulate_kernel(
        out_ptr,
        x_ptr,
        weight_ptr,
        scale_ptr,
        shift_ptr,
        n_cols,
        eps,
        stride_row,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        offsets = row * stride_row + cols
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(x * x, axis=0) / n_cols
        normed = x * tl.math.rsqrt(variance + eps)
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(shift_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        output = normed * weight * (1.0 + scale) + shift
        tl.store(out_ptr + offsets, output.to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _residual_gate_rmsnorm_modulate_kernel(
        hidden_ptr,
        out_ptr,
        residual_ptr,
        branch_ptr,
        gate_ptr,
        weight_ptr,
        scale_ptr,
        shift_ptr,
        n_cols,
        eps,
        stride_row,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        offsets = row * stride_row + cols
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        branch = tl.load(branch_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        value = residual + gate * branch
        tl.store(hidden_ptr + offsets, value.to(hidden_ptr.dtype.element_ty), mask=mask)
        variance = tl.sum(value * value, axis=0) / n_cols
        normed = value * tl.math.rsqrt(variance + eps)
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(shift_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        output = normed * weight * (1.0 + scale) + shift
        tl.store(out_ptr + offsets, output.to(out_ptr.dtype.element_ty), mask=mask)


def can_fuse(x, segments):
    return (
        triton is not None
        and x.device.type == "cuda"
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and x.ndim == 2
        and x.stride(-1) == 1
        and all(isinstance(row, int) for _, _, row in segments)
    )


def _launch_config(hidden_size):
    block = triton.next_power_of_2(hidden_size)
    if block >= 8192:
        return block, 16
    if block >= 2048:
        return block, 8
    return block, 4


def rmsnorm_modulate(x, weight, scale, shift, segments, eps):
    output = torch.empty_like(x)
    hidden_size = x.shape[-1]
    block, warps = _launch_config(hidden_size)
    for start, stop, row in segments:
        if start == stop:
            continue
        _rmsnorm_modulate_kernel[(stop - start,)](
            output[start:stop],
            x[start:stop],
            weight,
            scale[row],
            shift[row],
            hidden_size,
            eps,
            x.stride(0),
            BLOCK=block,
            num_warps=warps,
        )
    return output


def residual_gate_rmsnorm_modulate(
    residual, branch, gate, weight, scale, shift, segments, eps
):
    hidden = torch.empty_like(residual)
    output = torch.empty_like(residual)
    hidden_size = residual.shape[-1]
    block, warps = _launch_config(hidden_size)
    for start, stop, row in segments:
        if start == stop:
            continue
        _residual_gate_rmsnorm_modulate_kernel[(stop - start,)](
            hidden[start:stop],
            output[start:stop],
            residual[start:stop],
            branch[start:stop],
            gate[row],
            weight,
            scale[row],
            shift[row],
            hidden_size,
            eps,
            residual.stride(0),
            BLOCK=block,
            num_warps=warps,
        )
    return hidden, output
