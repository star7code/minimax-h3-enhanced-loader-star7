"""Native SM75 exact-block launcher for FastVideo VSA.

The CUDA entry point only replaces VSA's selected-block softmax.  VSA's
gate-compressed pooled branch remains in :mod:`fasth3_vsa`, so this module does
not turn the checkpoint into the different Sol-Attn approximation.
"""

from __future__ import annotations

import ctypes
import os
import platform
from pathlib import Path

import torch


_LIBRARY = None
_LOAD_ERROR: Exception | None = None


def _library_path() -> Path:
    if platform.system() != "Windows" or platform.machine().lower() not in {
        "amd64", "x86_64",
    }:
        raise RuntimeError("native FastVideo VSA SM75 is currently Windows-only")
    loader_root = Path(__file__).resolve().parent
    candidates = (
        # The enhanced loader is independently installable.  Its VSA runtime
        # must never depend on a sibling chunk-node installation or version.
        loader_root / "bin" / "win_amd64" / "star7_vsa_sm75_v2.dll",
        loader_root / "bin" / "win_amd64" / "star7_vsa_sm75_v1.dll",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "native FastVideo VSA SM75 binary is missing; checked: "
        + ", ".join(str(path) for path in candidates)
    )


def _load():
    global _LIBRARY, _LOAD_ERROR
    if _LIBRARY is not None:
        return _LIBRARY
    if _LOAD_ERROR is not None:
        raise RuntimeError(f"native FastVideo VSA SM75 failed to load: {_LOAD_ERROR}")
    try:
        path = _library_path()
        library = ctypes.CDLL(str(path))
        launch = library.star7_vsa_sm75_launch_exact
        launch.argtypes = [
            *([ctypes.c_uint64] * 9),
            *([ctypes.c_int] * 4),
            ctypes.c_float,
            ctypes.c_uint64,
        ]
        launch.restype = ctypes.c_int
        launch_all_int8 = library.star7_vsa_sm75_launch_all_int8
        launch_all_int8.argtypes = [
            *([ctypes.c_uint64] * 10),
            *([ctypes.c_int] * 5), ctypes.c_float, ctypes.c_uint64,
        ]
        launch_all_int8.restype = ctypes.c_int
        quantize = library.star7_sol_sm75_quantize
        quantize.argtypes = [
            *([ctypes.c_uint64] * 3),
            *([ctypes.c_int] * 4), ctypes.c_uint64,
        ]
        quantize.restype = ctypes.c_int
        quantize_v = library.star7_sla_sm75_quant_v_int8
        quantize_v.argtypes = [
            *([ctypes.c_uint64] * 3),
            *([ctypes.c_int] * 5),
            *([ctypes.c_int64] * 3),
            ctypes.c_int, ctypes.c_uint64,
        ]
        quantize_v.restype = ctypes.c_int
        _LIBRARY = library
        return library
    except Exception as exc:
        _LOAD_ERROR = exc
        raise


def availability() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "NVIDIA CUDA is unavailable"
    if torch.cuda.get_device_capability() != (7, 5):
        return False, "native FastVideo VSA path requires exactly SM75"
    try:
        _load()
    except Exception as exc:
        return False, str(exc)
    mode = "All-INT8" if _all_int8_enabled() else "QK-INT8/PV-FP16"
    return True, f"native SM75 Q64/K64 VSA kernel ({mode})"


def _all_int8_enabled() -> bool:
    value = os.environ.get("STAR7_VSA_SM75_ALL_INT8", "1").strip().lower()
    return value not in {"0", "false", "off", "no"}


def _quantize(value: torch.Tensor, block: int) -> tuple[torch.Tensor, torch.Tensor]:
    if value.dtype != torch.float16 or value.ndim != 4 or not value.is_cuda:
        raise TypeError("native VSA quantization requires CUDA FP16 [B,H,T,D]")
    if value.shape[-1] != 128 or not value.is_contiguous():
        raise ValueError("native VSA quantization requires contiguous head_dim=128")
    if block not in (16, 64):
        raise ValueError("native VSA quantization block must be 16 or 64")
    batch, heads, length, _ = value.shape
    output = torch.empty_like(value, dtype=torch.int8)
    scale = torch.empty(
        (batch, heads, (length + block - 1) // block),
        dtype=torch.float32,
        device=value.device,
    )
    stream = torch.cuda.current_stream(value.device).cuda_stream
    code = int(_load().star7_sol_sm75_quantize(
        value.data_ptr(), output.data_ptr(), scale.data_ptr(),
        batch, heads, length, block, stream,
    ))
    if code:
        raise RuntimeError(f"native VSA INT8 quantization failed with cudaError={code}")
    return output, scale


def _quantize_v(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    batch, heads, length, head_dim = value.shape
    padded_length = ((length + 63) // 64) * 64
    output = torch.empty(
        (batch, heads, head_dim, padded_length),
        dtype=torch.int8,
        device=value.device,
    )
    scale = torch.empty(
        (batch, heads, head_dim), dtype=torch.float32, device=value.device
    )
    stream = torch.cuda.current_stream(value.device).cuda_stream
    code = int(_load().star7_sla_sm75_quant_v_int8(
        value.data_ptr(), output.data_ptr(), scale.data_ptr(),
        batch, heads, length, head_dim, padded_length,
        value.stride(0), value.stride(1), value.stride(2), 1, stream,
    ))
    if code:
        raise RuntimeError(f"native VSA V quantization failed with cudaError={code}")
    return output, scale, padded_length


@torch.inference_mode()
def launch_exact(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    row_count: torch.Tensor,
    lut: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> torch.Tensor:
    """Run the selected VSA blocks and leave compression to the caller."""
    if not (q.shape == k.shape == v.shape):
        raise ValueError("native VSA Q/K/V shapes must match")
    if q.ndim != 4 or q.shape[-1] != 128:
        raise ValueError("native VSA expects [B,H,T,128] Q/K/V")
    if not all(x.is_cuda and x.is_contiguous() for x in (q, k, v)):
        raise ValueError("native VSA expects contiguous CUDA tensors")
    if q.dtype != torch.float16 or k.dtype != torch.float16 or v.dtype != torch.float16:
        raise TypeError("native VSA currently expects FP16 Q/K/V")
    batch, heads, length, _ = q.shape
    key_blocks = (length + 63) // 64
    if row_count.shape != (batch, heads, key_blocks):
        raise ValueError("native VSA row_count shape does not match Q/K/V")
    if lut.ndim != 4 or lut.shape[:3] != row_count.shape:
        raise ValueError("native VSA LUT shape does not match row_count")
    lut_stride = int(lut.shape[-1])
    if lut_stride <= 0 or lut_stride > key_blocks:
        raise ValueError("native VSA LUT stride is invalid")
    if variable_block_sizes.shape != (key_blocks,):
        raise ValueError("native VSA block-size table does not match Q/K/V")
    q_int8, q_scale = _quantize(q, 16)
    k_int8, k_scale = _quantize(k, 64)
    output = torch.empty_like(v)
    stream = torch.cuda.current_stream(q.device).cuda_stream
    if _all_int8_enabled():
        v_int8, v_scale, padded_length = _quantize_v(v)
        code = int(_load().star7_vsa_sm75_launch_all_int8(
            q_int8.data_ptr(), k_int8.data_ptr(), v_int8.data_ptr(),
            q_scale.data_ptr(), k_scale.data_ptr(), v_scale.data_ptr(),
            row_count.data_ptr(), lut.data_ptr(),
            variable_block_sizes.data_ptr(), output.data_ptr(),
            batch, heads, length, lut_stride, padded_length,
            128 ** -0.5, stream,
        ))
    else:
        code = int(_load().star7_vsa_sm75_launch_exact(
            q_int8.data_ptr(), k_int8.data_ptr(), v.data_ptr(),
            q_scale.data_ptr(), k_scale.data_ptr(), row_count.data_ptr(),
            lut.data_ptr(), variable_block_sizes.data_ptr(), output.data_ptr(),
            batch, heads, length, lut_stride, 128 ** -0.5, stream,
        ))
    if code:
        raise RuntimeError(f"native FastVideo VSA SM75 launch failed with cudaError={code}")
    return output
