# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 VSA tile-64 inference adapted to ComfyUI packed sequences."""

from __future__ import annotations

import functools
import logging
import math
import os
import time
from dataclasses import dataclass

import torch

from .vendor.vsa_triton import triton_block_sparse_attn_forward


_LOG = logging.getLogger("Star7FastH3VSA")
_VERBOSE = os.environ.get("STAR7_VSA_VERBOSE", "0").strip().lower() in {
    "1", "true", "yes", "on"
}


TILE_SHAPE = (4, 4, 4)
TILE_ELEMS = math.prod(TILE_SHAPE)


@dataclass(frozen=True)
class VSAGeometry:
    variable_block_sizes: torch.Tensor
    packed_to_tiled: torch.Tensor
    padding_indices: torch.Tensor
    num_prefix_tiles: int
    num_video_tiles: int
    total_length: int


def _canonical_device(device: torch.device) -> torch.device:
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


@functools.lru_cache(maxsize=12)
def _video_tile_indices(
    shape: tuple[int, int, int], device: torch.device
) -> torch.Tensor:
    t, h, w = shape
    tt, th, tw = TILE_SHAPE
    raster = torch.arange(t * h * w, device=device, dtype=torch.long).reshape(t, h, w)
    tiles = []
    for ti in range(math.ceil(t / tt)):
        for hi in range(math.ceil(h / th)):
            for wi in range(math.ceil(w / tw)):
                tiles.append(
                    raster[
                        ti * tt:min((ti + 1) * tt, t),
                        hi * th:min((hi + 1) * th, h),
                        wi * tw:min((wi + 1) * tw, w),
                    ].flatten()
                )
    return torch.cat(tiles)


@functools.lru_cache(maxsize=12)
def _geometry_cached(
    prefix_segments: tuple[int, ...],
    video_shape: tuple[int, int, int],
    device: torch.device,
) -> VSAGeometry:
    device = _canonical_device(device)
    prefix_segments = tuple(int(n) for n in prefix_segments if int(n) > 0)
    prefix_length = sum(prefix_segments)
    prefix_sizes = []
    for segment in prefix_segments:
        full, remainder = divmod(segment, TILE_ELEMS)
        prefix_sizes.extend([TILE_ELEMS] * full)
        if remainder:
            prefix_sizes.append(remainder)

    t, h, w = video_shape
    tt, th, tw = TILE_SHAPE
    axis_sizes = []
    for length, tile in zip((t, h, w), TILE_SHAPE):
        count = math.ceil(length / tile)
        sizes = [tile] * count
        sizes[-1] = length - (count - 1) * tile
        axis_sizes.append(sizes)
    video_sizes = [
        ts * hs * ws
        for ts in axis_sizes[0]
        for hs in axis_sizes[1]
        for ws in axis_sizes[2]
    ]
    sizes = torch.tensor(prefix_sizes + video_sizes, dtype=torch.long, device=device)
    starts = torch.arange(sizes.numel(), device=device) * TILE_ELEMS
    offsets = torch.arange(TILE_ELEMS, device=device)[None, :]
    non_pad = (starts[:, None] + offsets)[offsets < sizes[:, None]]
    tile_order = torch.cat(
        [
            torch.arange(prefix_length, device=device, dtype=torch.long),
            _video_tile_indices(video_shape, device) + prefix_length,
        ]
    )
    packed_to_tiled = non_pad[torch.argsort(tile_order)]
    padding_mask = torch.ones(
        sizes.numel() * TILE_ELEMS, dtype=torch.bool, device=device
    )
    padding_mask[packed_to_tiled] = False
    padding_indices = padding_mask.nonzero(as_tuple=False).flatten()
    total = prefix_length + t * h * w
    if int(sizes.sum()) != total or packed_to_tiled.numel() != total:
        raise ValueError(
            f"Invalid VSA geometry: prefix={prefix_segments}, video={video_shape}, "
            f"sizes={int(sizes.sum())}, total={total}"
        )
    return VSAGeometry(
        variable_block_sizes=sizes,
        packed_to_tiled=packed_to_tiled,
        padding_indices=padding_indices,
        num_prefix_tiles=len(prefix_sizes),
        num_video_tiles=len(video_sizes),
        total_length=total,
    )


def build_geometry(
    prefix_segments: tuple[int, ...],
    video_shape: tuple[int, int, int],
    device: torch.device,
) -> VSAGeometry:
    return _geometry_cached(
        tuple(prefix_segments), tuple(video_shape), _canonical_device(device)
    )


def _tile(x: torch.Tensor, geometry: VSAGeometry) -> torch.Tensor:
    # BHSD -> padded BHSD in tile order.
    target = torch.empty(
        x.shape[0], x.shape[1],
        geometry.variable_block_sizes.numel() * TILE_ELEMS,
        x.shape[-1],
        dtype=x.dtype,
        device=x.device,
    )
    target[:, :, geometry.packed_to_tiled] = x
    if geometry.padding_indices.numel():
        target[:, :, geometry.padding_indices] = 0
    return target


def _pool_tiles(x: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
    # BHSD -> BHND, with fp32 accumulation and exact boundary normalization.
    b, h, sequence, d = x.shape
    count = sequence // TILE_ELEMS
    pooled = x.view(b, h, count, TILE_ELEMS, d).sum(dim=3, dtype=torch.float32)
    return pooled / sizes.view(1, 1, -1, 1)


def _route_lut(scores: torch.Tensor, geometry: VSAGeometry, sparsity: float):
    """Build the kernel LUT directly, avoiding a dense bool mask and full sort."""
    n_tiles = scores.shape[-1]
    keep_video = max(
        1,
        min(math.ceil((1.0 - float(sparsity)) * geometry.num_video_tiles),
            geometry.num_video_tiles),
    )
    if keep_video == geometry.num_video_tiles:
        lut = torch.arange(n_tiles, device=scores.device, dtype=torch.int32)
        lut = lut.view(1, 1, 1, n_tiles).expand(*scores.shape[:-1], n_tiles)
        counts = torch.full(
            scores.shape[:-1], n_tiles, dtype=torch.int32, device=scores.device
        )
        return lut.contiguous(), counts

    start = geometry.num_prefix_tiles
    selected = scores[..., start:].topk(keep_video, dim=-1).indices + start
    # The old mask-to-index route emitted ascending key ids. Sorting only the
    # selected 10% keeps the same online-softmax traversal without sorting the
    # complete dense mask.
    selected = selected.sort(dim=-1).values.to(torch.int32)
    sparse_count = start + keep_video
    max_slots = n_tiles if start else sparse_count
    lut = torch.empty(
        *scores.shape[:-1], max_slots, dtype=torch.int32, device=scores.device
    )
    if start:
        prefix = torch.arange(start, device=scores.device, dtype=torch.int32)
        lut[..., :start] = prefix
    lut[..., start:sparse_count] = selected
    counts = torch.full(
        scores.shape[:-1], sparse_count, dtype=torch.int32, device=scores.device
    )
    # Text, conditions and audio queries stay fully dense.
    if start:
        all_keys = torch.arange(n_tiles, device=scores.device, dtype=torch.int32)
        lut[:, :, :start, :] = all_keys
        counts[:, :, :start] = n_tiles
    return lut.contiguous(), counts.contiguous()


def native_sm75_status() -> tuple[bool, str]:
    """Report whether the native VSA launcher is ready without importing Triton."""
    try:
        from . import vsa_sm75_native
        return vsa_sm75_native.availability()
    except Exception as exc:
        return False, str(exc)


@torch.inference_mode()
def sparse_attention_consume(
    owned_qkvg: list[torch.Tensor],
    *,
    prefix_segments: tuple[int, ...],
    video_shape: tuple[int, int, int],
    sparsity: float = 0.9,
    profile: bool = False,
) -> torch.Tensor:
    """Consume Q/K/V/gate tensors and return BHSD in original packed order."""
    if len(owned_qkvg) != 4:
        raise ValueError("FastH3 VSA expects owned [Q, K, V, gate] tensors")
    q, k, v, gate = owned_qkvg
    owned_qkvg.clear()
    if not (q.shape == k.shape == v.shape == gate.shape):
        raise ValueError(
            f"FastH3 VSA Q/K/V/gate shapes differ: "
            f"{q.shape}, {k.shape}, {v.shape}, {gate.shape}"
        )
    geometry = build_geometry(prefix_segments, video_shape, q.device)
    if q.shape[2] != geometry.total_length:
        raise ValueError(
            f"FastH3 VSA sequence mismatch: attention={q.shape[2]}, "
            f"geometry={geometry.total_length}, prefix={prefix_segments}, "
            f"video={video_shape}"
        )

    profile_marks = []
    if profile:
        torch.cuda.synchronize(q.device)
        profile_marks.append(("start", time.perf_counter()))
    q = _tile(q, geometry)
    k = _tile(k, geometry)
    v = _tile(v, geometry)
    gate = _tile(gate, geometry)
    if profile:
        torch.cuda.synchronize(q.device)
        profile_marks.append(("tile", time.perf_counter()))
    q_pool = _pool_tiles(q, geometry.variable_block_sizes)
    k_pool = _pool_tiles(k, geometry.variable_block_sizes)
    scores = torch.matmul(q_pool, k_pool.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if profile:
        torch.cuda.synchronize(q.device)
        profile_marks.append(("pool+scores", time.perf_counter()))
    q2k_idx, q2k_num = _route_lut(scores, geometry, sparsity)
    if profile:
        torch.cuda.synchronize(q.device)
        profile_marks.append(("topk+lut", time.perf_counter()))
    native_sm75 = False
    is_sm75 = (
        q.device.type == "cuda"
        and torch.cuda.get_device_capability(q.device) == (7, 5)
    )
    if is_sm75:
        try:
            from . import vsa_sm75_native
            available, reason = vsa_sm75_native.availability()
        except Exception as exc:
            available, reason = False, str(exc)
        if available:
            output = vsa_sm75_native.launch_exact(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                q2k_num.to(torch.int32).contiguous(),
                q2k_idx.to(torch.int32).contiguous(),
                geometry.variable_block_sizes.to(torch.int32).contiguous(),
            )
            native_sm75 = True
            if _VERBOSE:
                _LOG.info(
                    "[Star7 FastH3 VSA] native SM75 exact path | tiles=%d | "
                    "lut=%s | gate-compressed branch=FP16",
                    int(geometry.variable_block_sizes.numel()),
                    tuple(q2k_idx.shape),
                )
        else:
            raise RuntimeError(
                "FastH3 VSA native SM75 path is unavailable: "
                f"{reason}. The Triton compatibility path is intentionally not "
                "used for this full-size run because it can take many minutes "
                "before the first sampling step."
            )
    else:
        output, _lse = triton_block_sparse_attn_forward(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            q2k_idx.to(torch.int32).contiguous(),
            q2k_num.to(torch.int32).contiguous(),
            geometry.variable_block_sizes.to(torch.int32).contiguous(),
        )
    if profile:
        torch.cuda.synchronize(q.device)
        profile_marks.append(("selected-blocks", time.perf_counter()))

    v_pool = _pool_tiles(v, geometry.variable_block_sizes)
    compressed = torch.matmul(torch.softmax(scores, dim=-1), v_pool)
    b, h, sequence, d = output.shape
    output = (
        output.view(b, h, -1, TILE_ELEMS, d)
        + compressed.unsqueeze(3).to(output.dtype)
        * gate.view(b, h, -1, TILE_ELEMS, d)
    ).view(b, h, sequence, d)
    result = output[:, :, geometry.packed_to_tiled]
    if profile:
        torch.cuda.synchronize(q.device)
        profile_marks.append(("compressed+untile", time.perf_counter()))
        durations = {
            name: (stamp - profile_marks[index - 1][1]) * 1000.0
            for index, (name, stamp) in enumerate(profile_marks)
            if index
        }
        _LOG.info(
            "[Star7 FastH3 VSA profile] tiles=%d | %s",
            int(geometry.variable_block_sizes.numel()),
            " | ".join(f"{name}={value:.2f}ms" for name, value in durations.items()),
        )
    del native_sm75
    return result
