#!/usr/bin/env python3
"""Merge an indexed FastH3 safetensors directory without loading tensors.

The output is written to a sibling temporary file and atomically renamed only
after its header, tensor count and final byte size have been validated.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path

from safetensors import safe_open


INDEX_NAMES = (
    "model.safetensors.index.json",
    "diffusion_pytorch_model.safetensors.index.json",
)
COPY_CHUNK = 16 * 1024 * 1024


def _read_header(path: Path) -> tuple[int, dict]:
    with path.open("rb") as handle:
        length_raw = handle.read(8)
        if len(length_raw) != 8:
            raise ValueError(f"Invalid safetensors header: {path}")
        header_length = struct.unpack("<Q", length_raw)[0]
        header = json.loads(handle.read(header_length).decode("utf-8"))
    return 8 + header_length, header


def _metadata(source_root: Path) -> dict[str, str]:
    manifest_path = source_root / "star7_fasth3_vsa_int8.json"
    if not manifest_path.is_file():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "format": "pt",
        "star7_fasth3_manifest": json.dumps(
            manifest, ensure_ascii=False, separators=(",", ":")
        ),
        "star7_variant": str(manifest.get("variant", "")),
        "star7_sampling_profile": str(manifest.get("sampling_profile", "")),
        "star7_quantization": str(manifest.get("quantization", "")),
    }


def merge(source_root: Path, output: Path) -> None:
    transformer = source_root / "transformer"
    index_path = next(
        (transformer / name for name in INDEX_NAMES if (transformer / name).is_file()),
        None,
    )
    if index_path is None:
        raise FileNotFoundError(f"No safetensors index found in {transformer}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index.get("weight_map", {})
    if not weight_map:
        raise ValueError(f"Empty weight_map in {index_path}")

    source_headers: dict[str, tuple[Path, int, dict]] = {}
    for shard_name in dict.fromkeys(weight_map.values()):
        shard_path = transformer / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing shard: {shard_path}")
        data_start, header = _read_header(shard_path)
        source_headers[shard_name] = (shard_path, data_start, header)

    merged_header: dict = {"__metadata__": _metadata(source_root)}
    cursor = 0
    copy_plan = []
    for tensor_name, shard_name in weight_map.items():
        shard_path, data_start, shard_header = source_headers[shard_name]
        entry = shard_header.get(tensor_name)
        if not isinstance(entry, dict):
            raise KeyError(f"{tensor_name} missing from {shard_path.name}")
        source_begin, source_end = map(int, entry["data_offsets"])
        size = source_end - source_begin
        if size < 0:
            raise ValueError(f"Invalid offsets for {tensor_name}")
        merged_header[tensor_name] = {
            "dtype": entry["dtype"],
            "shape": entry["shape"],
            "data_offsets": [cursor, cursor + size],
        }
        copy_plan.append((shard_path, data_start + source_begin, size))
        cursor += size

    header_bytes = json.dumps(
        merged_header, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    header_bytes += b" " * ((8 - len(header_bytes) % 8) % 8)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".merging")
    if temporary.exists():
        temporary.unlink()
    try:
        with temporary.open("wb") as destination:
            destination.write(struct.pack("<Q", len(header_bytes)))
            destination.write(header_bytes)
            open_source = None
            open_path = None
            try:
                for shard_path, absolute_offset, remaining in copy_plan:
                    if shard_path != open_path:
                        if open_source is not None:
                            open_source.close()
                        open_source = shard_path.open("rb")
                        open_path = shard_path
                    open_source.seek(absolute_offset)
                    while remaining:
                        block = open_source.read(min(COPY_CHUNK, remaining))
                        if not block:
                            raise EOFError(f"Unexpected EOF in {shard_path}")
                        destination.write(block)
                        remaining -= len(block)
            finally:
                if open_source is not None:
                    open_source.close()

        expected_size = 8 + len(header_bytes) + cursor
        if temporary.stat().st_size != expected_size:
            raise ValueError(
                f"Merged size mismatch: {temporary.stat().st_size} != {expected_size}"
            )
        with safe_open(temporary, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            metadata = handle.metadata() or {}
        if len(keys) != len(weight_map) or set(keys) != set(weight_map):
            raise ValueError(
                f"Merged key mismatch: {len(keys)} != {len(weight_map)}"
            )
        if metadata.get("star7_variant") != "fasth3_vsa_datafree_v1":
            raise ValueError("Merged file is missing the FastH3 VSA manifest")
        os.replace(temporary, output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise

    print(f"Merged {len(weight_map)} tensors into {output}")
    print(f"Size: {output.stat().st_size / (1024 ** 3):.2f} GiB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    merge(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
