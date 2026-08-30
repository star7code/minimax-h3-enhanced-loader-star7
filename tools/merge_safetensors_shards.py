"""Merge safetensors shards without materializing their tensors in RAM."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
from pathlib import Path


def _read_header(path: Path):
    with path.open("rb") as handle:
        (header_size,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(header_size))
    metadata = header.pop("__metadata__", {})
    payload_offset = 8 + header_size
    payload_size = path.stat().st_size - payload_offset
    end = max((item["data_offsets"][1] for item in header.values()), default=0)
    if end != payload_size:
        raise ValueError(f"Non-contiguous safetensors payload in {path}: {end} != {payload_size}")
    return header, metadata, payload_offset, payload_size


def merge(source: Path, output: Path, manifest_path: Path | None) -> None:
    shards = sorted(source.glob("star7_fasth3_int8-*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No Star7 INT8 shards found in {source}")
    if output.exists():
        raise FileExistsError(output)

    manifest = {}
    if manifest_path and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    merged = {}
    shard_info = []
    base_offset = 0
    for shard in shards:
        tensors, _, payload_offset, payload_size = _read_header(shard)
        duplicates = set(merged).intersection(tensors)
        if duplicates:
            raise ValueError(f"Duplicate tensor names in {shard}: {sorted(duplicates)[:5]}")
        for name, item in tensors.items():
            start, end = item["data_offsets"]
            merged[name] = {
                "dtype": item["dtype"],
                "shape": item["shape"],
                "data_offsets": [start + base_offset, end + base_offset],
            }
        shard_info.append((shard, payload_offset, payload_size))
        base_offset += payload_size

    metadata = {
        "format": "pt",
        "star7_variant": str(manifest.get("variant", "fasth3_dense_v1")),
        "star7_sampling_profile": str(manifest.get("sampling_profile", "fasth3_4step_v1")),
        "star7_quantization": str(manifest.get("quantization", "int8_tensorwise_convrot")),
        "star7_fasth3_manifest": json.dumps(manifest, ensure_ascii=False, separators=(",", ":")),
    }
    header = {"__metadata__": metadata, **merged}
    header_bytes = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * ((8 - len(header_bytes) % 8) % 8)

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    try:
        with partial.open("xb") as target:
            target.write(struct.pack("<Q", len(header_bytes)))
            target.write(header_bytes)
            for index, (shard, payload_offset, payload_size) in enumerate(shard_info, 1):
                print(f"[Star7 merge] {index}/{len(shard_info)} {shard.name}", flush=True)
                with shard.open("rb") as source_file:
                    source_file.seek(payload_offset)
                    remaining = payload_size
                    while remaining:
                        block = source_file.read(min(64 * 1024 * 1024, remaining))
                        if not block:
                            raise EOFError(f"Unexpected end of {shard}")
                        target.write(block)
                        remaining -= len(block)
            target.flush()
            os.fsync(target.fileno())
        os.replace(partial, output)
    finally:
        if partial.exists():
            partial.unlink()

    tensors, saved_metadata, _, payload_size = _read_header(output)
    if len(tensors) != len(merged) or payload_size != base_offset:
        raise RuntimeError("Merged safetensors validation failed")

    digest = hashlib.sha256()
    with output.open("rb") as handle:
        for block in iter(lambda: handle.read(64 * 1024 * 1024), b""):
            digest.update(block)
    sha256 = digest.hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{sha256}  {output.name}\n", encoding="ascii"
    )
    print(
        f"[Star7 merge] complete | tensors={len(tensors)} | "
        f"size={output.stat().st_size / 2**30:.2f} GiB | sha256={sha256} | "
        f"profile={saved_metadata.get('star7_sampling_profile')}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    merge(args.source.resolve(), args.output.resolve(), args.manifest.resolve() if args.manifest else None)


if __name__ == "__main__":
    main()
