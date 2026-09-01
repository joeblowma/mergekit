# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

import json
import logging
import os
import shutil
import struct
import tempfile
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import safetensors
import torch

LOG = logging.getLogger(__name__)


class TensorWriter:
    out_path: str
    override_basename: Optional[str]
    max_shard_size: int
    safe_serialization: bool
    use_async: bool
    disk_spool: bool
    expected_final_size: Optional[int]

    shards_written: int
    weight_map: Dict[str, str]
    current_shard: Dict[str, torch.Tensor]
    current_shard_size: int

    _lock: threading.RLock
    _executor: Optional[ThreadPoolExecutor]
    _write_futures: List[Future]
    _spool_dir: Optional[str]
    _spool_entries: Dict[str, Tuple[str, int]]
    _spool_sequence: int
    _finalized: bool

    def __init__(
        self,
        out_path: str,
        max_shard_size: int = 1000 * 1000 * 1000 * 5,
        safe_serialization: bool = True,
        override_basename: Optional[str] = None,
        use_async: bool = False,
        max_write_threads: int = 1,
        disk_spool: bool = False,
        expected_final_size: Optional[int] = None,
    ) -> None:
        os.makedirs(out_path, exist_ok=True)

        if disk_spool and not safe_serialization:
            raise ValueError("disk_spool requires safe_serialization=True")
        if disk_spool and use_async:
            raise ValueError("disk_spool requires synchronous writes (use_async=False)")
        if disk_spool and expected_final_size is None:
            raise ValueError(
                "disk_spool requires expected_final_size so staging free space can be preflighted"
            )
        if expected_final_size is not None and expected_final_size < 0:
            raise ValueError("expected_final_size must be non-negative")

        self.out_path = out_path
        self.override_basename = override_basename
        self.max_shard_size = max_shard_size
        self.safe_serialization = safe_serialization
        self.use_async = use_async
        self.disk_spool = disk_spool
        self.expected_final_size = expected_final_size

        self.shards_written = 0
        self.weight_map = {}
        self.current_shard = {}
        self.current_shard_size = 0
        self.total_size = 0

        self._lock = threading.RLock()
        self._write_futures = []
        self._spool_dir = None
        self._spool_entries = {}
        self._spool_sequence = 0
        self._finalized = False
        if self.use_async:
            self._executor = ThreadPoolExecutor(max_workers=max_write_threads)
        if self.disk_spool:
            self._spool_dir = self._create_spool_dir()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # A failed merge must leave the staged tensors available for diagnosis or
        # recovery rather than publishing a partial adapter.
        if exc_type is None or not self.disk_spool:
            self.finalize()

    def save_tensor(self, name: str, tensor: torch.Tensor, clone: bool = False):
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        if clone:
            tensor = tensor.clone()

        tensor_size = tensor.numel() * tensor.element_size()
        with self._lock:
            if self.disk_spool:
                self._spool_tensor(name, tensor, tensor_size)
                return
            if (
                self.current_shard
                and self.max_shard_size > 0
                and self.current_shard_size + tensor_size > self.max_shard_size
            ):
                self._flush_current_shard()

            self.current_shard[name] = tensor
            self.current_shard_size += tensor_size

    def _flush_current_shard(self):
        """
        Dispatches the current shard to be written to disk by a background thread.

        This method must be called within a lock.
        """
        if not self.current_shard:
            return

        shard_to_write = self.current_shard
        shard_index = self.shards_written

        self.total_size += self.current_shard_size
        self.current_shard = {}
        self.current_shard_size = 0
        self.shards_written += 1

        prefix, extension = self._get_name_components()
        shard_name = f"{prefix}-{shard_index + 1}.{extension}"
        shard_path = os.path.join(self.out_path, shard_name)
        for key in shard_to_write:
            self.weight_map[key] = shard_name

        if self.use_async:
            LOG.info(f"Dispatching shard #{shard_index + 1} to be written to disk.")

            future = self._executor.submit(
                self._write_shard_task, shard_to_write, shard_index, shard_path
            )
            self._write_futures.append(future)
        else:
            # directly execute
            self._write_shard_task(
                shard_data=shard_to_write,
                shard_index=shard_index,
                shard_path=shard_path,
            )

    def _write_shard_task(
        self, shard_data: Dict[str, torch.Tensor], shard_index: int, shard_path: str
    ):
        LOG.info(f"Writing shard #{shard_index + 1}...")
        if self.safe_serialization:
            self._save_st(shard_data, shard_path)
        else:
            torch.save(shard_data, shard_path)
        LOG.info(f"Finished writing shard #{shard_index + 1}.")

    def finalize(self):
        if self.disk_spool:
            with self._lock:
                if self._finalized:
                    return
                self._finalize_spool()
                self._finalized = True
            return

        with self._lock:
            self._flush_current_shard()

        if self.use_async:
            if self._write_futures:
                LOG.info(
                    f"Waiting for {len(self._write_futures)} shard{'s' if len(self._write_futures) > 1 else ''} to finish writing..."
                )
                for future in self._write_futures:
                    future.result()
                LOG.info("All shards have been written to disk.")
                self._write_futures.clear()
            self._executor.shutdown()

        with self._lock:
            LOG.info("Finalizing shard names and creating index file.")
            prefix, extension = self._get_name_components()
            total_shards = self.shards_written

            # Standardize shard names to Hugging Face format
            name_remap = {}
            if total_shards == 1:
                name_remap[f"{prefix}-1.{extension}"] = f"{prefix}.{extension}"
            else:
                for idx in range(total_shards):
                    old_name = f"{prefix}-{idx + 1}.{extension}"
                    new_name = (
                        f"{prefix}-{idx + 1:05d}-of-{total_shards:05d}.{extension}"
                    )
                    name_remap[old_name] = new_name

            for old_name, new_name in name_remap.items():
                old_path = os.path.join(self.out_path, old_name)
                new_path = os.path.join(self.out_path, new_name)
                os.rename(old_path, new_path)

            # Write index file if needed
            if total_shards > 1:
                for key in self.weight_map:
                    self.weight_map[key] = name_remap.get(
                        self.weight_map[key], self.weight_map[key]
                    )

                index_filename = f"{prefix}.{extension}.index.json"
                index_path = os.path.join(self.out_path, index_filename)
                with open(index_path, "w", encoding="utf-8") as f:
                    content = {
                        "metadata": {
                            "total_size": self.total_size,
                            "mergekit_version": "0.1.4",
                        },
                        "weight_map": self.weight_map,
                    }
                    json.dump(content, f, indent=2)

    def _get_name_components(self):
        if self.override_basename:
            basename = self.override_basename
        else:
            basename = "model" if self.safe_serialization else "pytorch_model"

        extension = "safetensors" if self.safe_serialization else "bin"
        return basename, extension

    def _save_st(self, shard_data: dict, shard_path: str):
        def _do_save(sd):
            safetensors.torch.save_file(
                sd,
                shard_path,
                metadata={"format": "pt"},
            )

        try:
            _do_save(shard_data)
        except RuntimeError as e:
            if (
                len(e.args) > 0
                and isinstance(e.args[0], str)
                and "share memory" in e.args[0]
            ):
                LOG.warning(
                    "Your model has duplicated tensors but the --clone-tensors "
                    "flag is not set."
                )
                shard_data = {key: shard_data[key].clone() for key in shard_data}
                _do_save(shard_data)
            else:
                raise

    def _create_spool_dir(self) -> str:
        """Create a durable, self-describing staging directory beside the output."""
        assert self.expected_final_size is not None
        required = max(1, 2 * self.expected_final_size)
        available = shutil.disk_usage(self.out_path).free
        if available < required:
            raise OSError(
                "Insufficient free space for disk_spool: "
                f"need about {required} bytes (2x expected_final_size), "
                f"but only {available} bytes are available at {self.out_path!r}."
            )

        spool_dir = tempfile.mkdtemp(
            prefix=".mergekit-tensor-spool-", dir=self.out_path
        )
        self._write_spool_manifest(spool_dir, state="staging")
        LOG.info("Disk-spooling tensors to recoverable staging directory %s", spool_dir)
        return spool_dir

    def _write_spool_manifest(self, spool_dir: str, state: str) -> None:
        """Atomically persist enough information to identify a recoverable spool."""
        manifest_path = os.path.join(spool_dir, "manifest.json")
        temporary_path = f"{manifest_path}.tmp"
        entries = [
            {"name": name, "file": filename, "size": size}
            for name, (filename, size) in self._spool_entries.items()
        ]
        with open(temporary_path, "w", encoding="utf-8") as manifest:
            json.dump(
                {
                    "format": "mergekit-tensor-writer-spool-v1",
                    "state": state,
                    "expected_final_size": self.expected_final_size,
                    "tensors": entries,
                },
                manifest,
                indent=2,
                sort_keys=True,
            )
            manifest.flush()
            os.fsync(manifest.fileno())
        os.replace(temporary_path, manifest_path)
        self._fsync_directory(spool_dir)

    def _spool_tensor(self, name: str, tensor: torch.Tensor, tensor_size: int) -> None:
        """Synchronously save one tensor, releasing its caller-owned reference on return."""
        assert self._spool_dir is not None
        previous_entry = self._spool_entries.get(name)
        sequence = self._spool_sequence
        spool_name = f"tensor-{sequence:08d}.safetensors"
        spool_path = os.path.join(self._spool_dir, spool_name)
        temporary_path = f"{spool_path}.tmp"
        self._save_st({name: tensor}, temporary_path)
        self._fsync_file(temporary_path)
        os.replace(temporary_path, spool_path)
        self._spool_sequence += 1
        self._spool_entries[name] = (spool_name, tensor_size)
        self.total_size = sum(size for _, size in self._spool_entries.values())
        self._write_spool_manifest(self._spool_dir, state="staging")
        if previous_entry is not None and previous_entry[0] != spool_name:
            # Once the manifest references the replacement, the superseded copy
            # is no longer required for recovery.
            os.unlink(os.path.join(self._spool_dir, previous_entry[0]))
            self._fsync_directory(self._spool_dir)

    def _finalize_spool(self) -> None:
        """Assemble staged safetensors by copying payload bytes, never all tensors at once."""
        assert self._spool_dir is not None
        self._write_spool_manifest(self._spool_dir, state="finalizing")
        prefix, _ = self._get_name_components()
        destination = os.path.join(self.out_path, f"{prefix}.safetensors")
        temporary_destination = f"{destination}.tmp"

        try:
            header, payloads = self._spool_final_header()
            with open(temporary_destination, "wb") as output:
                output.write(struct.pack("<Q", len(header)))
                output.write(header)
                for spool_path, offset, size in payloads:
                    with open(spool_path, "rb") as staged:
                        staged.seek(offset)
                        self._copy_exact(staged, output, size)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_destination, destination)
            self._fsync_directory(self.out_path)
        except Exception:
            # The staging directory and its manifest intentionally remain intact.
            # Remove only an incomplete final temp file, never the previous output.
            if os.path.exists(temporary_destination):
                os.unlink(temporary_destination)
            raise

        # A fully closed and atomically published artifact is the only condition
        # under which staging may be removed.
        shutil.rmtree(self._spool_dir)
        self._spool_dir = None

    def _spool_final_header(self) -> Tuple[bytes, List[Tuple[str, int, int]]]:
        entries = {"__metadata__": {"format": "pt"}}
        payloads = []
        data_offset = 0
        assert self._spool_dir is not None
        for name, (filename, expected_size) in self._spool_entries.items():
            path = os.path.join(self._spool_dir, filename)
            tensor_info, payload_offset, payload_size = self._read_staged_tensor(
                path, name
            )
            if payload_size != expected_size:
                raise RuntimeError(
                    f"Staged tensor {name!r} has {payload_size} bytes, expected {expected_size}"
                )
            entries[name] = {
                "dtype": tensor_info["dtype"],
                "shape": tensor_info["shape"],
                "data_offsets": [data_offset, data_offset + payload_size],
            }
            payloads.append((path, payload_offset, payload_size))
            data_offset += payload_size

        header = json.dumps(entries, separators=(",", ":")).encode("utf-8")
        # Safetensors headers are padded to eight bytes; JSON whitespace is valid.
        header += b" " * ((-len(header)) % 8)
        return header, payloads

    @staticmethod
    def _read_staged_tensor(path: str, name: str) -> Tuple[dict, int, int]:
        with open(path, "rb") as staged:
            header_size_data = staged.read(8)
            if len(header_size_data) != 8:
                raise RuntimeError(
                    f"Staged tensor file {path!r} has no safetensors header"
                )
            header_size = struct.unpack("<Q", header_size_data)[0]
            header = json.loads(staged.read(header_size))
        tensor_names = [key for key in header if key != "__metadata__"]
        if tensor_names != [name]:
            raise RuntimeError(
                f"Staged tensor file {path!r} does not contain only {name!r}"
            )
        tensor_info = header[name]
        offsets = tensor_info.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or offsets[0] > offsets[1]
        ):
            raise RuntimeError(f"Staged tensor file {path!r} has invalid data offsets")
        payload_size = offsets[1] - offsets[0]
        return tensor_info, 8 + header_size + offsets[0], payload_size

    @staticmethod
    def _copy_exact(source, destination, size: int) -> None:
        remaining = size
        while remaining:
            chunk = source.read(min(16 * 1024 * 1024, remaining))
            if not chunk:
                raise RuntimeError("Unexpected end of staged tensor payload")
            destination.write(chunk)
            remaining -= len(chunk)

    @staticmethod
    def _fsync_file(path: str) -> None:
        # Windows requires a writable handle for FlushFileBuffers/fsync.
        with open(path, "r+b") as file:
            os.fsync(file.fileno())

    @staticmethod
    def _fsync_directory(path: str) -> None:
        # Windows cannot open directories as file descriptors. File fsync and the
        # atomic replace above still provide the durable staging contract there.
        if os.name == "nt":
            return
        directory_fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
