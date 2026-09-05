# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

import json
import struct
from abc import ABC, abstractmethod
from typing import BinaryIO, Dict, Optional, Sequence, Tuple

import safetensors
import torch

from mergekit.io.lazy_unpickle import (
    DeferredLoad,
    LazyUnpickleModule,
    TorchArchiveReader,
    torch_lazy_load,
)


_SAFETENSORS_DTYPE_MAP = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def _contiguous_stride(shape: Sequence[int]) -> Tuple[int, ...]:
    strides = []
    acc = 1
    for dim in reversed(shape):
        strides.append(acc)
        acc *= dim
    return tuple(reversed(strides))


class TensorLoader(ABC):
    """Base class for (potentially lazy) tensor loaders."""

    @abstractmethod
    def get_tensor(self, key: str) -> torch.Tensor: ...

    @abstractmethod
    def keys(self) -> Sequence[str]: ...

    @classmethod
    def get(
        cls,
        shard_path: str,
        use_lazy_unpickle: bool = False,
        device: Optional[str] = None,
        low_memory_extreme: bool = False,
    ) -> "TensorLoader":
        if shard_path.lower().endswith(".safetensors"):
            if low_memory_extreme:
                return SafetensorsSingleTensorLoader(shard_path, device=device)
            # not a subclass of TensorLoader, but exposes same api
            return safetensors.safe_open(
                shard_path, framework="pt", device=device or "cpu"
            )
        elif use_lazy_unpickle:
            return LazyPickleLoader(shard_path, device=device)
        return DumbPytorchLoader(shard_path, device=device)


class LazyPickleLoader(TensorLoader):
    """Loader for pytorch files using a custom unpickler and vigorous monkeypatching."""

    zip_reader: TorchArchiveReader
    index: Dict[str, DeferredLoad]
    device: Optional[str] = None

    def __init__(self, path: str, device: Optional[str] = None):
        self.zip_reader = TorchArchiveReader(path)
        self.device = device
        with torch_lazy_load():
            self.index = torch.load(path, pickle_module=LazyUnpickleModule)

    def get_tensor(self, key: str) -> torch.Tensor:
        if key not in self.index:
            raise KeyError(key)

        return self.index[key].execute(self.zip_reader, map_location=self.device)

    def keys(self) -> Sequence[str]:
        return self.index.keys()


class DumbPytorchLoader(TensorLoader):
    """Naive `torch.load` shard loading."""

    tensors: Dict[str, torch.Tensor]

    def __init__(self, path: str, device: Optional[str] = None):
        self.tensors = torch.load(path, map_location=device, weights_only=True)

    def get_tensor(self, key: str) -> torch.Tensor:
        return self.tensors[key]

    def keys(self) -> Sequence[str]:
        return self.tensors.keys()


class SafetensorsSingleTensorLoader(TensorLoader):
    """Reads individual tensors from a safetensors file via seek+read.

    Unlike `safetensors.safe_open`, this never mmaps the whole shard. On
    Windows the whole-file mmap counts toward process private commit (~shard
    size), so keeping a plain file handle and reading only the requested
    tensor's bytes bounds source-loading commit by tensor size instead of
    shard size.
    """

    _header: Dict[str, dict]
    _data_offset: int
    _file: BinaryIO
    path: str
    device: Optional[str]

    def __init__(self, path: str, device: Optional[str] = None):
        self.path = path
        self.device = device
        self._file = open(path, "rb")
        (header_len,) = struct.unpack("<Q", self._file.read(8))
        header = json.loads(self._file.read(header_len).decode("utf-8"))
        self._header = {
            key: value for key, value in header.items() if key != "__metadata__"
        }
        self._data_offset = 8 + header_len

    def keys(self) -> Sequence[str]:
        return list(self._header.keys())

    def get_tensor(self, key: str) -> torch.Tensor:
        info = self._header[key]
        begin, end = info["data_offsets"]
        self._file.seek(self._data_offset + begin)
        raw = self._file.read(end - begin)
        dtype = _SAFETENSORS_DTYPE_MAP[info["dtype"]]
        shape = info["shape"]
        storage = torch.UntypedStorage.from_buffer(raw, "little", dtype=dtype)
        tensor = torch.tensor([], dtype=dtype, device=storage.device)
        tensor.set_(storage, 0, shape, _contiguous_stride(shape))
        if self.device is not None:
            tensor = tensor.to(self.device)
        return tensor

    def __del__(self):
        file = getattr(self, "_file", None)
        if file is not None:
            try:
                file.close()
            except Exception:
                pass
