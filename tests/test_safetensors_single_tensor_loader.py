"""Tests for SafetensorsSingleTensorLoader — the seek+read safetensors reader.

Covers dtype/shape correctness, memory ownership, loader selection via
TensorLoader.get and LazyTensorLoader, and metadata exclusion.
"""

import struct
from pathlib import Path

import pytest
import safetensors
import safetensors.torch
import torch

from mergekit.io.loader import SafetensorsSingleTensorLoader, TensorLoader
from mergekit.io.lazy_tensor_loader import LazyTensorLoader, ShardedTensorIndex


# All dtypes supported by safetensors with corresponding torch dtypes
DTYPE_CASES = [
    ("F64", torch.float64),
    ("F32", torch.float32),
    ("F16", torch.float16),
    ("BF16", torch.bfloat16),
    ("I64", torch.int64),
    ("I32", torch.int32),
    ("I16", torch.int16),
    ("I8", torch.int8),
    ("U8", torch.uint8),
    ("BOOL", torch.bool),
]

# Shape cases: 0-d scalar, 1-d, 2-d square, 2-d non-square
SHAPE_CASES = [
    ((), "scalar"),
    ((5,), "1d"),
    ((4, 4), "2d_square"),
    ((3, 7), "2d_nonsquare"),
]


def _make_tensor(safetensors_dtype: str, shape):
    """Create a torch tensor of the given safetensors dtype and shape."""
    dtype_map = {
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
    dtype = dtype_map[safetensors_dtype]
    if safetensors_dtype == "BOOL":
        return torch.randint(0, 2, shape, dtype=dtype)
    if dtype in (torch.float16, torch.bfloat16):
        return torch.randn(shape, dtype=torch.float32).to(dtype)
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype)
    # Integer types
    iinfo = torch.iinfo(dtype)
    return torch.randint(max(iinfo.min, 0), min(iinfo.max, 10), shape, dtype=dtype)


@pytest.fixture
def synthetic_safetensors_file(tmp_path):
    """Create a synthetic safetensors file covering all dtypes and shapes."""
    tensors = {}
    for st_dtype, _ in DTYPE_CASES:
        for shape, shape_name in SHAPE_CASES:
            key = f"{st_dtype}_{shape_name}"
            tensors[key] = _make_tensor(st_dtype, shape)

    # Add metadata to test exclusion
    path = str(tmp_path / "test.safetensors")
    safetensors.torch.save_file(tensors, path, metadata={"foo": "bar", "format": "pt"})
    return path, tensors


class TestReaderCorrectness:
    """Reader correctness: dtype, shape, values, memory ownership, keys."""

    def test_tensors_match_safe_open(self, synthetic_safetensors_file):
        """Every tensor matches safetensors.safe_open in dtype, shape, and values."""
        path, expected = synthetic_safetensors_file
        loader = SafetensorsSingleTensorLoader(path)
        ref = safetensors.safe_open(path, framework="pt", device="cpu")

        for key in expected:
            actual = loader.get_tensor(key)
            expected_tensor = ref.get_tensor(key)
            assert actual.dtype == expected_tensor.dtype, f"{key}: dtype mismatch"
            assert actual.shape == expected_tensor.shape, f"{key}: shape mismatch"
            assert torch.equal(actual, expected_tensor), f"{key}: value mismatch"

    def test_keys_excludes_metadata(self, synthetic_safetensors_file):
        """keys() returns tensor names only, excluding __metadata__."""
        path, expected = synthetic_safetensors_file
        loader = SafetensorsSingleTensorLoader(path)
        keys = loader.keys()
        assert "__metadata__" not in keys
        assert set(keys) == set(expected.keys())

    def test_tensor_owns_writable_private_memory(self, synthetic_safetensors_file):
        """Returned tensor owns writable private memory (mutating doesn't affect fresh read)."""
        path, expected = synthetic_safetensors_file
        loader = SafetensorsSingleTensorLoader(path)

        for key in expected:
            t1 = loader.get_tensor(key)
            if t1.numel() == 0:
                continue

            # Every read returns a tensor backed by independent storage
            t2 = loader.get_tensor(key)
            if t1.storage().nbytes() > 0:
                assert t1.data_ptr() != t2.data_ptr(), (
                    f"{key}: fresh read shares storage with previous read"
                )

            # In-place mutation succeeds and does not affect a fresh read
            if t1.dtype == torch.bool:
                t1.logical_not_()
            else:
                t1.mul_(2)
            t3 = loader.get_tensor(key)
            if t1.numel() > 0 and t3.numel() > 0:
                assert not torch.equal(t1, t3), (
                    f"{key}: mutation affected fresh read"
                )

    def test_zero_length_tensor(self, tmp_path):
        """Zero-length tensors are handled correctly."""
        tensors = {"zero": torch.zeros(0, dtype=torch.float32)}
        path = str(tmp_path / "zero.safetensors")
        safetensors.torch.save_file(tensors, path)

        loader = SafetensorsSingleTensorLoader(path)
        t = loader.get_tensor("zero")
        assert t.shape == (0,)
        assert t.dtype == torch.float32
        assert t.numel() == 0

    def test_contiguous_storage(self, synthetic_safetensors_file):
        """Returned tensor is contiguous in storage."""
        path, expected = synthetic_safetensors_file
        loader = SafetensorsSingleTensorLoader(path)
        for key in expected:
            t = loader.get_tensor(key)
            assert t.is_contiguous(), f"{key}: not contiguous"

    def test_del_closes_file(self, tmp_path):
        """__del__ closes the underlying file handle."""
        tensors = {"a": torch.ones(3, dtype=torch.float32)}
        path = str(tmp_path / "del_test.safetensors")
        safetensors.torch.save_file(tensors, path)

        loader = SafetensorsSingleTensorLoader(path)
        loader.__del__()
        assert loader._file.closed

    def test_device_forwarding(self, tmp_path):
        """device parameter is forwarded to the returned tensor."""
        tensors = {"a": torch.ones(3, dtype=torch.float32)}
        path = str(tmp_path / "device_test.safetensors")
        safetensors.torch.save_file(tensors, path)

        loader = SafetensorsSingleTensorLoader(path, device="cpu")
        t = loader.get_tensor("a")
        assert t.device.type == "cpu"

    def test_multiple_reads_same_tensor(self, synthetic_safetensors_file):
        """Reading the same tensor multiple times returns independent copies."""
        path, expected = synthetic_safetensors_file
        loader = SafetensorsSingleTensorLoader(path)
        for key in expected:
            t1 = loader.get_tensor(key)
            t2 = loader.get_tensor(key)
            assert torch.equal(t1, t2)
            # They should be independent storage (not views)
            if t1.numel() > 0:
                assert t1.data_ptr() != t2.data_ptr()

    def test_scalar_tensor(self, tmp_path):
        """0-d scalar tensors are read correctly."""
        tensors = {"scalar": torch.tensor(42, dtype=torch.int32)}
        path = str(tmp_path / "scalar.safetensors")
        safetensors.torch.save_file(tensors, path)

        loader = SafetensorsSingleTensorLoader(path)
        t = loader.get_tensor("scalar")
        assert t.shape == ()
        assert t.dtype == torch.int32
        assert t.item() == 42


class TestLoaderSelection:
    """Loader selection via TensorLoader.get and LazyTensorLoader."""

    def test_tensor_loader_get_default(self, tmp_path):
        """TensorLoader.get with low_memory_extreme=False returns safe_open (not our loader)."""
        tensors = {"a": torch.ones(3, dtype=torch.float32)}
        path = str(tmp_path / "test.safetensors")
        safetensors.torch.save_file(tensors, path)

        loader = TensorLoader.get(path, low_memory_extreme=False)
        assert not isinstance(loader, SafetensorsSingleTensorLoader)
        assert hasattr(loader, "get_tensor")
        assert hasattr(loader, "keys")

    def test_tensor_loader_get_extreme(self, tmp_path):
        """TensorLoader.get with low_memory_extreme=True returns SafetensorsSingleTensorLoader."""
        tensors = {"a": torch.ones(3, dtype=torch.float32)}
        path = str(tmp_path / "test.safetensors")
        safetensors.torch.save_file(tensors, path)

        loader = TensorLoader.get(path, low_memory_extreme=True)
        assert isinstance(loader, SafetensorsSingleTensorLoader)

    def test_lazy_tensor_loader_default(self, tmp_path):
        """LazyTensorLoader default low_memory_extreme=False uses safe_open path."""
        tensors = {"a": torch.ones(3, dtype=torch.float32)}
        path = str(tmp_path / "model.safetensors")
        safetensors.torch.save_file(tensors, path)

        index = ShardedTensorIndex.from_file(path)
        loader = LazyTensorLoader(index)
        assert loader.low_memory_extreme is False
        t = loader.get_tensor("a")
        assert torch.equal(t, torch.ones(3))

    def test_lazy_tensor_loader_extreme(self, tmp_path):
        """LazyTensorLoader with low_memory_extreme=True uses SafetensorsSingleTensorLoader."""
        tensors = {"a": torch.ones(3, dtype=torch.float32)}
        path = str(tmp_path / "model.safetensors")
        safetensors.torch.save_file(tensors, path)

        index = ShardedTensorIndex.from_file(path)
        loader = LazyTensorLoader(index)
        loader.low_memory_extreme = True
        assert loader.low_memory_extreme is True
        t = loader.get_tensor("a")
        assert torch.equal(t, torch.ones(3))

    def test_pickle_branches_unaffected(self, tmp_path):
        """Pickle branches (non-safetensors) are unaffected by low_memory_extreme."""
        tensors = {"a": torch.ones(3, dtype=torch.float32)}
        path = str(tmp_path / "test.bin")
        torch.save(tensors, path)

        # low_memory_extreme=True on a .bin file should still use DumbPytorchLoader
        loader = TensorLoader.get(path, low_memory_extreme=True)
        assert not isinstance(loader, SafetensorsSingleTensorLoader)
        assert torch.equal(loader.get_tensor("a"), torch.ones(3))

    def test_lazy_unpickle_branches_unaffected(self, tmp_path):
        """Lazy unpickle branches are unaffected by low_memory_extreme."""
        tensors = {"a": torch.ones(3, dtype=torch.float32)}
        path = str(tmp_path / "test_lazy.bin")
        torch.save(tensors, path)

        loader = TensorLoader.get(path, use_lazy_unpickle=True, low_memory_extreme=True)
        assert not isinstance(loader, SafetensorsSingleTensorLoader)
        assert torch.equal(loader.get_tensor("a"), torch.ones(3))

    def test_low_memory_extreme_false_preserves_default_behavior(self, tmp_path):
        """Default and --low-memory-alone do NOT use SafetensorsSingleTensorLoader."""
        tensors = {"a": torch.ones(3, dtype=torch.float32)}
        path = str(tmp_path / "test.safetensors")
        safetensors.torch.save_file(tensors, path)

        # Default behavior (no flags)
        loader_default = TensorLoader.get(path)
        assert not isinstance(loader_default, SafetensorsSingleTensorLoader)

        # Explicit False
        loader_false = TensorLoader.get(path, low_memory_extreme=False)
        assert not isinstance(loader_false, SafetensorsSingleTensorLoader)