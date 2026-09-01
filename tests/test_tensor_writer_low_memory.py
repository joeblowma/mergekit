"""Focused tests for the bounded-memory adapter safetensors spool."""

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from mergekit.io import tasks as io_tasks
from mergekit.io.tensor_writer import TensorWriter


def _adapter_tensors():
    return {
        "base_model.model.linear.lora_A.weight": torch.arange(
            8, dtype=torch.float32
        ).reshape(2, 4),
        "base_model.model.linear.lora_B.weight": torch.arange(
            6, dtype=torch.float16
        ).reshape(3, 2),
    }


def _expected_size(tensors) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())


def _assert_single_adapter_file(out_path: Path):
    """Assert the PEFT-compatible single-file contract and no staging debris."""

    safetensors_files = sorted(
        path.name for path in out_path.iterdir() if path.suffix == ".safetensors"
    )
    assert safetensors_files == ["adapter_model.safetensors"]
    assert not [path for path in out_path.iterdir() if path.is_dir()]
    assert not list(out_path.glob("*.index.json"))
    assert not list(out_path.glob("adapter_model-*-of-*.safetensors"))
    assert not list(out_path.glob("*.tmp"))


def _write_spooled_adapter(out_path: Path):
    tensors = _adapter_tensors()
    writer = TensorWriter(
        str(out_path),
        max_shard_size=-1,
        safe_serialization=True,
        override_basename="adapter_model",
        disk_spool=True,
        expected_final_size=_expected_size(tensors),
    )
    for name, tensor in tensors.items():
        writer.save_tensor(name, tensor)
    writer.finalize()
    return tensors


def test_spool_arguments_are_opt_in_and_task_forwards_them(monkeypatch, tmp_path):
    """Both writer layers expose disabled-by-default spool controls."""

    signature = inspect.signature(TensorWriter.__init__)
    assert signature.parameters["disk_spool"].default is False
    assert signature.parameters["expected_final_size"].default is None

    default_task = io_tasks.TensorWriterTask(
        out_path=str(tmp_path / "default"), max_shard_size=-1
    )
    assert default_task.disk_spool is False
    assert default_task.expected_final_size is None

    calls = []

    class FakeWriter:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setattr(io_tasks, "TensorWriter", FakeWriter)
    expected_size = 123
    task = io_tasks.TensorWriterTask(
        out_path=str(tmp_path / "spooled"),
        max_shard_size=-1,
        safe_serialization=True,
        disk_spool=True,
        expected_final_size=expected_size,
    )
    task.execute()

    assert len(calls) == 1
    assert calls[0][1]["disk_spool"] is True
    assert calls[0][1]["expected_final_size"] == expected_size


def test_default_writer_keeps_normal_single_file_behavior(tmp_path):
    """The old writer path remains the default and has no spool artifacts."""

    out_path = tmp_path / "normal"
    writer = TensorWriter(str(out_path), safe_serialization=True)
    assert writer.disk_spool is False
    writer.save_tensor("ordinary.weight", torch.ones(2, 3, dtype=torch.float32))
    writer.finalize()

    assert sorted(path.name for path in out_path.iterdir()) == ["model.safetensors"]
    loaded = load_file(str(out_path / "model.safetensors"))
    assert loaded["ordinary.weight"].dtype == torch.float32
    assert tuple(loaded["ordinary.weight"].shape) == (2, 3)


def test_spooled_adapter_has_exact_names_dtypes_shapes_and_no_shards(tmp_path):
    """A spooled adapter is one exact PEFT safetensors file."""

    out_path = tmp_path / "adapter"
    tensors = _write_spooled_adapter(out_path)

    _assert_single_adapter_file(out_path)
    actual = load_file(str(out_path / "adapter_model.safetensors"))
    assert set(actual) == set(tensors)
    for name, expected in tensors.items():
        assert actual[name].dtype == expected.dtype
        assert tuple(actual[name].shape) == tuple(expected.shape)
        assert torch.equal(actual[name], expected)


def test_spool_staging_is_cleaned_after_success(tmp_path):
    """Successful finalization removes temporary spool/staging files."""

    out_path = tmp_path / "adapter"
    _write_spooled_adapter(out_path)

    _assert_single_adapter_file(out_path)


def test_spool_preflights_insufficient_disk_space(monkeypatch, tmp_path):
    """Insufficient staging space fails before creating spool or output files."""

    out_path = tmp_path / "adapter"
    expected_final_size = 4096
    available = 4095
    monkeypatch.setattr(
        "mergekit.io.tensor_writer.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=available),
    )

    with pytest.raises(
        OSError,
        match=rf"need about {2 * expected_final_size} bytes.*only {available} bytes",
    ):
        TensorWriter(
            str(out_path),
            safe_serialization=True,
            override_basename="adapter_model",
            disk_spool=True,
            expected_final_size=expected_final_size,
        )

    # TensorWriter creates the requested output directory before preflighting;
    # only that directory itself is allowed to exist after the early failure.
    assert out_path.is_dir()
    assert not [path for path in out_path.iterdir() if path.is_dir()]
    assert not (out_path / "adapter_model.safetensors").exists()


def test_spool_staging_is_preserved_when_finalization_fails(monkeypatch, tmp_path):
    """A failed finalization leaves staging available for diagnosis."""

    out_path = tmp_path / "adapter"
    tensors = _adapter_tensors()
    writer = TensorWriter(
        str(out_path),
        max_shard_size=-1,
        safe_serialization=True,
        override_basename="adapter_model",
        disk_spool=True,
        expected_final_size=_expected_size(tensors),
    )
    for name, tensor in tensors.items():
        writer.save_tensor(name, tensor)

    def fail_during_finalization():
        raise OSError("intentional finalization failure")

    monkeypatch.setattr(writer, "_get_name_components", fail_during_finalization)
    with pytest.raises(OSError, match="intentional finalization failure"):
        writer.finalize()

    assert not (out_path / "adapter_model.safetensors").exists()
    output_entries = [path for path in out_path.iterdir()]
    staging_attributes = (
        "_spool_dir",
        "_spool_path",
        "_staging_dir",
        "_staging_path",
    )
    staging_paths = [
        Path(getattr(writer, attribute))
        for attribute in staging_attributes
        if getattr(writer, attribute, None) is not None
    ]
    assert output_entries or any(path.exists() for path in staging_paths)
    assert any(
        path.name != "adapter_model.safetensors" for path in output_entries
    ) or any(path.exists() for path in staging_paths)


def test_spooled_adapter_is_accepted_by_peft(tmp_path):
    """The emitted file can be loaded through PEFT without a model download."""

    pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    from peft import LoraConfig, PeftModel, get_peft_model

    config = transformers.GPT2Config(
        vocab_size=32,
        n_positions=8,
        n_ctx=8,
        n_embd=8,
        n_layer=1,
        n_head=2,
        bos_token_id=0,
        eos_token_id=1,
    )
    source_model = transformers.GPT2LMHeadModel(config)
    peft_config = LoraConfig(
        r=2,
        lora_alpha=2,
        lora_dropout=0.0,
        target_modules=["c_attn"],
        task_type="CAUSAL_LM",
    )
    reference = get_peft_model(source_model, peft_config)
    reference_dir = tmp_path / "reference"
    reference.save_pretrained(str(reference_dir), safe_serialization=True)

    output_dir = tmp_path / "adapter"
    state = load_file(str(reference_dir / "adapter_model.safetensors"))
    writer = TensorWriter(
        str(output_dir),
        max_shard_size=-1,
        safe_serialization=True,
        override_basename="adapter_model",
        disk_spool=True,
        expected_final_size=_expected_size(state),
    )
    for name, tensor in state.items():
        writer.save_tensor(name, tensor)
    writer.finalize()
    (output_dir / "adapter_config.json").write_text(
        (reference_dir / "adapter_config.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    loaded_base = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(**config.to_dict())
    )
    loaded = PeftModel.from_pretrained(loaded_base, str(output_dir), is_trainable=False)

    _assert_single_adapter_file(output_dir)
    assert loaded.peft_config["default"].r == 2
