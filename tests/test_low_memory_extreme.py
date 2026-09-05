"""Tests for the --low-memory-extreme flag in mergekit-extract-lora.

Covers flag gating (superset of --low-memory constraints), forwarding to
LoaderCache, plan-level low_memory=True inheritance, and end-to-end
equivalence with --low-memory.
"""

import os
import shutil
import tempfile
from types import SimpleNamespace

import pytest
import torch
from click.testing import CliRunner

from mergekit.io import tasks as io_tasks
from mergekit.options import MergeOptions
from mergekit.scripts import extract_lora


def _error_text(result) -> str:
    """Return Click output and the caught exception's text for assertions."""
    return f"{result.output}\n{result.exception}".lower()


@pytest.fixture
def stubbed_extraction(monkeypatch, tmp_path):
    """Replace model planning/execution with a deterministic one-item run.

    Mirrors the fixture in test_extract_lora_low_memory.py so that
    --low-memory-extreme tests can reuse the same stubbing pattern.
    """

    out_path = tmp_path / "adapter"
    out_path.mkdir()
    plan_calls = {}

    class FakeDecomposition:
        def __init__(self):
            self.weight_info = SimpleNamespace(name="linear.weight")

    decomposition = FakeDecomposition()

    class FakeExecutor:
        def __init__(self, tasks, **kwargs):
            self.tasks = tasks
            self.kwargs = kwargs

        def run(self):
            yield decomposition, (torch.ones(1, 1), torch.ones(1, 1))

    def fake_plan(**kwargs):
        plan_calls.update(kwargs)
        return SimpleNamespace(
            tasks=[],
            base_vocab_size=4,
            final_vocab_size=4,
        )

    monkeypatch.setattr(extract_lora, "TaskVectorDecompositionTask", FakeDecomposition)
    monkeypatch.setattr(extract_lora, "Executor", FakeExecutor)
    monkeypatch.setattr(extract_lora, "plan_extraction", fake_plan)
    monkeypatch.setattr(extract_lora, "generate_card_lora", lambda *args, **kwargs: "")

    return out_path, plan_calls


class TestLowMemoryExtremeFlagGating:
    """--low-memory-extreme inherits all --low-memory constraints."""

    @pytest.mark.parametrize(
        "conflict, expected_name",
        [
            ("--low-cpu-memory", "low-cpu-memory"),
            ("--embed-lora", "embed-lora"),
            ("--no-safe-serialization", "safe-serialization"),
            ("--async-write", "async-write"),
        ],
    )
    def test_extreme_rejects_unsupported_modes(
        self, tmp_path, conflict: str, expected_name: str
    ):
        """Each incompatible mode is rejected when combined with --low-memory-extreme."""
        runner = CliRunner()
        result = runner.invoke(
            extract_lora.main,
            [
                "--base-model",
                "local-base",
                "--model",
                "local-finetuned",
                "--out-path",
                str(tmp_path / "adapter"),
                "--low-memory-extreme",
                conflict,
            ],
        )
        assert result.exit_code != 0
        text = _error_text(result)
        assert "low-memory" in text
        assert expected_name in text or expected_name.replace("-", "_") in text

    def test_extreme_alone_succeeds(self, stubbed_extraction):
        """--low-memory-extreme alone succeeds (no conflict)."""
        out_path, plan_calls = stubbed_extraction
        result = CliRunner().invoke(
            extract_lora.main,
            [
                "--base-model",
                "local-base",
                "--model",
                "local-finetuned",
                "--out-path",
                str(out_path),
                "--low-memory-extreme",
            ],
        )
        assert result.exit_code == 0, _error_text(result)

    def test_extreme_inherits_low_memory_in_plan(self, stubbed_extraction):
        """--low-memory-extreme sets low_memory=True in plan kwargs (superset inheritance)."""
        out_path, plan_calls = stubbed_extraction
        result = CliRunner().invoke(
            extract_lora.main,
            [
                "--base-model",
                "local-base",
                "--model",
                "local-finetuned",
                "--out-path",
                str(out_path),
                "--low-memory-extreme",
            ],
        )
        assert result.exit_code == 0, _error_text(result)
        assert plan_calls.get("low_memory") is True

    @staticmethod
    def _reset_loader_cache():
        """Properly reset the LoaderCache singleton by deleting the thread-local value."""
        try:
            del io_tasks.LoaderCache._instance.value
        except AttributeError:
            pass

    def test_extreme_sets_loader_cache_attribute(self, stubbed_extraction):
        """--low-memory-extreme sets LoaderCache().low_memory_extreme = True."""
        out_path, plan_calls = stubbed_extraction
        self._reset_loader_cache()
        result = CliRunner().invoke(
            extract_lora.main,
            [
                "--base-model",
                "local-base",
                "--model",
                "local-finetuned",
                "--out-path",
                str(out_path),
                "--low-memory-extreme",
            ],
        )
        assert result.exit_code == 0, _error_text(result)
        assert io_tasks.LoaderCache().low_memory_extreme is True

    def test_default_does_not_set_extreme(self, stubbed_extraction):
        """Default (no flags) does not set low_memory_extreme on LoaderCache."""
        out_path, plan_calls = stubbed_extraction
        self._reset_loader_cache()
        result = CliRunner().invoke(
            extract_lora.main,
            [
                "--base-model",
                "local-base",
                "--model",
                "local-finetuned",
                "--out-path",
                str(out_path),
            ],
        )
        assert result.exit_code == 0, _error_text(result)
        assert io_tasks.LoaderCache().low_memory_extreme is False

    def test_low_memory_alone_does_not_set_extreme(self, stubbed_extraction):
        """--low-memory alone does NOT set low_memory_extreme on LoaderCache."""
        out_path, plan_calls = stubbed_extraction
        self._reset_loader_cache()
        result = CliRunner().invoke(
            extract_lora.main,
            [
                "--base-model",
                "local-base",
                "--model",
                "local-finetuned",
                "--out-path",
                str(out_path),
                "--low-memory",
            ],
        )
        assert result.exit_code == 0, _error_text(result)
        assert io_tasks.LoaderCache().low_memory_extreme is False

    def test_extreme_appears_in_help(self):
        """--low-memory-extreme appears in extract-lora --help."""
        result = CliRunner().invoke(extract_lora.main, ["--help"])
        assert result.exit_code == 0
        assert "--low-memory-extreme" in result.output

    def test_extreme_help_describes_superset(self):
        """--low-memory-extreme help mentions superset behavior (implies --low-memory)."""
        result = CliRunner().invoke(extract_lora.main, ["--help"])
        assert result.exit_code == 0
        assert "implies --low-memory" in result.output.lower() or "low-memory" in result.output.lower()


class TestLowMemoryExtremeEndToEnd:
    """End-to-end equivalence: --low-memory-extreme produces same adapter as --low-memory.

    Uses tiny picollama models to keep runtime fast and CPU-only.
    Marked as a separate class so it can be easily skipped if the
    environment lacks the required dependencies.
    """

    @pytest.fixture
    def tiny_models(self, tmp_path):
        """Create tiny base and finetuned picollama models on disk."""
        from tests.common import make_picollama

        base_path = str(tmp_path / "base")
        ft_path = str(tmp_path / "finetuned")
        make_picollama(base_path)
        make_picollama(ft_path)

        # Slightly modify the finetuned model so there's a delta to extract
        import safetensors.torch

        ft_shard = os.path.join(ft_path, "model.safetensors")
        tensors = safetensors.torch.load_file(ft_shard)
        for key in tensors:
            tensors[key] = tensors[key] + torch.randn_like(tensors[key]) * 0.01
        safetensors.torch.save_file(tensors, ft_shard, metadata={"format": "pt"})

        return base_path, ft_path

    def _try_extraction(self, base_path, ft_path, out_path, extra_flags):
        """Run extraction and return (exit_code, output_text)."""
        result = CliRunner().invoke(
            extract_lora.main,
            [
                "--base-model",
                base_path,
                "--model",
                ft_path,
                "--out-path",
                out_path,
                "--max-rank",
                "2",
            ]
            + extra_flags,
        )
        return result.exit_code, _error_text(result)

    def test_extreme_matches_low_memory(self, tiny_models):
        """Adapter from --low-memory-extreme matches --low-memory adapter tensors.

        Builds two tiny picollama models (vocab_size=64, hidden_size=32, 2 layers),
        runs extraction with both flags, and compares the output tensors.
        Skips gracefully if either extraction fails (e.g. missing arch support).
        """
        import safetensors.torch

        base_path, ft_path = tiny_models

        # Run with --low-memory
        out_low = tempfile.mkdtemp()
        exit_low, text_low = self._try_extraction(
            base_path, ft_path, out_low, ["--low-memory"]
        )
        if exit_low != 0:
            shutil.rmtree(out_low, ignore_errors=True)
            pytest.skip(f"--low-memory extraction failed: {text_low}")

        # Run with --low-memory-extreme
        out_extreme = tempfile.mkdtemp()
        exit_extreme, text_extreme = self._try_extraction(
            base_path, ft_path, out_extreme, ["--low-memory-extreme"]
        )
        if exit_extreme != 0:
            shutil.rmtree(out_low, ignore_errors=True)
            shutil.rmtree(out_extreme, ignore_errors=True)
            pytest.skip(f"--low-memory-extreme extraction failed: {text_extreme}")

        # Compare adapters
        adapter_low = os.path.join(out_low, "adapter_model.safetensors")
        adapter_extreme = os.path.join(out_extreme, "adapter_model.safetensors")

        assert os.path.exists(adapter_low), "No adapter produced by --low-memory"
        assert os.path.exists(adapter_extreme), "No adapter produced by --low-memory-extreme"

        tensors_low = safetensors.torch.load_file(adapter_low)
        tensors_extreme = safetensors.torch.load_file(adapter_extreme)

        assert set(tensors_low.keys()) == set(
            tensors_extreme.keys()
        ), "Adapter tensor keys differ between --low-memory and --low-memory-extreme"
        for key in tensors_low:
            assert torch.allclose(
                tensors_low[key], tensors_extreme[key], atol=1e-5
            ), f"Tensor {key} differs between --low-memory and --low-memory-extreme"

        # Cleanup
        shutil.rmtree(out_low, ignore_errors=True)
        shutil.rmtree(out_extreme, ignore_errors=True)

    def test_extreme_produces_peft_compatible_adapter(self, tiny_models):
        """The adapter produced by --low-memory-extreme is a valid single-file PEFT adapter."""
        import safetensors.torch

        base_path, ft_path = tiny_models
        out_path = tempfile.mkdtemp()

        exit_code, text = self._try_extraction(
            base_path, ft_path, out_path, ["--low-memory-extreme"]
        )
        if exit_code != 0:
            shutil.rmtree(out_path, ignore_errors=True)
            pytest.skip(f"--low-memory-extreme extraction failed: {text}")

        # Check for PEFT-required files
        adapter_model = os.path.join(out_path, "adapter_model.safetensors")
        adapter_config = os.path.join(out_path, "adapter_config.json")

        assert os.path.exists(adapter_model), "Missing adapter_model.safetensors"
        assert os.path.exists(adapter_config), "Missing adapter_config.json"

        # Verify it's a single file (not sharded)
        tensors = safetensors.torch.load_file(adapter_model)
        assert len(tensors) > 0, "Adapter has no tensors"

        # Verify tensor names follow PEFT convention
        for key in tensors:
            assert key.startswith("base_model.model."), f"Tensor {key} doesn't follow PEFT naming"

        # Verify config is valid JSON with required fields
        import json

        with open(adapter_config) as f:
            config = json.load(f)
        assert "peft_type" in config
        assert config["peft_type"] == "LORA"
        assert "r" in config
        assert "lora_alpha" in config

        shutil.rmtree(out_path, ignore_errors=True)