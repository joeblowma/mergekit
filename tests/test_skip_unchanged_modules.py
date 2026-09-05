"""Tests for the --skip-unchanged-modules flag in mergekit-extract-lora.

Covers the byte-exact base-vs-finetuned comparison (identical / different /
shape-mismatch / missing-base), planning-side removal from ``modules_to_save``
and the task graph, and an end-to-end tiny-model run whose unchanged embeddings
are dropped from both the adapter and its config (with a PEFT round-trip load).
"""

import json
import os
import shutil

import pytest
import safetensors.torch
import torch
from click.testing import CliRunner

from mergekit.architecture import WeightInfo
from mergekit.common import ModelReference
from mergekit.io.tasks import LoaderCache, SaveTensor
from mergekit.options import MergeOptions
from mergekit.scripts import extract_lora


def _write_model(path, tensors):
    os.makedirs(path, exist_ok=True)
    safetensors.torch.save_file(
        {k: torch.as_tensor(v).clone() for k, v in tensors.items()},
        os.path.join(path, "model.safetensors"),
        metadata={"format": "pt"},
    )


def _reset_loader_cache():
    LoaderCache.loaders.clear()
    try:
        del LoaderCache._instance.value
    except AttributeError:
        pass


@pytest.fixture
def loader_cache():
    _reset_loader_cache()
    LoaderCache().setup(MergeOptions())
    yield
    _reset_loader_cache()


class TestCompareUnchanged:
    def test_identical(self, loader_cache, tmp_path):
        base = str(tmp_path / "base")
        ft = str(tmp_path / "ft")
        emb = torch.randn(4, 3)
        _write_model(base, {"embed.weight": emb})
        _write_model(ft, {"embed.weight": emb.clone()})
        wi = WeightInfo(name="embed.weight")

        should_skip, bytes_saved = extract_lora._compare_unchanged(
            ModelReference(model=base),
            ModelReference(model=ft),
            wi,
            "embed",
            (4, 3),
            (4, 3),
            {},
        )
        assert should_skip is True
        assert bytes_saved == emb.numel() * emb.element_size()

    def test_different(self, loader_cache, tmp_path):
        base = str(tmp_path / "base")
        ft = str(tmp_path / "ft")
        _write_model(base, {"embed.weight": torch.randn(4, 3)})
        _write_model(ft, {"embed.weight": torch.randn(4, 3)})
        wi = WeightInfo(name="embed.weight")

        should_skip, _ = extract_lora._compare_unchanged(
            ModelReference(model=base),
            ModelReference(model=ft),
            wi,
            "embed",
            (4, 3),
            (4, 3),
            {},
        )
        assert should_skip is False

    def test_shape_mismatch_keeps_full_copy(self, loader_cache, tmp_path):
        base = str(tmp_path / "base")
        ft = str(tmp_path / "ft")
        _write_model(base, {"embed.weight": torch.randn(4, 3)})
        _write_model(ft, {"embed.weight": torch.randn(8, 3)})  # vocab extension
        wi = WeightInfo(name="embed.weight")

        should_skip, _ = extract_lora._compare_unchanged(
            ModelReference(model=base),
            ModelReference(model=ft),
            wi,
            "embed",
            (8, 3),
            (4, 3),
            {},
        )
        assert should_skip is False

    def test_missing_base_tensor_keeps_full_copy(self, loader_cache, tmp_path):
        base = str(tmp_path / "base")
        ft = str(tmp_path / "ft")
        _write_model(base, {"other.weight": torch.randn(4, 3)})
        _write_model(ft, {"embed.weight": torch.randn(4, 3)})
        wi = WeightInfo(name="embed.weight", optional=True)

        should_skip, _ = extract_lora._compare_unchanged(
            ModelReference(model=base),
            ModelReference(model=ft),
            wi,
            "embed",
            (4, 3),
            (4, 3),
            {},
        )
        assert should_skip is False


def _make_picollama_with_identical_embeddings(tmp_path):
    """Base + finetuned picollama where only projection layers differ."""
    from tests.common import make_picollama

    base = str(tmp_path / "base")
    ft = str(tmp_path / "ft")
    make_picollama(base)
    shutil.copytree(base, ft)

    ft_shard = os.path.join(ft, "model.safetensors")
    tensors = {k: v.clone() for k, v in safetensors.torch.load_file(ft_shard).items()}
    for key in list(tensors):
        if "proj" in key and key.endswith(".weight"):
            tensors[key] = tensors[key] + torch.randn_like(tensors[key]) * 0.01
    safetensors.torch.save_file(tensors, ft_shard, metadata={"format": "pt"})
    return base, ft


class TestSkipUnchangedPlanning:
    def test_plan_removes_skipped_modules_and_tasks(self, tmp_path):
        base, ft = _make_picollama_with_identical_embeddings(tmp_path)
        _reset_loader_cache()
        LoaderCache().setup(MergeOptions())
        modules_to_save = []

        plan = extract_lora.plan_extraction(
            base_model_ref=ModelReference(model=base),
            model_ref=ModelReference(model=ft),
            modules_to_save=modules_to_save,
            out_path=str(tmp_path / "adapter"),
            options=MergeOptions(),
            max_rank=2,
            exclude_regexes=[],
            include_regexes=[],
            skip_unchanged_modules=True,
        )

        assert "embed_tokens" not in modules_to_save
        assert "lm_head" not in modules_to_save

        save_names = [
            t.tensor_name for t in plan.tasks if isinstance(t, SaveTensor)
        ]
        assert not any(
            "embed" in n or "lm_head" in n or "norm" in n for n in save_names
        )

        skipped_names = {s.name for s in plan.skipped_modules}
        assert "model.embed_tokens" in skipped_names
        assert "lm_head" in skipped_names


class TestSkipUnchangedEndToEnd:
    def test_identical_embeddings_dropped_and_peft_loads(self, tmp_path):
        base, ft = _make_picollama_with_identical_embeddings(tmp_path)
        out = str(tmp_path / "adapter")

        result = CliRunner().invoke(
            extract_lora.main,
            [
                "--base-model",
                base,
                "--model",
                ft,
                "--out-path",
                out,
                "--max-rank",
                "2",
                "--skip-unchanged-modules",
            ],
        )
        assert result.exit_code == 0, f"{result.output}\n{result.exception}"

        adapter = safetensors.torch.load_file(
            os.path.join(out, "adapter_model.safetensors")
        )
        assert adapter, "Adapter has no tensors"
        assert not any(
            ("embed" in k or "lm_head" in k or "norm" in k) for k in adapter
        )

        with open(os.path.join(out, "adapter_config.json")) as f:
            config = json.load(f)
        assert "embed_tokens" not in config["modules_to_save"]
        assert "lm_head" not in config["modules_to_save"]
        assert all("proj" in t for t in config["target_modules"])

        # Real PEFT round-trip load.
        import peft
        from transformers import LlamaForCausalLM

        base_model = LlamaForCausalLM.from_pretrained(base)
        peft_model = peft.PeftModel.from_pretrained(base_model, out)
        assert peft_model.active_adapters == ["default"]

    def test_works_with_low_memory_extreme(self, tmp_path):
        base, ft = _make_picollama_with_identical_embeddings(tmp_path)
        out = str(tmp_path / "adapter")

        result = CliRunner().invoke(
            extract_lora.main,
            [
                "--base-model",
                base,
                "--model",
                ft,
                "--out-path",
                out,
                "--max-rank",
                "2",
                "--skip-unchanged-modules",
                "--low-memory-extreme",
            ],
        )
        assert result.exit_code == 0, f"{result.output}\n{result.exception}"
        adapter = safetensors.torch.load_file(
            os.path.join(out, "adapter_model.safetensors")
        )
        assert not any(("embed" in k or "lm_head" in k) for k in adapter)
