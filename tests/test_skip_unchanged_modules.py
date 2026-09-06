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
from mergekit.io import tasks as io_tasks
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


def _make_picollama_lora_pair(tmp_path, changed_key="model.layers.0.self_attn.q_proj.weight"):
    """Base + finetuned picollama where only one LoRA projection module differs.

    The changed module gets a rank-1 weight delta; every other LoRA target
    (and every full-save module) stays byte-identical to the base model.
    """
    from tests.common import make_picollama

    base = str(tmp_path / "base")
    ft = str(tmp_path / "ft")
    make_picollama(base)
    shutil.copytree(base, ft)

    ft_shard = os.path.join(ft, "model.safetensors")
    tensors = {k: v.clone() for k, v in safetensors.torch.load_file(ft_shard).items()}
    w = tensors[changed_key]
    torch.manual_seed(0)
    tensors[changed_key] = w + (
        torch.randn(w.shape[0], 1) @ torch.randn(1, w.shape[1])
    ) * 0.1
    safetensors.torch.save_file(tensors, ft_shard, metadata={"format": "pt"})
    return base, ft


def _make_picogranite_bias_pair(tmp_path):
    """Base + finetuned Granite with biases on every Linear module.

    Only ``model.layers.0.self_attn.q_proj.weight`` gets a rank-1 delta; its
    bias (and every other module, weight or bias) stays byte-identical.
    """
    from transformers import GraniteConfig, GraniteForCausalLM

    base = str(tmp_path / "base")
    ft = str(tmp_path / "ft")
    cfg = GraniteConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=1,
        attention_bias=True,
        mlp_bias=True,
    )
    GraniteForCausalLM(cfg).save_pretrained(base, safe_serialization=True)
    shutil.copytree(base, ft)

    ft_shard = os.path.join(ft, "model.safetensors")
    tensors = {k: v.clone() for k, v in safetensors.torch.load_file(ft_shard).items()}
    key = "model.layers.0.self_attn.q_proj.weight"
    w = tensors[key]
    torch.manual_seed(0)
    tensors[key] = w + (
        torch.randn(w.shape[0], 1) @ torch.randn(1, w.shape[1])
    ) * 0.1
    safetensors.torch.save_file(tensors, ft_shard, metadata={"format": "pt"})
    return base, ft


class TestZeroDeltaSkipEndToEnd:
    def test_zero_delta_module_skipped_with_flag(self, tmp_path):
        base, ft = _make_picollama_lora_pair(tmp_path)
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

        changed_a = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        changed_b = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"
        unchanged_a = "base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight"
        unchanged_b = "base_model.model.model.layers.0.self_attn.k_proj.lora_B.weight"

        # The changed module is present and reconstructs its (rank-1) delta.
        assert changed_a in adapter, "changed module lora_A missing"
        assert changed_b in adapter, "changed module lora_B missing"
        A = adapter[changed_a]
        B = adapter[changed_b]
        assert torch.count_nonzero(A).item() > 0
        assert torch.count_nonzero(B).item() > 0
        ft_q = safetensors.torch.load_file(
            os.path.join(ft, "model.safetensors")
        )["model.layers.0.self_attn.q_proj.weight"]
        base_q = safetensors.torch.load_file(
            os.path.join(base, "model.safetensors")
        )["model.layers.0.self_attn.q_proj.weight"]
        delta = (ft_q - base_q).to(torch.float32)
        recon = (B @ A).to(torch.float32)
        assert torch.allclose(recon, delta, atol=1e-4)

        # The byte-identical module has no lora_A/lora_B keys at all.
        assert unchanged_a not in adapter
        assert unchanged_b not in adapter

        with open(os.path.join(out, "adapter_config.json")) as f:
            config = json.load(f)
        assert "q_proj" in config["target_modules"]
        assert "k_proj" not in config["target_modules"]
        assert "k_proj" not in config.get("rank_pattern", {})

    def test_zero_delta_module_written_without_flag(self, tmp_path):
        base, ft = _make_picollama_lora_pair(tmp_path)
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
            ],
        )
        assert result.exit_code == 0, f"{result.output}\n{result.exception}"

        adapter = safetensors.torch.load_file(
            os.path.join(out, "adapter_model.safetensors")
        )
        unchanged_a = "base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight"
        unchanged_b = "base_model.model.model.layers.0.self_attn.k_proj.lora_B.weight"
        assert unchanged_a in adapter, "zero module lora_A missing without flag"
        assert unchanged_b in adapter, "zero module lora_B missing without flag"
        assert torch.count_nonzero(adapter[unchanged_a]).item() == 0
        assert torch.count_nonzero(adapter[unchanged_b]).item() == 0


class TestZeroDeltaBias:
    def test_bias_coverage(self, tmp_path):
        base, ft = _make_picogranite_bias_pair(tmp_path)
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

        # Kept module (nonzero weight delta, zero bias delta) still writes its
        # lora_B.bias, even though the bias delta is zero.
        q_bias = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.bias"
        assert q_bias in adapter, "kept module's lora_B.bias missing"
        assert torch.count_nonzero(adapter[q_bias]).item() == 0

        # Skipped module (zero weight AND bias delta) has no lora keys at all.
        k = "base_model.model.model.layers.0.self_attn.k_proj"
        assert f"{k}.lora_A.weight" not in adapter
        assert f"{k}.lora_B.weight" not in adapter
        assert f"{k}.lora_B.bias" not in adapter


class TestZeroDeltaRegistry:
    def test_record_get_reset(self):
        extract_lora.reset_zero_delta_skips()
        assert extract_lora.get_zero_delta_skips() == {}

        extract_lora.record_zero_delta_skip("model.layers.0.self_attn.q_proj.weight", 1234)
        extract_lora.record_zero_delta_skip("model.layers.0.self_attn.k_proj.weight", 5678)
        assert extract_lora.get_zero_delta_skips() == {
            "model.layers.0.self_attn.q_proj.weight": 1234,
            "model.layers.0.self_attn.k_proj.weight": 5678,
        }

        extract_lora.reset_zero_delta_skips()
        assert extract_lora.get_zero_delta_skips() == {}


class TestZeroDeltaSkipEdge:
    def test_weight_zero_bias_nonzero_kept(self):
        """A zero weight delta with a nonzero bias delta is NOT skipped."""
        extract_lora.reset_zero_delta_skips()
        ref = ModelReference(model="local-base")
        load = io_tasks.LoadTensor(model=ref, tensor="linear.weight")
        task = extract_lora.TaskVectorDecompositionTask(
            weight_info=WeightInfo(name="linear.weight"),
            input_task=load,
            max_rank=4,
            distribute_scale=True,
            skip_zero_delta=True,
            bias_task=load,
        )

        a, b = task.execute(
            task_vector=torch.zeros(5, 3), bias=torch.ones(5)
        )

        assert a is not None and b is not None
        # The module is kept: zero A/B is still produced (lossless), with the
        # real bias written separately by LoRABiasSaveTask.
        assert torch.count_nonzero(a).item() == 0
        assert torch.count_nonzero(b).item() == 0
        assert extract_lora.get_zero_delta_skips() == {}
