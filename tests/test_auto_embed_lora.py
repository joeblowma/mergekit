"""Tests for the --auto-embed-lora / --embed-lora-tolerance flags.

Covers the adaptive rank selector, end-to-end decomposition of a synthetic
low-rank embedding delta (key naming, reconstruction residual, config, PEFT
round-trip), the full-rank fallback path, and the zero-delta skip path.
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
from mergekit.scripts import extract_lora


class TestSelectAdaptiveRank:
    def test_zero_delta(self):
        assert extract_lora._select_adaptive_rank(torch.zeros(8), 64, 32, 1e-3) == 0

    def test_low_rank_delta(self):
        torch.manual_seed(0)
        u = torch.randn(64, 2)
        v = torch.randn(2, 32)
        _, s, _ = torch.linalg.svd(
            (u @ v).to(torch.float32), full_matrices=False
        )
        assert extract_lora._select_adaptive_rank(s, 64, 32, 1e-3) == 2

    def test_full_rank_noise_falls_back(self):
        torch.manual_seed(1)
        _, s, _ = torch.linalg.svd(
            torch.randn(64, 32).to(torch.float32), full_matrices=False
        )
        assert extract_lora._select_adaptive_rank(s, 64, 32, 1e-3) == -1

    # --- _needed_rank_for_tolerance ---

    def test_needed_rank_zero_delta(self):
        assert extract_lora._needed_rank_for_tolerance(torch.zeros(8), 1e-3) == 0

    def test_needed_rank_low_rank(self):
        torch.manual_seed(0)
        _, s, _ = torch.linalg.svd(
            (torch.randn(64, 2) @ torch.randn(2, 32)).to(torch.float32),
            full_matrices=False,
        )
        assert extract_lora._needed_rank_for_tolerance(s, 1e-3) == 2

    def test_needed_rank_full_when_tolerance_not_met(self):
        torch.manual_seed(1)
        _, s, _ = torch.linalg.svd(
            torch.randn(64, 32).to(torch.float32), full_matrices=False
        )
        # full_rank is 32; tolerance 1e-3 won't be met so it returns full_rank
        assert extract_lora._needed_rank_for_tolerance(s, 1e-3) == 32


def _write_picollama_pair(tmp_path, embed_delta_fn, modify_proj=True):
    """Base + finetuned picollama with a custom embedding delta."""
    from tests.common import make_picollama

    base = str(tmp_path / "base")
    ft = str(tmp_path / "ft")
    make_picollama(base)
    shutil.copytree(base, ft)

    ft_shard = os.path.join(ft, "model.safetensors")
    tensors = {k: v.clone() for k, v in safetensors.torch.load_file(ft_shard).items()}
    emb_key = "model.embed_tokens.weight"
    tensors[emb_key] = tensors[emb_key] + embed_delta_fn(tensors[emb_key])
    if modify_proj:
        for key in list(tensors):
            if "proj" in key and key.endswith(".weight"):
                tensors[key] = tensors[key] + torch.randn_like(tensors[key]) * 0.01
    safetensors.torch.save_file(tensors, ft_shard, metadata={"format": "pt"})
    return base, ft


def _run(base, ft, out, extra=None):
    return CliRunner().invoke(
        extract_lora.main,
        [
            "--base-model",
            base,
            "--model",
            ft,
            "--out-path",
            out,
            "--max-rank",
            "8",
            "--auto-embed-lora",
        ]
        + (extra or []),
    )


def _embed_delta(base, ft, key="model.embed_tokens.weight"):
    ft_t = safetensors.torch.load_file(os.path.join(ft, "model.safetensors"))[key]
    base_t = safetensors.torch.load_file(os.path.join(base, "model.safetensors"))[key]
    return (ft_t - base_t).to(torch.float32)


class TestAutoEmbedLoraEndToEnd:
    def test_low_rank_delta_decomposes(self, tmp_path):
        def delta_fn(_emb):
            torch.manual_seed(0)
            return (torch.randn(64, 2) @ torch.randn(2, 32)) * 0.1

        base, ft = _write_picollama_pair(tmp_path, delta_fn)
        out = str(tmp_path / "adapter")
        result = _run(base, ft, out)
        assert result.exit_code == 0, f"{result.output}\n{result.exception}"

        adapter = safetensors.torch.load_file(
            os.path.join(out, "adapter_model.safetensors")
        )
        a_key = "base_model.model.model.embed_tokens.lora_embedding_A"
        b_key = "base_model.model.model.embed_tokens.lora_embedding_B"
        assert a_key in adapter and b_key in adapter
        # No full copy and no `.weight` suffix for the embedding LoRA.
        assert "base_model.model.model.embed_tokens.weight" not in adapter
        assert "base_model.model.model.embed_tokens.lora_embedding_A.weight" not in adapter

        A = adapter[a_key]
        B = adapter[b_key]
        rank = A.shape[0]
        assert 0 < rank <= 8

        delta = _embed_delta(base, ft)
        # lora_embedding_A is [r, vocab], lora_embedding_B is [hidden, r];
        # PEFT reconstructs delta = A^T @ B^T.
        recon = (A.T @ B.T).to(torch.float32)
        residual = torch.linalg.norm(delta - recon).item()
        norm = torch.linalg.norm(delta).item()
        assert residual <= 1e-3 * norm

        with open(os.path.join(out, "adapter_config.json")) as f:
            config = json.load(f)
        assert "embed_tokens" in config["target_modules"]
        assert "embed_tokens" not in config["modules_to_save"]

        import peft
        from transformers import LlamaForCausalLM

        base_model = LlamaForCausalLM.from_pretrained(base)
        peft_model = peft.PeftModel.from_pretrained(base_model, out)
        assert peft_model.active_adapters == ["default"]

    def test_full_rank_noise_falls_back_to_full_copy(self, tmp_path):
        def delta_fn(_emb):
            torch.manual_seed(1)
            return torch.randn(64, 32) * 0.1

        base, ft = _write_picollama_pair(tmp_path, delta_fn)
        out = str(tmp_path / "adapter")
        result = _run(base, ft, out)
        assert result.exit_code == 0, f"{result.output}\n{result.exception}"

        adapter = safetensors.torch.load_file(
            os.path.join(out, "adapter_model.safetensors")
        )
        assert "base_model.model.model.embed_tokens.weight" in adapter
        assert "base_model.model.model.embed_tokens.lora_embedding_A" not in adapter

        with open(os.path.join(out, "adapter_config.json")) as f:
            config = json.load(f)
        assert "embed_tokens" in config["modules_to_save"]
        assert "embed_tokens" not in config["target_modules"]

    def test_zero_delta_writes_nothing(self, tmp_path):
        def delta_fn(_emb):
            return torch.zeros_like(_emb)

        base, ft = _write_picollama_pair(tmp_path, delta_fn)
        out = str(tmp_path / "adapter")
        result = _run(base, ft, out)
        assert result.exit_code == 0, f"{result.output}\n{result.exception}"

        adapter = safetensors.torch.load_file(
            os.path.join(out, "adapter_model.safetensors")
        )
        assert not any("embed_tokens" in k for k in adapter)

        with open(os.path.join(out, "adapter_config.json")) as f:
            config = json.load(f)
        assert "embed_tokens" not in config["modules_to_save"]
        assert "embed_tokens" not in config["target_modules"]

    def test_works_with_low_memory(self, tmp_path):
        def delta_fn(_emb):
            torch.manual_seed(0)
            return (torch.randn(64, 2) @ torch.randn(2, 32)) * 0.1

        base, ft = _write_picollama_pair(tmp_path, delta_fn)
        out = str(tmp_path / "adapter")
        result = _run(base, ft, out, extra=["--low-memory"])
        assert result.exit_code == 0, f"{result.output}\n{result.exception}"

        adapter = safetensors.torch.load_file(
            os.path.join(out, "adapter_model.safetensors")
        )
        assert "base_model.model.model.embed_tokens.lora_embedding_A" in adapter
        assert "base_model.model.model.embed_tokens.lora_embedding_B" in adapter

    # --- caplog decision-logging tests ---

    def test_low_rank_delta_logs_decision(self, tmp_path, caplog):
        import logging

        def delta_fn(_emb):
            torch.manual_seed(0)
            return (torch.randn(64, 2) @ torch.randn(2, 32)) * 0.1

        base, ft = _write_picollama_pair(tmp_path, delta_fn)
        out = str(tmp_path / "adapter")
        with caplog.at_level(logging.INFO, logger="extract_lora"):
            result = _run(base, ft, out)
        assert result.exit_code == 0, f"{result.output}\n{result.exception}"
        lora_msgs = [
            r for r in caplog.records
            if "adaptive LoRA rank=" in r.message
        ]
        assert len(lora_msgs) == 1, f"expected 1 lora log, got {len(lora_msgs)}: {[r.message for r in caplog.records]}"
        msg = lora_msgs[0].message
        assert "embed_tokens" in msg
        assert "rank=" in msg
        assert "tolerance=" in msg
        assert "energy=" in msg

    def test_full_rank_fallback_logs_decision(self, tmp_path, caplog):
        import logging

        def delta_fn(_emb):
            torch.manual_seed(1)
            return torch.randn(64, 32) * 0.1

        base, ft = _write_picollama_pair(tmp_path, delta_fn)
        out = str(tmp_path / "adapter")
        with caplog.at_level(logging.INFO, logger="extract_lora"):
            result = _run(base, ft, out)
        assert result.exit_code == 0, f"{result.output}\n{result.exception}"
        full_msgs = [
            r for r in caplog.records
            if "full copy" in r.message
        ]
        assert len(full_msgs) == 1, (
            f"expected 1 full-copy log, got {len(full_msgs)}: "
            f"{[r.message for r in caplog.records]}"
        )
        msg = full_msgs[0].message
        assert "embed_tokens" in msg
        assert "needs rank" in msg
        assert "breakeven rank" in msg

    def test_zero_delta_logs_decision(self, tmp_path, caplog):
        import logging

        def delta_fn(_emb):
            return torch.zeros_like(_emb)

        base, ft = _write_picollama_pair(tmp_path, delta_fn)
        out = str(tmp_path / "adapter")
        with caplog.at_level(logging.INFO, logger="extract_lora"):
            result = _run(base, ft, out)
        assert result.exit_code == 0, f"{result.output}\n{result.exception}"
        # Both embed_tokens and lm_head produce a zero-delta log; check for
        # the one we care about.
        embed_msgs = [
            r for r in caplog.records
            if "model.embed_tokens: delta is zero" in r.message
        ]
        assert len(embed_msgs) == 1, (
            f"expected 1 embed_tokens skip log, got {len(embed_msgs)}: "
            f"{[r.message for r in caplog.records]}"
        )


class TestAutoEmbedLoraPrecedence:
    def test_embed_lora_takes_precedence(self, tmp_path, caplog):
        """--embed-lora disables --auto-embed-lora with a warning."""
        import logging

        from tests.common import make_picollama

        base = str(tmp_path / "base")
        ft = str(tmp_path / "ft")
        make_picollama(base)
        shutil.copytree(base, ft)
        ft_shard = os.path.join(ft, "model.safetensors")
        tensors = {k: v.clone() for k, v in safetensors.torch.load_file(ft_shard).items()}
        tensors["model.embed_tokens.weight"] = (
            tensors["model.embed_tokens.weight"] + torch.randn_like(tensors["model.embed_tokens.weight"]) * 0.01
        )
        safetensors.torch.save_file(tensors, ft_shard, metadata={"format": "pt"})

        out = str(tmp_path / "adapter")
        with caplog.at_level(logging.WARNING, logger="extract_lora"):
            result = _run(base, ft, out, extra=["--embed-lora"])
        assert result.exit_code == 0, f"{result.output}\n{result.exception}"
        assert any("precedence" in r.message for r in caplog.records)


class _FakeWriter:
    """Minimal TensorWriter stand-in for driving task.execute directly."""

    def __init__(self):
        self.tensors = {}

    def save_tensor(self, name, tensor, clone=False):
        self.tensors[name] = tensor.detach().clone()


class TestSvdFitsOnCuda:
    """Unit tests for the VRAM-fit heuristic."""

    def test_cuda_unavailable_returns_false(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert (
            extract_lora._svd_fits_on_cuda(100, free_memory_source=lambda: 10**9)
            is False
        )

    def test_fits_when_free_memory_is_enough(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        # required = 100 * 2.5 = 250 <= 1000 * 0.9 = 900
        assert (
            extract_lora._svd_fits_on_cuda(100, free_memory_source=lambda: 1000)
            is True
        )

    def test_does_not_fit_when_free_memory_insufficient(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        # required = 1000 * 2.5 = 2500 > 1000 * 0.9 = 900
        assert (
            extract_lora._svd_fits_on_cuda(1000, free_memory_source=lambda: 1000)
            is False
        )

    def test_zero_free_memory_returns_false(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert (
            extract_lora._svd_fits_on_cuda(1, free_memory_source=lambda: 0) is False
        )


class TestSvdOomFallback:
    """CPU fallback when an SVD does not fit on (or OOMs) the accelerator."""

    @staticmethod
    def _adaptive_task(tmp_path, tolerance=1e-3):
        writer_task = io_tasks.TensorWriterTask(
            out_path=str(tmp_path), max_shard_size=-1
        )
        ref = ModelReference(model="local-base")
        load = io_tasks.LoadTensor(model=ref, tensor="embed_tokens.weight")
        return extract_lora.AdaptiveEmbeddingSaveTask(
            weight_info=WeightInfo(name="embed_tokens.weight", is_embed=True),
            writer_task=writer_task,
            base_tensor=load,
            model_tensor=load,
            tolerance=tolerance,
        )

    @staticmethod
    def _low_rank_delta():
        torch.manual_seed(0)
        base = torch.zeros(8, 4)
        delta = (torch.randn(8, 2) @ torch.randn(2, 4)) * 0.1
        return base, delta

    def test_adaptive_embedding_oom_falls_back_to_cpu(self, tmp_path, monkeypatch):
        """A GPU SVD that OOMs is retried on CPU and still produces correct A/B."""
        monkeypatch.setattr(extract_lora, "_on_cuda", lambda t: True)
        monkeypatch.setattr(
            extract_lora,
            "_svd_fits_on_cuda",
            lambda bytes_, free_memory_source=None: True,
        )

        real_svd = torch.linalg.svd
        calls = {"n": 0}

        def fake_svd(a, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise torch.OutOfMemoryError("simulated CUDA OOM")
            return real_svd(a, **kwargs)

        monkeypatch.setattr(torch.linalg, "svd", fake_svd)

        base, delta = self._low_rank_delta()
        model = base + delta

        task = self._adaptive_task(tmp_path)
        writer = _FakeWriter()
        extract_lora.reset_embed_lora_decisions()
        task.execute(writer, base, model)

        assert calls["n"] == 2
        assert extract_lora.get_embed_lora_decisions() == {"embed_tokens": ("lora", 2)}

        a = writer.tensors["base_model.model.embed_tokens.lora_embedding_A"]
        b = writer.tensors["base_model.model.embed_tokens.lora_embedding_B"]
        recon = (a.T @ b.T).to(torch.float32)
        assert torch.linalg.norm(delta - recon).item() <= 1e-3 * torch.linalg.norm(
            delta
        ).item()

    def test_adaptive_embedding_too_large_for_gpu_uses_cpu(
        self, tmp_path, monkeypatch, caplog
    ):
        """When the heuristic says no fit, the SVD goes straight to CPU with a log."""
        import logging

        monkeypatch.setattr(extract_lora, "_on_cuda", lambda t: True)
        monkeypatch.setattr(
            extract_lora,
            "_svd_fits_on_cuda",
            lambda bytes_, free_memory_source=None: False,
        )
        monkeypatch.setattr(extract_lora, "_cuda_free_bytes", lambda: 2**30)

        base, delta = self._low_rank_delta()
        model = base + delta

        task = self._adaptive_task(tmp_path)
        writer = _FakeWriter()
        extract_lora.reset_embed_lora_decisions()
        with caplog.at_level(logging.INFO, logger="extract_lora"):
            task.execute(writer, base, model)

        assert extract_lora.get_embed_lora_decisions() == {"embed_tokens": ("lora", 2)}
        assert any("too large for GPU" in r.message for r in caplog.records)

    def test_task_vector_decomposition_oom_falls_back_to_cpu(self, monkeypatch):
        """The pre-existing decomposition path also retries a GPU OOM on CPU."""
        real_svd = torch.linalg.svd
        calls = {"n": 0}

        def fake_svd(a, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise torch.OutOfMemoryError("simulated CUDA OOM")
            return real_svd(a, **kwargs)

        monkeypatch.setattr(torch.linalg, "svd", fake_svd)

        task = extract_lora.TaskVectorDecompositionTask(
            weight_info=WeightInfo(name="linear.weight"),
            input_task=io_tasks.LoadTensor(
                model=ModelReference(model="local-base"), tensor="linear.weight"
            ),
            max_rank=4,
            distribute_scale=True,
        )
        torch.manual_seed(1)
        tv = torch.randn(5, 3)
        a, b = task.execute(task_vector=tv)

        assert calls["n"] == 2
        assert torch.allclose(b @ a, tv, atol=1e-5)
        assert a.dtype == tv.dtype and b.dtype == tv.dtype


class TestAdaptiveEmbeddingLogMessages:
    """Direct task-execution tests for the INFO log lines."""

    @staticmethod
    def _adaptive_task(tmp_path, tolerance=1e-3):
        writer_task = io_tasks.TensorWriterTask(
            out_path=str(tmp_path), max_shard_size=-1
        )
        ref = ModelReference(model="local-base")
        load = io_tasks.LoadTensor(model=ref, tensor="embed_tokens.weight")
        return extract_lora.AdaptiveEmbeddingSaveTask(
            weight_info=WeightInfo(name="embed_tokens.weight", is_embed=True),
            writer_task=writer_task,
            base_tensor=load,
            model_tensor=load,
            tolerance=tolerance,
        )

    def test_missing_base_logs_info(self, tmp_path, caplog):
        import logging

        task = self._adaptive_task(tmp_path)
        writer = _FakeWriter()
        model = torch.randn(8, 4)
        extract_lora.reset_embed_lora_decisions()
        with caplog.at_level(logging.INFO, logger="extract_lora"):
            task.execute(writer, base=None, model=model)
        assert extract_lora.get_embed_lora_decisions() == {
            "embed_tokens": ("full", None)
        }
        assert any(
            "base tensor missing" in r.message for r in caplog.records
        ), f"missing base log not found: {[r.message for r in caplog.records]}"

    def test_missing_model_logs_info(self, tmp_path, caplog):
        import logging

        task = self._adaptive_task(tmp_path)
        writer = _FakeWriter()
        extract_lora.reset_embed_lora_decisions()
        with caplog.at_level(logging.INFO, logger="extract_lora"):
            task.execute(writer, base=torch.randn(8, 4), model=None)
        assert extract_lora.get_embed_lora_decisions() == {
            "embed_tokens": ("skipped", 0)
        }
        assert any(
            "model tensor missing" in r.message for r in caplog.records
        ), f"missing model log not found: {[r.message for r in caplog.records]}"

    def test_full_copy_logs_needed_rank_and_breakeven(self, tmp_path, caplog):
        """Full-copy fallback logs the tolerance-needed rank and breakeven."""
        import logging

        task = self._adaptive_task(tmp_path, tolerance=1e-3)
        writer = _FakeWriter()
        base = torch.zeros(8, 4)
        # Full-rank noise delta — tolerance won't be met at any rank < full_rank
        torch.manual_seed(1)
        model = base + torch.randn(8, 4) * 0.1
        extract_lora.reset_embed_lora_decisions()
        with caplog.at_level(logging.INFO, logger="extract_lora"):
            task.execute(writer, base, model)
        assert extract_lora.get_embed_lora_decisions() == {
            "embed_tokens": ("full", None)
        }
        full_msgs = [r for r in caplog.records if "full copy" in r.message]
        assert len(full_msgs) == 1, (
            f"expected 1 full-copy log, got {len(full_msgs)}: "
            f"{[r.message for r in caplog.records]}"
        )
        msg = full_msgs[0].message
        assert "needs rank" in msg
        assert "breakeven rank" in msg
        # 8*4//(8+4) + 1 = 32//12 + 1 = 2 + 1 = 3
        assert "breakeven rank 3" in msg
