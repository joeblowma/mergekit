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
