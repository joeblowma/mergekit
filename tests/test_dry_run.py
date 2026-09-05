"""Tests for the --dry-run flag in mergekit-extract-lora.

Covers: exit code 0 with an empty out_path, a human-readable report with
estimates, estimates computed without --low-memory, and an estimated size that
predicts the actual tiny-run adapter within +/-5%.
"""

import json
import os
import shutil

import pytest
import safetensors.torch
import torch
from click.testing import CliRunner

from mergekit.common import ModelReference
from mergekit.options import MergeOptions
from mergekit.scripts import extract_lora


def _make_tiny_models(tmp_path):
    """Base + finetuned picollama where projection layers differ."""
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


class TestDryRunCli:
    def test_exit_zero_and_out_path_empty(self, tmp_path):
        base, ft = _make_tiny_models(tmp_path)
        out = str(tmp_path / "adapter")
        os.makedirs(out, exist_ok=True)

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
                "--dry-run",
            ],
        )
        assert result.exit_code == 0, f"{result.output}\n{result.exception}"
        assert os.listdir(out) == []

    def test_report_contains_estimates(self, tmp_path):
        base, ft = _make_tiny_models(tmp_path)
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
                "--dry-run",
                "--skip-unchanged-modules",
            ],
        )
        assert result.exit_code == 0, f"{result.output}\n{result.exception}"
        output = result.output
        assert "dry-run report" in output
        assert "Estimated adapter bytes" in output
        assert "ROUGH time estimate" in output
        assert "100 MB/s" in output
        assert "skipped-unchanged" in output


class TestDryRunEstimates:
    def test_estimates_computed_without_low_memory(self, tmp_path):
        base, ft = _make_tiny_models(tmp_path)
        plan = extract_lora.plan_extraction(
            base_model_ref=ModelReference(model=base),
            model_ref=ModelReference(model=ft),
            modules_to_save=[],
            out_path=str(tmp_path / "adapter"),
            options=MergeOptions(),
            max_rank=2,
            exclude_regexes=[],
            include_regexes=[],
            low_memory=False,
            dry_run=True,
        )
        assert plan.estimated_tensor_bytes > 0
        assert plan.estimated_tensor_count > 0
        assert plan.expected_final_size is not None
        assert plan.module_breakdown
        treatments = {e.treatment for e in plan.module_breakdown}
        assert "lora" in treatments
        assert "full" in treatments

    def test_estimated_size_within_five_percent_of_actual(self, tmp_path):
        base, ft = _make_tiny_models(tmp_path)
        plan = extract_lora.plan_extraction(
            base_model_ref=ModelReference(model=base),
            model_ref=ModelReference(model=ft),
            modules_to_save=[],
            out_path=str(tmp_path / "adapter"),
            options=MergeOptions(),
            max_rank=2,
            exclude_regexes=[],
            include_regexes=[],
            low_memory=False,
            dry_run=True,
        )
        estimated = plan.estimated_tensor_bytes

        out = str(tmp_path / "actual")
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
        actual = sum(t.numel() * t.element_size() for t in adapter.values())

        assert actual > 0
        assert abs(estimated - actual) <= 0.05 * actual

    def test_dry_run_config_not_written(self, tmp_path):
        """Dry-run must not write adapter_config.json or README.md."""
        base, ft = _make_tiny_models(tmp_path)
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
                "--dry-run",
            ],
        )
        assert result.exit_code == 0, f"{result.output}\n{result.exception}"
        assert not os.path.exists(out) or os.listdir(out) == []
