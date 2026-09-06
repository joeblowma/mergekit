# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

import json
import logging
import math
import os
import re
import sys
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

import click
import torch
import torch.nn as nn
import tqdm
import transformers
from pydantic import BaseModel

from mergekit.architecture import WeightInfo, arch_info_for_config
from mergekit.card import generate_card_lora
from mergekit.common import ModelReference, dtype_from_name, get_auto_cls
from mergekit.graph import Executor, Task
from mergekit.io.tasks import (
    FinalizeModel,
    LoaderCache,
    LoadTensor,
    SaveTensor,
    TensorWriterTask,
)
from mergekit.io.tensor_writer import TensorWriter
from mergekit.multigpu_executor import MultiGPUExecutor
from mergekit.options import MergeOptions, PrettyPrintHelp, add_merge_options

LOG = logging.getLogger("extract_lora")


@click.command("mergekit-extract-lora", cls=PrettyPrintHelp)
@click.option(
    "--model",
    required=True,
    help="Fine-tuned model path",
)
@click.option(
    "--base-model",
    required=True,
    help="Base model path",
)
@click.option(
    "--out-path",
    required=True,
    help="Output path for extracted LoRA adapter",
)
@click.option(
    "--max-rank",
    type=int,
    default=128,
    help="Maximum rank for LoRA decomposition",
)
@click.option(
    "--distribute-scale/--no-distribute-scale",
    is_flag=True,
    default=True,
    help="Distribute scale between A and B matrices",
)
@click.option(
    "--embed-lora/--no-embed-lora",
    is_flag=True,
    default=False,
    help="Extract LoRA weights for embeddings (vs. in modules_to_save)",
)
@click.option(
    "--save-module",
    "modules_to_save",
    type=str,
    multiple=True,
    default=[],
    help="Save the specified module(s) at full rank",
)
@click.option(
    "--exclude-regex",
    "-e",
    "exclude_regexes",
    type=str,
    multiple=True,
    help="Exclude modules matching the specified regex",
)
@click.option(
    "--include-regex",
    "-i",
    "include_regexes",
    type=str,
    multiple=True,
    help="Include modules matching the specified regex",
)
@click.option(
    "--sv-epsilon",
    type=float,
    default=0,
    help="Threshold for singular values to discard",
    show_default=True,
)
@click.option(
    "--skip-undecomposable",
    is_flag=True,
    help="Skip saving undecomposable modules",
    default=False,
)
@click.option(
    "--low-memory",
    is_flag=True,
    default=False,
    help=(
        "Spool extracted adapter tensors to disk instead of holding them all in RAM "
        "until finalize, keeping memory bounded. Use when RAM/commit charge grows "
        "toward the system limit (RAM + pagefile) while extracting large models. "
        "Requires --safe-serialization; cannot be combined with --low-cpu-memory, "
        "--embed-lora, or --async-write. Note: full float32 SVD is still a "
        "per-matrix memory limit."
    ),
)
@click.option(
    "--low-memory-extreme",
    is_flag=True,
    default=False,
    help=(
        "Lowest RAM/commit mode: implies --low-memory (spools adapter tensors to "
        "disk) AND streams source tensors by reading only the requested tensor's "
        "bytes instead of mmap-ing whole safetensors shards. On Windows the shard "
        "mmap counts toward process private commit, so this bounds source-loading "
        "commit by tensor size instead of shard size. Requires --safe-serialization; "
        "cannot be combined with --low-cpu-memory, --embed-lora, or --async-write. "
        "Note: full float32 SVD is still a per-matrix memory limit."
    ),
)
@click.option(
    "--skip-unchanged-modules",
    is_flag=True,
    default=False,
    help=(
        "Skip full-rank (modules_to_save) tensors whose fine-tuned weights are "
        "byte-identical to the base model. Reads both copies of each candidate "
        "tensor from disk and removes unchanged modules from the adapter and "
        "its modules_to_save config. Respects --low-memory-extreme single-tensor "
        "reading."
    ),
)
@click.option(
    "--auto-embed-lora",
    is_flag=True,
    default=False,
    help=(
        "Adaptively decompose embedding-type modules (input/output embeddings) "
        "into LoRA when that wins on size, instead of saving full copies. "
        "Performs a full float32 SVD of the embedding delta and only writes "
        "lora_embedding_A/B when rank*(vocab+dim) < vocab*dim; otherwise falls "
        "back to a full copy. If the delta is too large for GPU memory the SVD "
        "automatically falls back to CPU (RAM need is ~3x the float32 delta "
        "size, e.g. ~12 GB for a 248k-vocab 9B model). Ignored when "
        "--embed-lora is set."
    ),
)
@click.option(
    "--embed-lora-tolerance",
    type=float,
    default=1e-3,
    help=(
        "Tail-energy tolerance for --auto-embed-lora: the chosen rank keeps "
        "||delta - delta_r||_F <= tolerance * ||delta||_F."
    ),
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help=(
        "Plan the extraction, print a size/time report, and exit without "
        "creating the executor or writing any files. With "
        "--skip-unchanged-modules the actual candidate-tensor reads still occur."
    ),
)
@add_merge_options
def main(
    base_model: str,
    model: str,
    out_path: str,
    max_rank: int,
    distribute_scale: bool,
    embed_lora: bool,
    modules_to_save: List[str],
    exclude_regexes: List[str],
    include_regexes: List[str],
    sv_epsilon: float,
    skip_undecomposable: bool,
    low_memory: bool,
    low_memory_extreme: bool,
    skip_unchanged_modules: bool,
    auto_embed_lora: bool,
    embed_lora_tolerance: float,
    dry_run: bool,
    merge_options: MergeOptions,
):
    merge_options.apply_global_options()

    if low_memory_extreme:
        low_memory = True

    if low_memory and merge_options.low_cpu_memory:
        raise click.UsageError("--low-memory cannot be combined with --low-cpu-memory.")
    if low_memory and embed_lora:
        raise click.UsageError("--low-memory cannot be combined with --embed-lora.")
    if low_memory and not merge_options.safe_serialization:
        raise click.UsageError(
            "--low-memory requires safe serialization; remove --no-safe-serialization."
        )
    if low_memory and merge_options.async_write:
        raise click.UsageError("--low-memory cannot be combined with --async-write.")

    if auto_embed_lora and embed_lora:
        LOG.warning(
            "--embed-lora takes precedence over --auto-embed-lora; "
            "ignoring --auto-embed-lora"
        )
        auto_embed_lora = False

    LoaderCache().setup(merge_options)
    LoaderCache().low_memory_extreme = low_memory_extreme

    modules_to_save = list(modules_to_save)

    base_model_ref = ModelReference.model_validate(base_model)
    model_ref = ModelReference.model_validate(model)
    plan_result = plan_extraction(
        base_model_ref=base_model_ref.merged(
            cache_dir=merge_options.lora_merge_cache,
            trust_remote_code=merge_options.trust_remote_code,
            lora_merge_dtype=merge_options.lora_merge_dtype,
        ),
        model_ref=model_ref.merged(
            cache_dir=merge_options.lora_merge_cache,
            trust_remote_code=merge_options.trust_remote_code,
            lora_merge_dtype=merge_options.lora_merge_dtype,
        ),
        modules_to_save=modules_to_save,
        out_path=out_path,
        options=merge_options,
        max_rank=max_rank,
        distribute_scale=distribute_scale,
        embed_lora=embed_lora,
        exclude_regexes=exclude_regexes,
        include_regexes=include_regexes,
        sv_epsilon=sv_epsilon,
        skip_undecomposable=skip_undecomposable,
        low_memory=low_memory,
        skip_unchanged_modules=skip_unchanged_modules,
        auto_embed_lora=auto_embed_lora,
        embed_lora_tolerance=embed_lora_tolerance,
        dry_run=dry_run,
    )

    if dry_run:
        _print_dry_run_report(
            plan_result, skip_unchanged_modules=skip_unchanged_modules
        )
        return

    tasks = plan_result.tasks
    if merge_options.multi_gpu:
        executor = MultiGPUExecutor(
            tasks, storage_device="cpu" if not merge_options.low_cpu_memory else None
        )
    else:
        executor = Executor(
            tasks,
            math_device=merge_options.device,
            storage_device=(
                merge_options.device if merge_options.low_cpu_memory else "cpu"
            ),
        )

    reset_embed_lora_decisions()
    module_real_ranks = {}
    for task, result in executor.run():
        if isinstance(task, TaskVectorDecompositionTask):
            module_real_ranks[task.weight_info.name.removesuffix(".weight")] = result[
                0
            ].shape[0]

    # Merge runtime adaptive-embedding decisions into the config assembly.
    for base_name, (treatment, rank) in get_embed_lora_decisions().items():
        if treatment == "lora":
            module_real_ranks[base_name] = rank
        if treatment in ("lora", "skipped"):
            _remove_module_from_save_list(modules_to_save, base_name)

    if module_real_ranks:
        real_max_rank = max(module_real_ranks.values())
    else:
        real_max_rank = max_rank
    config_dict = make_config_dict(
        base_ref=base_model_ref,
        max_rank=real_max_rank,
        modules_to_save=modules_to_save,
        target_modules=list(
            set(key.split(".")[-1] for key in module_real_ranks.keys())
        ),
        module_ranks=module_real_ranks,
    )
    with open(os.path.join(out_path, "adapter_config.json"), "w") as f:
        json.dump(config_dict, f, indent=4)

    invocation = " ".join(sys.argv)
    with open(os.path.join(out_path, "README.md"), "w", encoding="utf-8") as f:
        f.write(
            generate_card_lora(
                base_model_ref,
                model_ref,
                invocation,
                os.path.basename(out_path),
                base_vocab_size=plan_result.base_vocab_size,
                final_vocab_size=plan_result.final_vocab_size,
            )
        )

    LOG.info(f"LoRA adapter extracted to {out_path}")


def make_config_dict(
    base_ref: ModelReference,
    max_rank: int,
    modules_to_save: List[str],
    target_modules: List[str],
    module_ranks: Dict[str, int],
):
    different_ranked = {k: v for k, v in module_ranks.items() if v != max_rank}
    return {
        "base_model_name_or_path": base_ref.model.path,
        "peft_type": "LORA",
        "use_rslora": False,
        "target_modules": target_modules,
        "modules_to_save": modules_to_save,
        "task_type": "CAUSAL_LM",
        "r": max_rank,
        "lora_alpha": max_rank,
        "rank_pattern": different_ranked,
        "alpha_pattern": different_ranked,
        "lora_dropout": 0.0,
        "fan_in_fan_out": False,
        "inference_mode": True,
    }


class TaskVectorDecompositionTask(Task[Tuple[torch.Tensor, torch.Tensor]]):
    weight_info: WeightInfo
    input_task: Task
    max_rank: int
    distribute_scale: bool = True
    transpose: bool = False
    sv_epsilon: float = 0

    def arguments(self) -> Dict[str, Any]:
        return {"task_vector": self.input_task}

    def execute(self, task_vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.transpose:
            task_vector = task_vector.T
        out_dtype = task_vector.dtype
        delta = task_vector.to(dtype=torch.float32)
        try:
            u, s, vh = torch.linalg.svd(delta, full_matrices=False)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            LOG.warning(
                "GPU out of memory during SVD of %s; retrying on CPU",
                self.weight_info.name,
            )
            delta = delta.to("cpu")
            u, s, vh = torch.linalg.svd(delta, full_matrices=False)
        del delta
        rank = min(self.max_rank, s.shape[0])
        if self.sv_epsilon > 0:
            rank = min((s > self.sv_epsilon).sum().item(), rank)
        if self.distribute_scale:
            sqrt_s = torch.diag(torch.sqrt(s[:rank]))
            scale_a = sqrt_s
            scale_b = sqrt_s
        else:
            scale_a = torch.diag(s[:rank])
            scale_b = torch.eye(rank)
        sqrt_s = torch.diag(torch.sqrt(s[:rank]))
        weight_a = scale_a @ vh[:rank]
        weight_b = u[:, :rank] @ scale_b

        return weight_a.to(dtype=out_dtype), weight_b.to(dtype=out_dtype)

    def group_label(self) -> Optional[str]:
        return self.input_task.group_label()

    def uses_accelerator(self):
        return True


class TaskVectorTask(Task[torch.Tensor]):
    base_tensor: Task
    model_tensor: Task

    def arguments(self) -> Dict[str, Any]:
        return {"base": self.base_tensor, "model": self.model_tensor}

    def execute(self, base: torch.Tensor, model: torch.Tensor) -> torch.Tensor:
        return model - base

    def group_label(self):
        return max(
            self.base_tensor.group_label() or "", self.model_tensor.group_label() or ""
        )

    def uses_accelerator(self):
        return True


class LoRAModuleSaveTask(Task):
    weight_info: WeightInfo
    writer_task: TensorWriterTask
    model_ref: ModelReference
    decomposition_task: TaskVectorDecompositionTask

    def arguments(self) -> Dict[str, Any]:
        return {"writer": self.writer_task, "decomp": self.decomposition_task}

    def execute(
        self, writer: TensorWriter, decomp: Tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        weight_a, weight_b = decomp
        if weight_a is None or weight_b is None:
            if not self.weight_info.optional:
                raise RuntimeError(
                    f"No SVD decomposition for required weight {self.weight_info.name}"
                )
            return
        lora_type = "lora_embedding" if self.decomposition_task.transpose else "lora"
        lora_suffix = ".weight" if not self.decomposition_task.transpose else ""
        base_name = self.weight_info.name.removesuffix(".weight")
        writer.save_tensor(
            f"base_model.model.{base_name}.{lora_type}_A{lora_suffix}", weight_a
        )
        writer.save_tensor(
            f"base_model.model.{base_name}.{lora_type}_B{lora_suffix}", weight_b
        )

    def priority(self) -> int:
        return 1000

    def group_label(self) -> Optional[str]:
        return self.decomposition_task.group_label()


# Thread-safe registry of runtime adaptive-embedding decisions, populated by
# AdaptiveEmbeddingSaveTask.execute and consumed by main() after executor.run().
# Stores no tensors, so executor last-use reclamation is unaffected.
_embed_lora_decisions: Dict[str, Tuple[str, Optional[int]]] = {}
_embed_lora_lock = threading.Lock()


def reset_embed_lora_decisions() -> None:
    """Clear the adaptive-embedding decision registry (used by tests)."""
    with _embed_lora_lock:
        _embed_lora_decisions.clear()


def record_embed_lora_decision(
    name: str, treatment: str, rank: Optional[int]
) -> None:
    with _embed_lora_lock:
        _embed_lora_decisions[name] = (treatment, rank)


def get_embed_lora_decisions() -> Dict[str, Tuple[str, Optional[int]]]:
    with _embed_lora_lock:
        return dict(_embed_lora_decisions)


# VRAM estimate for a float32 SVD: the delta matrix plus an output factor of
# comparable size plus the gesdd/cuSOLVER workspace. 2.5x the delta is a safe
# upper bound that keeps the GPU fast path for everything that comfortably fits.
_SVD_CUDA_ESTIMATE_FACTOR = 2.5
# Only commit to the GPU when the estimate fits within 90% of currently free
# VRAM, leaving headroom for the caching allocator and co-resident tensors.
_SVD_CUDA_SAFETY_MARGIN = 0.9


def _on_cuda(tensor: torch.Tensor) -> bool:
    """Whether ``tensor`` lives on a CUDA device.

    A one-line seam over ``tensor.is_cuda`` so CPU-only tests can exercise the
    accelerator branch without a real GPU.
    """
    return tensor.is_cuda


def _cuda_free_bytes() -> int:
    """Free bytes on the current CUDA device (0 when CUDA is unavailable)."""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.mem_get_info()[0]


def _svd_fits_on_cuda(
    delta_f32_bytes: int,
    free_memory_source: Optional[Callable[[], int]] = None,
) -> bool:
    """Estimate whether a float32 SVD of ``delta_f32_bytes`` fits in free VRAM.

    ``delta_f32_bytes`` is the byte size of the float32 delta matrix. The SVD
    needs roughly the delta plus a same-sized output factor plus a gesdd
    workspace, estimated here as ``delta_f32_bytes * _SVD_CUDA_ESTIMATE_FACTOR``.
    The estimate must fit within ``_SVD_CUDA_SAFETY_MARGIN`` of free VRAM.
    ``free_memory_source`` injects the free-byte source for tests.
    """
    if not torch.cuda.is_available():
        return False
    if free_memory_source is None:
        free_memory_source = _cuda_free_bytes
    free_bytes = free_memory_source()
    if free_bytes <= 0:
        return False
    required = delta_f32_bytes * _SVD_CUDA_ESTIMATE_FACTOR
    return required <= free_bytes * _SVD_CUDA_SAFETY_MARGIN


def _embedding_delta_svd(
    base: torch.Tensor, model: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(u, s, vh) = SVD(float32(model - base).T)`` with a CPU fallback.

    Uses the GPU when the delta is already on CUDA and the estimated SVD
    footprint fits in free VRAM; otherwise computes on CPU. A CUDA
    out-of-memory error empties the CUDA cache and retries the SVD on CPU.
    The GPU fast path is byte-identical to a plain ``torch.linalg.svd``.
    """
    use_cuda = _on_cuda(model)
    delta_bytes = model.numel() * 4  # torch.float32 is 4 bytes per element

    if use_cuda and _svd_fits_on_cuda(delta_bytes):
        # GPU attempt.
        delta = (model - base).to(dtype=torch.float32)
        try:
            u, s, vh = torch.linalg.svd(delta.T, full_matrices=False)
            del delta
            return u, s, vh
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            LOG.warning("GPU out of memory during embedding SVD; retrying on CPU")
            delta_cpu = delta.to("cpu")
            del delta
    else:
        if use_cuda:
            LOG.info(
                "embedding SVD too large for GPU (need ~%s, free %s), using CPU",
                _human_bytes(delta_bytes * _SVD_CUDA_ESTIMATE_FACTOR),
                _human_bytes(_cuda_free_bytes()),
            )
        # Compute the float32 delta directly on CPU so the large CUDA input
        # copies are never duplicated.
        delta_cpu = (model.to("cpu") - base.to("cpu")).to(dtype=torch.float32)

    u, s, vh = torch.linalg.svd(delta_cpu.T, full_matrices=False)
    del delta_cpu
    return u, s, vh


class AdaptiveEmbeddingSaveTask(Task):
    """Decompose an embedding delta into LoRA when it wins on size.

    Loads base + finetuned tensors, computes delta = model - base, runs a full
    float32 SVD, and selects the smallest rank r with tail-energy ratio
    ``||delta - delta_r||_F <= tolerance * ||delta||_F``. If ``r * (m + n) < m * n``
    (element counts), writes lora_embedding_A/B; otherwise writes a full copy of
    the finetuned tensor. A zero delta writes nothing.
    """

    weight_info: WeightInfo
    writer_task: TensorWriterTask
    base_tensor: Task
    model_tensor: Task
    tolerance: float = 1e-3

    def arguments(self) -> Dict[str, Any]:
        return {
            "writer": self.writer_task,
            "base": self.base_tensor,
            "model": self.model_tensor,
        }

    def execute(
        self, writer: TensorWriter, base: Optional[torch.Tensor], model: torch.Tensor
    ) -> None:
        name = self.weight_info.name
        base_name = name.removesuffix(".weight")

        if base is None or model is None:
            if model is not None:
                LOG.info("%s: base tensor missing, saving full copy", base_name)
                writer.save_tensor(f"base_model.model.{name}", model)
                record_embed_lora_decision(base_name, "full", None)
            else:
                LOG.info("%s: model tensor missing, omitting from adapter", base_name)
                record_embed_lora_decision(base_name, "skipped", 0)
            return

        out_dtype = model.dtype
        m, n = model.shape
        # SVD on delta.T to match LoRAModuleSaveTask's transpose branch
        # (lora_embedding_A is [r, vocab], lora_embedding_B is [hidden, r]).
        # Falls back to CPU when the float32 delta won't fit on the GPU.
        u, s, vh = _embedding_delta_svd(base, model)
        del base
        rank = _select_adaptive_rank(s, m, n, self.tolerance)
        if rank == 0:
            LOG.info("%s: delta is zero, omitting from adapter", base_name)
            record_embed_lora_decision(base_name, "skipped", 0)
            return
        if rank < 0:
            needed_rank = _needed_rank_for_tolerance(s, self.tolerance)
            breakeven = m * n // (m + n) + 1
            LOG.info(
                "%s: full copy (tolerance=%.1e needs rank %d, "
                "LoRA breakeven rank %d)",
                base_name, self.tolerance, needed_rank, breakeven,
            )
            writer.save_tensor(f"base_model.model.{name}", model)
            record_embed_lora_decision(base_name, "full", None)
            return

        energy_frac = float((s[:rank] ** 2).sum() / (s ** 2).sum())
        LOG.info(
            "%s: adaptive LoRA rank=%d (tolerance=%.1e, energy=%.6f)",
            base_name, rank, self.tolerance, energy_frac,
        )
        # Same naming/scaling convention as LoRAModuleSaveTask's transpose
        # branch (distribute_scale=True): sqrt(S) split into both factors.
        sqrt_s = torch.diag(torch.sqrt(s[:rank]))
        weight_a = sqrt_s @ vh[:rank]
        weight_b = u[:, :rank] @ sqrt_s
        writer.save_tensor(
            f"base_model.model.{base_name}.lora_embedding_A",
            weight_a.to(dtype=out_dtype),
        )
        writer.save_tensor(
            f"base_model.model.{base_name}.lora_embedding_B",
            weight_b.to(dtype=out_dtype),
        )
        record_embed_lora_decision(base_name, "lora", rank)

    def priority(self) -> int:
        return 1000

    def group_label(self) -> Optional[str]:
        return max(
            self.base_tensor.group_label() or "", self.model_tensor.group_label() or ""
        )

    def uses_accelerator(self):
        return True


def _needed_rank_for_tolerance(s: torch.Tensor, tolerance: float) -> int:
    """Smallest rank r with tail-energy ≤ tolerance²·total.

    Returns 0 for a zero delta, or the rank (possibly ``full_rank`` if the
    tolerance is not met by any prefix).  Singular values ``s`` must be in
    descending order.
    """
    s2 = s * s
    total = s2.sum()
    if total.item() == 0.0:
        return 0
    full_rank = s.shape[0]
    tol2 = tolerance * tolerance
    cum = torch.cumsum(s2, dim=0)
    for r in range(1, full_rank + 1):
        tail_ratio2 = (total - cum[r - 1]) / total
        if tail_ratio2.item() <= tol2:
            return r
    return full_rank


def _select_adaptive_rank(
    s: torch.Tensor, m: int, n: int, tolerance: float
) -> int:
    """Pick the adaptive LoRA rank for an embedding delta.

    Returns 0 for a zero delta, -1 when a full copy wins on size, and the chosen
    rank otherwise. Singular values ``s`` must be in descending order.
    """
    needed = _needed_rank_for_tolerance(s, tolerance)
    if needed == 0:
        return 0
    full_rank = s.shape[0]
    if needed < full_rank and needed * (m + n) < m * n:
        return needed
    return -1


def _wi_load(model_ref: ModelReference, weight_info: WeightInfo) -> LoadTensor:
    return LoadTensor(
        model=model_ref,
        tensor=weight_info.name,
        dtype=weight_info.force_dtype,
        optional=weight_info.optional,
        aliases=weight_info.aliases,
        tied_names=weight_info.tied_names,
    )


def _make_dummy_model(
    model_ref: ModelReference, trust_remote_code: bool = False
) -> transformers.PreTrainedModel:
    model_cfg = transformers.AutoConfig.from_pretrained(
        model_ref.model.path,
        revision=model_ref.model.revision,
        trust_remote_code=trust_remote_code,
    )
    auto_cls = get_auto_cls(model_cfg.architectures[0])
    with torch.device("meta"):
        res = auto_cls.from_config(model_cfg, trust_remote_code=trust_remote_code)
    return res


class PlanResults(BaseModel):
    tasks: List[Task]
    base_vocab_size: int
    final_vocab_size: int
    module_breakdown: List["ModulePlanEntry"] = []
    estimated_tensor_bytes: int = 0
    estimated_tensor_count: int = 0
    skipped_modules: List["SkippedModuleInfo"] = []
    expected_final_size: Optional[int] = None


class ModulePlanEntry(BaseModel):
    """Per-module breakdown entry for the dry-run report."""

    name: str
    treatment: str  # "lora" | "full" | "skipped_unchanged" | "adaptive"
    shape: Tuple[int, ...] = ()
    est_bytes: int = 0
    rank: Optional[int] = None
    lora_bytes: Optional[int] = None
    full_bytes: Optional[int] = None


class SkippedModuleInfo(BaseModel):
    name: str
    bytes_saved: int


def _tensor_output_size(tensor: torch.Tensor, weight_info: WeightInfo) -> int:
    dtype = (
        dtype_from_name(weight_info.force_dtype)
        if weight_info.force_dtype
        else tensor.dtype
    )
    return tensor.numel() * torch.empty((), dtype=dtype).element_size()


def _estimate_full_module_size(
    module: nn.Module, wi: WeightInfo, bias_wi: Optional[WeightInfo]
) -> Tuple[int, int]:
    size = _tensor_output_size(module.weight, wi)
    count = 1
    if bias_wi is not None and getattr(module, "bias", None) is not None:
        size += _tensor_output_size(module.bias, bias_wi)
        count += 1
    return size, count


def _estimate_lora_module_size(
    module: nn.Module,
    wi: WeightInfo,
    bias_wi: Optional[WeightInfo],
    max_rank: int,
) -> Tuple[int, int]:
    shape = module.weight.shape
    # LoRA treats convolution kernels as an out_features by flattened
    # in_features matrix. This is also the usual two-dimensional shape for
    # Linear and Embedding weights, so the estimate covers every supported
    # module type without dropping convolution kernel dimensions.
    out_features = shape[0]
    in_features = math.prod(shape[1:])
    rank = min(max_rank, out_features, in_features)
    size = (
        rank
        * (out_features + in_features)
        * (_tensor_output_size(module.weight, wi) // module.weight.numel())
    )
    count = 2
    if bias_wi is not None and getattr(module, "bias", None) is not None:
        size += _tensor_output_size(module.bias, bias_wi)
        count += 1
    return size, count


def _resolve_tensor_name(loader, wi: WeightInfo) -> Optional[str]:
    """Resolve a WeightInfo to the actual tensor name present in a loader index."""
    all_names = [wi.name] + list(wi.aliases or []) + list(wi.tied_names or [])
    for name in all_names:
        if name in loader.index.tensor_paths:
            return name
    return None


def _remove_module_from_save_list(modules_to_save: List[str], name: str) -> None:
    """Remove a module (by full name and leaf name) from modules_to_save in place."""
    leaf = name.split(".")[-1]
    modules_to_save[:] = [m for m in modules_to_save if m not in (name, leaf)]


def _compare_unchanged(
    base_model_ref: ModelReference,
    model_ref: ModelReference,
    wi: WeightInfo,
    name: str,
    ft_shape: Tuple[int, ...],
    base_shape: Optional[Tuple[int, ...]],
    cache: Dict[Tuple[str, str], Tuple[bool, int]],
) -> Tuple[bool, int]:
    """Compare base vs finetuned weight byte-exactly.

    Returns ``(should_skip, bytes_saved)``. Tied weights (multiple module names
    resolving to the same underlying tensor) are compared once via ``cache`` keyed
    on the resolved base/model tensor names.
    """
    if base_shape is None:
        LOG.info(f"{name}: no matching base tensor; keeping full copy")
        return False, 0
    if tuple(ft_shape) != tuple(base_shape):
        LOG.info(
            f"{name}: shape mismatch {tuple(base_shape)} -> {tuple(ft_shape)}; "
            "keeping full copy"
        )
        return False, 0

    base_loader = LoaderCache().get(base_model_ref)
    model_loader = LoaderCache().get(model_ref)
    base_name = _resolve_tensor_name(base_loader, wi)
    model_name = _resolve_tensor_name(model_loader, wi)
    if base_name is None or model_name is None:
        LOG.info(f"{name}: tensor missing on one side; keeping full copy")
        return False, 0

    key = (base_name, model_name)
    if key in cache:
        return cache[key]

    base_tensor = base_loader.get_tensor(base_name, device="cpu")
    model_tensor = model_loader.get_tensor(model_name, device="cpu")
    try:
        equal = (
            base_tensor.shape == model_tensor.shape
            and torch.equal(base_tensor, model_tensor)
        )
        bytes_saved = model_tensor.numel() * model_tensor.element_size()
    finally:
        del base_tensor, model_tensor

    result = (equal, bytes_saved)
    cache[key] = result
    return result


def plan_extraction(
    base_model_ref: ModelReference,
    model_ref: ModelReference,
    modules_to_save: List[str],
    out_path: str,
    options: MergeOptions,
    max_rank: int,
    distribute_scale: bool = True,
    embed_lora: bool = False,
    exclude_regexes: Optional[List[str]] = None,
    include_regexes: Optional[List[str]] = None,
    sv_epsilon: float = 0,
    skip_undecomposable: bool = False,
    low_memory: bool = False,
    skip_unchanged_modules: bool = False,
    auto_embed_lora: bool = False,
    embed_lora_tolerance: float = 1e-3,
    dry_run: bool = False,
) -> PlanResults:
    targets = []
    module_plans = []
    module_breakdown = []
    skipped_modules = []
    estimated_tensor_bytes = 0
    estimated_tensor_count = 0
    estimate = low_memory or dry_run

    name_to_wi = all_weights_map(model_ref, options)
    dummy_base = _make_dummy_model(base_model_ref, options.trust_remote_code)
    dummy_model = _make_dummy_model(model_ref, options.trust_remote_code)

    embed_in = dummy_model.get_input_embeddings()
    embed_out = dummy_model.get_output_embeddings()

    ft_vocab = embed_in.weight.shape[0]
    base_vocab = dummy_base.get_input_embeddings().weight.shape[0]
    if ft_vocab != base_vocab and embed_lora:
        LOG.warning(
            f"Vocabulary size mismatch: fine-tuned model has {ft_vocab} tokens, base model has {base_vocab} tokens"
        )
        LOG.warning("Enforcing embeddings in modules_to_save, embed_lora=False")
        embed_lora = False
    if ft_vocab != base_vocab and auto_embed_lora:
        LOG.warning(
            "Vocabulary size mismatch: disabling --auto-embed-lora so embeddings "
            "are saved at full rank"
        )
        auto_embed_lora = False

    base_shape_map = {}
    for _name, _module in dummy_base.named_modules():
        if hasattr(_module, "weight"):
            base_shape_map[_name] = tuple(_module.weight.shape)
    del dummy_base

    warned_modules = set()
    skip_cache: Dict[Tuple[str, str], Tuple[bool, int]] = {}

    def _should_extract(name: str) -> bool:
        if include_regexes and not any(re.search(r, name) for r in include_regexes):
            return False
        if any(re.search(r, name) for r in exclude_regexes):
            return False
        return True

    def _record_full_save(
        name: str,
        module: nn.Module,
        wi: WeightInfo,
        bias_wi: Optional[WeightInfo],
        ft_shape: Tuple[int, ...],
        is_embedding: bool,
    ) -> None:
        """Handle a full-rank save decision (skip-unchanged / adaptive / full)."""
        nonlocal estimated_tensor_bytes, estimated_tensor_count

        # Feature A: skip byte-identical full-rank modules.
        if skip_unchanged_modules:
            should_skip, bytes_saved = _compare_unchanged(
                base_model_ref,
                model_ref,
                wi,
                name,
                ft_shape,
                base_shape_map.get(name),
                skip_cache,
            )
            if should_skip:
                LOG.info(
                    f"{name} unchanged vs base; skipping ({bytes_saved} bytes saved)"
                )
                _remove_module_from_save_list(modules_to_save, name)
                skipped_modules.append(
                    SkippedModuleInfo(name=name, bytes_saved=bytes_saved)
                )
                if dry_run:
                    module_breakdown.append(
                        ModulePlanEntry(
                            name=name,
                            treatment="skipped_unchanged",
                            shape=ft_shape,
                            est_bytes=0,
                        )
                    )
                return

        # Feature B: adaptively decompose embeddings instead of a full copy.
        if (
            auto_embed_lora
            and is_embedding
            and bias_wi is None
            and base_shape_map.get(name) == ft_shape
        ):
            size_full, count_full = _estimate_full_module_size(module, wi, bias_wi)
            size_lora, _ = _estimate_lora_module_size(module, wi, None, max_rank)
            rank_bound = min(max_rank, ft_shape[0], math.prod(ft_shape[1:]))
            if estimate:
                estimated_tensor_bytes += size_full
                estimated_tensor_count += count_full
            if dry_run:
                module_breakdown.append(
                    ModulePlanEntry(
                        name=name,
                        treatment="adaptive",
                        shape=ft_shape,
                        est_bytes=size_full,
                        rank=rank_bound,
                        lora_bytes=size_lora,
                        full_bytes=size_full,
                    )
                )
            module_plans.append(("adaptive", wi, bias_wi))
            return

        LOG.info(f"Planning to save {name} at full rank")
        if estimate:
            size, count = _estimate_full_module_size(module, wi, bias_wi)
            estimated_tensor_bytes += size
            estimated_tensor_count += count
            if dry_run:
                module_breakdown.append(
                    ModulePlanEntry(
                        name=name,
                        treatment="full",
                        shape=ft_shape,
                        est_bytes=size,
                        full_bytes=size,
                    )
                )
        module_plans.append(("save", wi, bias_wi))

    for name, module in tqdm.tqdm(
        list(dummy_model.named_modules()), desc="Planning operations"
    ):
        wi = name_to_wi.get(name + ".weight")
        bias_wi = name_to_wi.get(name + ".bias")
        if wi is None:
            if hasattr(module, "weight"):
                LOG.warning(
                    f"Weight {name} present in model but not in architecture info"
                )
                wi = WeightInfo(
                    name=name + ".weight",
                    optional=True,
                    is_embed=isinstance(module, nn.Embedding),
                )
            else:
                continue

        is_embedding = (
            module == embed_in
            or module == embed_out
            or isinstance(module, nn.Embedding)
        )

        if (
            (not embed_lora)
            and is_embedding
            and not any(re.search(r, name) for r in exclude_regexes or [])
        ):
            # If embeddings are not explicitly excluded but embed_lora is False,
            # save them at full rank instead of decomposing
            key = name.split(".")[-1]
            if key not in modules_to_save:
                LOG.warning(f"Adding {key} to modules_to_save")
                modules_to_save.append(key)

        if name in modules_to_save or (name.split(".")[-1] in modules_to_save):
            _record_full_save(
                name, module, wi, bias_wi, tuple(module.weight.shape), is_embedding
            )
        elif _should_extract(name):
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Embedding)):
                LOG.info(f"Planning LoRA extraction for {name}")
                if estimate:
                    size, count = _estimate_lora_module_size(
                        module, wi, bias_wi, max_rank
                    )
                    estimated_tensor_bytes += size
                    estimated_tensor_count += count
                    if dry_run:
                        shape = tuple(module.weight.shape)
                        module_breakdown.append(
                            ModulePlanEntry(
                                name=name,
                                treatment="lora",
                                shape=shape,
                                est_bytes=size,
                                rank=min(max_rank, shape[0], math.prod(shape[1:])),
                            )
                        )
                module_plans.append(
                    ("lora", wi, bias_wi, isinstance(module, nn.Embedding))
                )
            else:
                key = name.split(".")[-1]
                message = (
                    f"{key} has unsupported module type {type(module).__name__} - "
                    + ("skipping" if skip_undecomposable else "saving at full rank")
                )
                if not skip_undecomposable:
                    # into modules_to_save it goes
                    if key not in modules_to_save:
                        modules_to_save.append(key)
                    _record_full_save(
                        name,
                        module,
                        wi,
                        bias_wi,
                        tuple(module.weight.shape),
                        is_embedding,
                    )
                if key not in warned_modules:
                    LOG.warning(message)
                    warned_modules.add(key)

    expected_final_size = None
    if low_memory or dry_run:
        # Safetensors metadata is small but non-zero; deliberately overestimate it
        # so TensorWriter's staging-space preflight has a conservative bound.
        expected_final_size = max(
            1, estimated_tensor_bytes + 4096 + (estimated_tensor_count * 1024)
        )

    writer_task = TensorWriterTask(
        out_path=out_path,
        override_basename="adapter_model",
        max_shard_size=-1,
        safe_serialization=options.safe_serialization,
        use_async=options.async_write,
        max_write_threads=options.write_threads,
        disk_spool=low_memory,
        expected_final_size=expected_final_size,
    )
    for module_plan in module_plans:
        if module_plan[0] == "save":
            _, wi, bias_wi = module_plan
            targets.extend(plan_module_to_save(model_ref, writer_task, wi, bias_wi))
        elif module_plan[0] == "adaptive":
            _, wi, _bias_wi = module_plan
            targets.extend(
                plan_adaptive_embedding(
                    base_model_ref,
                    model_ref,
                    wi,
                    writer_task,
                    embed_lora_tolerance,
                )
            )
        else:
            _, wi, bias_wi, transpose = module_plan
            targets.extend(
                plan_lora_module(
                    base_model_ref,
                    model_ref,
                    wi,
                    bias_wi,
                    writer_task,
                    max_rank,
                    distribute_scale,
                    transpose=transpose,
                    sv_epsilon=sv_epsilon,
                )
            )

    save_tasks = [
        t
        for t in targets
        if isinstance(t, (SaveTensor, LoRAModuleSaveTask, AdaptiveEmbeddingSaveTask))
    ]
    finalize = FinalizeModel(tensor_save_tasks=save_tasks, writer_task=writer_task)
    return PlanResults(
        tasks=targets + [finalize],
        base_vocab_size=base_vocab,
        final_vocab_size=ft_vocab,
        module_breakdown=module_breakdown,
        estimated_tensor_bytes=estimated_tensor_bytes,
        estimated_tensor_count=estimated_tensor_count,
        skipped_modules=skipped_modules,
        expected_final_size=expected_final_size,
    )


def plan_lora_module(
    base_model_ref: ModelReference,
    model_ref: ModelReference,
    wi: WeightInfo,
    bias_wi: Optional[WeightInfo],
    writer_task: TensorWriterTask,
    max_rank: int,
    distribute_scale: bool = True,
    transpose: bool = False,
    sv_epsilon: float = 0,
) -> List[Task]:
    targets = []
    base_load_task = _wi_load(base_model_ref, wi)
    model_load_task = _wi_load(model_ref, wi)
    tv_task = TaskVectorTask(base_tensor=base_load_task, model_tensor=model_load_task)
    decomp_task = TaskVectorDecompositionTask(
        weight_info=wi,
        input_task=tv_task,
        max_rank=max_rank,
        distribute_scale=distribute_scale,
        transpose=transpose,
        sv_epsilon=sv_epsilon,
    )
    targets.append(decomp_task)
    targets.append(
        LoRAModuleSaveTask(
            weight_info=wi,
            writer_task=writer_task,
            model_ref=model_ref,
            decomposition_task=decomp_task,
        )
    )
    if bias_wi is not None:
        base_bias_load_task = _wi_load(base_model_ref, bias_wi)
        model_bias_load_task = _wi_load(model_ref, bias_wi)
        tv_bias_task = TaskVectorTask(
            base_tensor=base_bias_load_task, model_tensor=model_bias_load_task
        )
        base_bias_name = bias_wi.name.removesuffix(".bias")
        name_out = f"base_model.model.{base_bias_name}.lora_B.bias"
        targets.append(
            SaveTensor(
                tensor_name=name_out,
                tensor_task=tv_bias_task,
                writer_task=writer_task,
                optional=bias_wi.optional,
                clone=False,
            )
        )
    return targets


def plan_module_to_save(
    model_ref: ModelReference,
    writer_task: TensorWriterTask,
    wi: WeightInfo,
    bias_wi: Optional[WeightInfo],
):
    save_tasks = []
    load_task = _wi_load(model_ref, wi)
    save_task = SaveTensor(
        tensor_name=f"base_model.model.{wi.name}",
        tensor_task=load_task,
        writer_task=writer_task,
        optional=wi.optional,
        clone=False,
    )
    save_tasks.append(save_task)
    if bias_wi is not None:
        bias_load_task = _wi_load(model_ref, bias_wi)
        bias_save_task = SaveTensor(
            tensor_name=f"base_model.model.{bias_wi.name}",
            tensor_task=bias_load_task,
            writer_task=writer_task,
            optional=bias_wi.optional,
            clone=False,
        )
        save_tasks.append(bias_save_task)
    return save_tasks


def plan_adaptive_embedding(
    base_model_ref: ModelReference,
    model_ref: ModelReference,
    wi: WeightInfo,
    writer_task: TensorWriterTask,
    tolerance: float = 1e-3,
) -> List[Task]:
    base_load_task = _wi_load(base_model_ref, wi)
    model_load_task = _wi_load(model_ref, wi)
    task = AdaptiveEmbeddingSaveTask(
        weight_info=wi,
        writer_task=writer_task,
        base_tensor=base_load_task,
        model_tensor=model_load_task,
        tolerance=tolerance,
    )
    return [task]


def _human_bytes(n: int) -> str:
    if n >= 1024**3:
        return f"{n / 1024**3:.2f} GB"
    if n >= 1024**2:
        return f"{n / 1024**2:.2f} MB"
    if n >= 1024:
        return f"{n / 1024:.2f} KB"
    return f"{n} B"


def _print_dry_run_report(
    plan_result: PlanResults, skip_unchanged_modules: bool = False
) -> None:
    click.echo("=" * 96)
    click.echo("mergekit-extract-lora dry-run report")
    click.echo("=" * 96)
    click.echo(f"{'Module':<40} {'Treatment':<32} {'Shape':<16} {'Est. bytes':>12}")
    click.echo("-" * 96)
    for entry in plan_result.module_breakdown:
        if entry.treatment == "lora":
            treatment = f"LoRA rank<={entry.rank}"
        elif entry.treatment == "full":
            treatment = "full-save"
        elif entry.treatment == "skipped_unchanged":
            treatment = "skipped-unchanged"
        elif entry.treatment == "adaptive":
            treatment = (
                f"adaptive [best: {_human_bytes(entry.lora_bytes or 0)}, "
                f"worst: {_human_bytes(entry.full_bytes or 0)}]"
            )
        else:
            treatment = entry.treatment
        shape_str = str(list(entry.shape)) if entry.shape else "-"
        click.echo(
            f"{entry.name:<40} {treatment:<32} {shape_str:<16} {entry.est_bytes:>12}"
        )
    click.echo("-" * 96)
    click.echo(f"Planned tensor count: {plan_result.estimated_tensor_count}")
    click.echo(
        f"Planned tensor bytes (raw): {plan_result.estimated_tensor_bytes} "
        f"({_human_bytes(plan_result.estimated_tensor_bytes)})"
    )
    total = plan_result.expected_final_size or 0
    click.echo(
        f"Estimated adapter bytes (incl. metadata overhead): {total} "
        f"({_human_bytes(total)})"
    )
    if skip_unchanged_modules and plan_result.skipped_modules:
        saved_total = sum(s.bytes_saved for s in plan_result.skipped_modules)
        click.echo("Skipped (unchanged) modules:")
        for skipped in plan_result.skipped_modules:
            click.echo(
                f"  - {skipped.name}: {_human_bytes(skipped.bytes_saved)} saved"
            )
        click.echo(
            f"Total bytes saved by skipping: {saved_total} "
            f"({_human_bytes(saved_total)})"
        )
    est_io = 2 * plan_result.estimated_tensor_bytes + total
    seconds = est_io / (100 * 1024 * 1024)
    click.echo(
        f"ROUGH time estimate: ~{seconds:.1f}s assuming 100 MB/s sustained I/O "
        f"(est. I/O bytes ~2x planned tensor bytes + output = {_human_bytes(est_io)})"
    )


def all_weights_map(
    model_ref: ModelReference, options: MergeOptions
) -> Dict[str, WeightInfo]:
    name_to_wi = {}
    model_cfg = model_ref.config(trust_remote_code=options.trust_remote_code)
    arch_info = arch_info_for_config(model_cfg)
    for wi in arch_info.all_weights(model_cfg):
        name_to_wi[wi.name] = wi
    return name_to_wi


if __name__ == "__main__":
    main()
