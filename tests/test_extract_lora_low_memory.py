"""Regression tests for the extractor-only low-memory option.

These tests deliberately stub model planning and execution.  The option and
guard behaviour should be testable without a model download, while the
bounded-memory writer itself is covered in ``test_tensor_writer_low_memory``.
"""

from types import SimpleNamespace

import pytest
import torch
from click.testing import CliRunner

from mergekit.architecture import WeightInfo
from mergekit.common import ModelReference
from mergekit.io import tasks as io_tasks
from mergekit.options import MergeOptions
from mergekit.scripts import extract_lora, merge_raw_pytorch


def _error_text(result) -> str:
    """Return Click output and the caught exception's text for assertions."""

    return f"{result.output}\n{result.exception}".lower()


def _invoke_with_low_memory(runner: CliRunner, out_path, conflict: str):
    return runner.invoke(
        extract_lora.main,
        [
            "--base-model",
            "local-base",
            "--model",
            "local-finetuned",
            "--out-path",
            str(out_path),
            "--low-memory",
            conflict,
        ],
    )


@pytest.fixture
def stubbed_extraction(monkeypatch, tmp_path):
    """Replace model planning/execution with a deterministic one-item run."""

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


def test_low_memory_is_an_extractor_only_option_and_not_merge_option():
    """The flag belongs to extract-lora, not the global option model."""

    assert "low_memory" not in MergeOptions.model_fields
    assert not hasattr(MergeOptions, "low_memory")
    assert "low_memory" not in MergeOptions.__annotations__

    runner = CliRunner()
    extractor_help = runner.invoke(extract_lora.main, ["--help"])
    assert extractor_help.exit_code == 0, extractor_help.output
    assert "--low-memory" in extractor_help.output

    global_help = runner.invoke(merge_raw_pytorch.main, ["--help"])
    assert global_help.exit_code == 0, global_help.output
    assert "--low-memory" not in global_help.output


@pytest.mark.parametrize(
    "conflict, expected_name",
    [
        ("--low-cpu-memory", "low-cpu-memory"),
        ("--embed-lora", "embed-lora"),
        ("--no-safe-serialization", "safe-serialization"),
        ("--async-write", "async-write"),
    ],
)
def test_low_memory_rejects_unsupported_modes(
    tmp_path, conflict: str, expected_name: str
):
    """Each non-spooling mode is rejected before any model access."""

    result = _invoke_with_low_memory(CliRunner(), tmp_path / "adapter", conflict)

    assert result.exit_code != 0
    text = _error_text(result)
    assert "low-memory" in text
    assert expected_name in text or expected_name.replace("-", "_") in text


def test_low_memory_is_forwarded_to_extraction_plan(stubbed_extraction):
    """The opt-in flag reaches planning without being put in MergeOptions."""

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
            "--low-memory",
        ],
    )

    assert result.exit_code == 0, _error_text(result)
    # ``low_memory`` is the intended plan-level spelling.  ``disk_spool`` is
    # accepted as a narrow compatibility variation if the owner keeps the
    # implementation detail at the plan boundary.
    low_memory = plan_calls.get("low_memory", plan_calls.get("disk_spool", False))
    assert low_memory is True


def test_extractor_initializes_loader_cache_and_forwards_lazy_unpickle(
    monkeypatch, stubbed_extraction
):
    """Extractor setup must preserve the existing lazy-unpickle option."""

    out_path, _ = stubbed_extraction
    setup_calls = []

    def record_setup(cache, options):
        setup_calls.append(options)

    monkeypatch.setattr(io_tasks.LoaderCache, "setup", record_setup)
    result = CliRunner().invoke(
        extract_lora.main,
        [
            "--base-model",
            "local-base",
            "--model",
            "local-finetuned",
            "--out-path",
            str(out_path),
            "--lazy-unpickle",
        ],
    )

    assert result.exit_code == 0, _error_text(result)
    assert len(setup_calls) == 1
    assert isinstance(setup_calls[0], MergeOptions)
    assert setup_calls[0].lazy_unpickle is True


def test_default_extraction_does_not_enable_low_memory(stubbed_extraction):
    """Omitting the flag keeps the legacy extraction mode disabled."""

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
        ],
    )

    assert result.exit_code == 0, _error_text(result)
    assert plan_calls.get("low_memory", plan_calls.get("disk_spool", False)) is False


def test_low_memory_plan_builds_spooled_lora_graph_without_download(
    monkeypatch, tmp_path
):
    """Plan a real small LoRA graph without loading either model from disk."""

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(4, 3)
            self.linear = torch.nn.Linear(3, 2)

        def get_input_embeddings(self):
            return self.embed_tokens

        def get_output_embeddings(self):
            return None

    weights = {
        "embed_tokens.weight": WeightInfo(name="embed_tokens.weight", is_embed=True),
        "linear.weight": WeightInfo(name="linear.weight"),
        "linear.bias": WeightInfo(name="linear.bias"),
    }
    monkeypatch.setattr(
        extract_lora,
        "_make_dummy_model",
        lambda _model_ref, _trust_remote_code=False: DummyModel(),
    )
    monkeypatch.setattr(
        extract_lora,
        "all_weights_map",
        lambda _model_ref, _options: weights,
    )

    writer_constructions = []
    real_writer_task = io_tasks.TensorWriterTask

    def capture_writer_task(**kwargs):
        writer_constructions.append(kwargs)
        return real_writer_task(**kwargs)

    monkeypatch.setattr(extract_lora, "TensorWriterTask", capture_writer_task)

    plan = extract_lora.plan_extraction(
        base_model_ref=ModelReference(model="local-base"),
        model_ref=ModelReference(model="local-finetuned"),
        modules_to_save=[],
        out_path=str(tmp_path / "adapter"),
        options=MergeOptions(),
        max_rank=2,
        exclude_regexes=[],
        include_regexes=[],
        low_memory=True,
    )

    assert len(writer_constructions) == 1
    writer_kwargs = writer_constructions[0]
    assert writer_kwargs["disk_spool"] is True
    assert writer_kwargs["expected_final_size"] > 0

    save_tasks = [
        task
        for task in plan.tasks
        if isinstance(task, (io_tasks.SaveTensor, extract_lora.LoRAModuleSaveTask))
    ]
    assert save_tasks
    writer_task = save_tasks[0].writer_task
    assert isinstance(writer_task, io_tasks.TensorWriterTask)
    assert writer_task.disk_spool is True
    assert writer_task.expected_final_size == writer_kwargs["expected_final_size"]

    def same_task(left, right):
        return type(left) is type(right) and left == right

    assert all(same_task(task.writer_task, writer_task) for task in save_tasks)

    finalize = next(
        task for task in plan.tasks if isinstance(task, io_tasks.FinalizeModel)
    )
    assert same_task(finalize.writer_task, writer_task)
    assert len(finalize.tensor_save_tasks) == len(save_tasks)
    assert all(
        any(same_task(save, dependency) for dependency in finalize.tensor_save_tasks)
        for save in save_tasks
    )

    lora_save = next(
        task for task in plan.tasks if isinstance(task, extract_lora.LoRAModuleSaveTask)
    )
    decomposition = lora_save.decomposition_task
    assert isinstance(decomposition, extract_lora.TaskVectorDecompositionTask)
    task_vector = decomposition.input_task
    assert isinstance(task_vector, extract_lora.TaskVectorTask)
    assert isinstance(task_vector.base_tensor, io_tasks.LoadTensor)
    assert isinstance(task_vector.model_tensor, io_tasks.LoadTensor)


def test_global_merge_command_rejects_low_memory_flag():
    """A global merge command must not accidentally acquire the extractor flag."""

    result = CliRunner().invoke(merge_raw_pytorch.main, ["--low-memory"])

    assert result.exit_code != 0
    assert "no such option" in _error_text(result)
