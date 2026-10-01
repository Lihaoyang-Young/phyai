"""CPU control-flow contracts with simulated U0 runners and processing."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from phyai.models.xiaomi_u0 import scheduler_unis as module
from phyai.models.xiaomi_u0.requests_unis import UNISGenerationRequest


def mapping() -> dict[str, list[int]]:
    """Return the released fixed-grid mapping."""
    return {
        **{str(i): [15 + i] for i in range(10)},
        "*": [9],
        "64*64": [21, 19, 9, 21, 19],
    }


@pytest.mark.parametrize(
    "key,value", [("6", [15]), ("*", [15]), ("64*64", [21]), ("0", [True])]
)
def test_header_rejects_wrong_mapping(key: str, value: list[int]) -> None:
    """Invalid mappings must fail before generation."""
    values = mapping()
    values[key] = value
    with pytest.raises(ValueError):
        module._HeaderTokenizer(values)


def test_header_rejects_missing_mapping() -> None:
    """Every required header component must be present."""
    values = mapping()
    del values["64*64"]
    with pytest.raises(ValueError):
        module._HeaderTokenizer(values)


def test_header_owns_mapping_copy() -> None:
    """Caller mutations must not change the accepted encoding."""
    values = mapping()
    tokenizer = module._HeaderTokenizer(values)
    values["6"][0] = 0
    assert tokenizer.encode("6") == [21]
    assert tokenizer.decode([21]) == "6"


def test_seed_only_touches_cpu_and_selected_cuda_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The seed helper must leave other GPU generators untouched."""
    calls = []
    cpu = SimpleNamespace(manual_seed=lambda seed: calls.append(("cpu", seed)))
    cuda = [
        SimpleNamespace(manual_seed=lambda seed: calls.append(("cuda0", seed))),
        SimpleNamespace(manual_seed=lambda seed: calls.append(("cuda1", seed))),
    ]
    monkeypatch.setattr(
        module,
        "torch",
        SimpleNamespace(
            random=SimpleNamespace(default_generator=cpu),
            cuda=SimpleNamespace(default_generators=cuda),
        ),
    )
    module._seed_request(SimpleNamespace(type="cuda", index=1), 42)
    assert calls == [("cpu", 42), ("cuda1", 42)]


class FakeRunner:
    """Model-independent runner that records plans, resets and injected failures."""

    def __init__(self, model, *, capacity, paged_backend):
        self.device = torch.device("cpu")
        self.reset_count = 0
        self.forward_count = 0
        self.fail_next = False
        self.cached_tokens = 0
        self.closed = False

    def setup(self):
        pass

    def reset(self):
        self.reset_count += 1
        self.cached_tokens = 0
        self.forward_count = 0

    def plan_inference(self, meta):
        self.cached_tokens = int(meta.seq_lens_kv[0])

    def forward(self, batch):
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("Injected forward failure")
        self.forward_count += 1
        logits = torch.full((1, 1, 282926), -torch.inf)
        logits[..., 151850] = 0
        return logits

    def close(self):
        self.closed = True


class FakeProcessor:
    """Exercise unconditional branch callbacks without asserting CFG numerics."""

    def __init__(self, **kwargs):
        self.branch = kwargs["model"]
        self.ids = kwargs["unconditional_ids"]
        self.mask = torch.ones_like(self.ids)
        self.handle = None

    def __call__(self, history, scores):
        self.ids = torch.cat((self.ids, history[:, -1:]), dim=1)
        self.mask = torch.cat((self.mask, torch.ones_like(history[:, -1:])), dim=1)
        output = self.branch(
            self.ids if self.handle is None else self.ids[:, -1:],
            attention_mask=self.mask,
            use_cache=True,
            past_key_values=self.handle,
        )
        self.handle = output["past_key_values"]
        return scores


@pytest.fixture
def scheduler(monkeypatch: pytest.MonkeyPatch):
    """Create a CPU simulator, not a replacement numerical reference."""
    monkeypatch.setattr(module, "UNISModelRunner", FakeRunner)
    monkeypatch.setattr(
        module,
        "UnbatchedClassifierFreeGuidanceLogitsForVisualTokenWithDifferentialTopKProcessor",
        FakeProcessor,
    )
    monkeypatch.setattr(module.torch.random, "fork_rng", lambda **kwargs: nullcontext())
    monkeypatch.setattr(module, "_seed_request", lambda device, seed: None)
    model = SimpleNamespace(lm_head=SimpleNamespace(out_features=282926))
    instance = module.UNISGenerationScheduler(
        model, capacity=64, tokenizer_mapping=mapping()
    )
    instance.setup()
    yield instance
    instance.close()


def request(*, replay=None, limit=3):
    """Construct short tokenized control-flow requests."""
    return UNISGenerationRequest(
        input_ids=torch.tensor([[151849, 151746]]),
        unconditional_ids=torch.tensor([[151849, 151746]]),
        max_new_tokens=limit,
        replay_ids=replay,
    )


def test_sampling_stops_at_eos(scheduler) -> None:
    """A sampled EOS terminates before the configured length limit."""
    result = scheduler.step(request())
    assert result.generated_ids.tolist() == [151850]
    assert result.stop_reason == "eos" and not result.replay
    assert scheduler.runners[0].forward_count == 1


def test_length_stop_and_next_request_reset(scheduler) -> None:
    """A new request starts with a fresh conditional and unconditional cache."""
    replay = torch.tensor([151852, 21, 19])
    first = scheduler.step(request(replay=replay))
    assert first.stop_reason == "length"
    assert first.generated_ids.tolist() == replay.tolist()
    assert first.conditional_cached_tokens == 4
    assert first.unconditional_cached_tokens == 5
    second = scheduler.step(request(replay=replay))
    assert second.generated_ids.tolist() == first.generated_ids.tolist()
    assert second.conditional_cached_tokens == 4
    assert second.unconditional_cached_tokens == 5
    assert all(runner.reset_count == 2 for runner in scheduler.runners)


def test_recovers_from_forward_exception(scheduler) -> None:
    """The following request must reset partially advanced branch state."""
    scheduler.runners[0].fail_next = True
    with pytest.raises(RuntimeError, match="Injected forward failure"):
        scheduler.step(request())
    result = scheduler.step(request())
    assert result.stop_reason == "eos"
    assert result.conditional_cached_tokens == 2
    assert result.unconditional_cached_tokens == 3
    assert all(runner.reset_count == 2 for runner in scheduler.runners)


@pytest.mark.parametrize(
    "kind", ["capacity", "dtype", "batch", "range", "limit", "replay"]
)
def test_invalid_request_does_not_reset_runner(scheduler, kind: str) -> None:
    """Reject invalid requests before modifying existing runner state."""
    values = dict(
        input_ids=torch.tensor([[151849, 151746]]),
        unconditional_ids=torch.tensor([[151849, 151746]]),
        max_new_tokens=3,
    )
    if kind == "capacity":
        values["max_new_tokens"] = 63
    elif kind == "dtype":
        values["input_ids"] = values["input_ids"].float()
    elif kind == "batch":
        values["input_ids"] = values["input_ids"].repeat(2, 1)
    elif kind == "range":
        values["input_ids"][0, 0] = 282926
    elif kind == "limit":
        values["max_new_tokens"] = 0
    else:
        values["replay_ids"] = torch.tensor([151852])
    with pytest.raises(ValueError):
        scheduler.step(UNISGenerationRequest(**values))
    assert all(runner.reset_count == 0 for runner in scheduler.runners)
