"""Structural grounded interfaces; implementations own lazy loading and locks.

SentenceScorer.score returns sigmoid probabilities, in input order. It MUST
reject non-finite values and output-length mismatches. Composer.compose must
forward req.temperature to the model (zero is a real value, not a falsey
fallback); raw_generate is the separate query-expansion path. Verifier receives
GroupEvidence entries, not bare multi-sentence EvidenceUnits: units[i].text is
exactly the unescaped sentence for marker [i + 1] passed to the legacy validator.

GroundedResources.attach_resources reuses an existing reranker without loading
another scorer. load is idempotent, with registry construction under a process
lock; locks has "load", "scorer", "composer" and optional "nli" keys. Shared
rerank and score inference use the same scorer lock, composer inference its own
lock. Template composition happens outside model locks. These are obligations
of the implementation, not eager model construction in a contract module.
"""

from __future__ import annotations

from typing import Any, ContextManager, Mapping, Protocol, Sequence, runtime_checkable

from .types import ComposeRequest, ComposeResult, GroundedAnswer, GroupEvidence, VerifiedSentence

__all__ = ["SentenceScorer", "Composer", "Verifier", "Engine", "GroundedResources"]


@runtime_checkable
class SentenceScorer(Protocol):
    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Return one finite [0, 1] probability per text; validate the length."""
        ...


@runtime_checkable
class Composer(Protocol):
    def compose(self, req: ComposeRequest) -> ComposeResult:
        ...

    def raw_generate(self, prompt: str, max_tokens: int) -> str:
        """Generate expansion text; template implementations raise ProviderUnavailable."""
        ...


@runtime_checkable
class Verifier(Protocol):
    def verify(self, group_text: str, units: Sequence[GroupEvidence]) -> list[VerifiedSentence]:
        ...


@runtime_checkable
class Engine(Protocol):
    def answer(
        self,
        original_query: str,
        context_hits: Sequence[Any],
        *,
        history: Sequence[Mapping[str, str]] = (),
        max_new_tokens: int = 200,
    ) -> GroundedAnswer:
        ...


@runtime_checkable
class GroundedResources(Protocol):
    scorer: SentenceScorer
    composer: Composer
    locks: Mapping[str, ContextManager[Any]]

    def attach_resources(self, *, reranker: Any) -> None:
        """Attach the backend's shared reranker once, before first load."""
        ...

    def load(self) -> None:
        """Load missing resources once; subsequent calls reuse those instances."""
        ...
