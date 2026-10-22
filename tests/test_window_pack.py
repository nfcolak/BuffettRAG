"""finance-recall-75-r4 group 1: FINANCE_WINDOW_PACK packs the finance reader window.

1: the merged neighbour chunks of the lowest-ranked hits go before any whole hit, and their ids leave `served_ids`;
2: then the passages beyond today's budget are shortened to their sentences with the most question words (at least 2).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from config import FINANCE_DEFAULTS
from src.evaluation.claim_validator import evidence_sentences
from src.generation import finance_answer, token_count, window_pack
from src.knowledge import KnowledgeStore
from src.services import ask_flow
from src.services.schemas import AskRequest
from src.storage import SearchHit, StoredDoc

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "eval"))
import finance_route_debug as frd  # noqa: E402

from test_ask_flow_finance import FakeLLM, TREND_Q  # noqa: E402
from test_finance_policy import FakeRetriever  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "knowledge_mini"
SETTINGS = ("FINANCE_WINDOW_PACK", "FINANCE_WINDOW_TOKENS", "FINANCE_K_TREND", "LLM_CONTEXT_PASSAGES", "LLM_N_CTX",
            "LLM_PASSAGE_ORDER", "REMOTE_LLM_URL", "LLM_TOKENIZER_JSON", "FINANCE_DEBUG")
CAP, LIMIT = 4, 12
FILLER = ("The annual meeting drew a large crowd in Omaha that spring.", "Charlie told a long story about a car dealer.",
          "Several managers retired after long and loyal careers.", "The board met twice during the summer months.")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in SETTINGS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RETRIEVAL_DEDUP", "0")
    token_count.reset()
    yield
    token_count.reset()


class FourCharTokenizer:
    """One token per four characters (rounded up), so the tests can size the window exactly."""

    def encode(self, text, add_special_tokens=True):
        return SimpleNamespace(ids=[0] * math.ceil(len(text) / 4))


def _anchor_text(year: int) -> str:
    return (f"In {year} Buffett said bonds were a poor bargain for owners. The office moved to a new floor that year. "
            f"Inflation hurt bonds badly over time, he wrote in {year}. The company bought a small candy maker. "
            f"His view of long bonds changed after the rate rise. The staff at headquarters stayed very small.")


def _chained_docs(n: int = LIMIT):
    """Per letter (one year each): a before chunk, the anchor and an after chunk, linked as neighbours."""
    docs, anchors = {}, []
    for i in range(n):
        year = 1980 + i
        ids = [f"{year}_p{j:04d}" for j in range(3)]
        texts = [" ".join(FILLER) + f" Note {i} came first.", _anchor_text(year), " ".join(FILLER) + f" Note {i} came last."]
        for j, (pid, text) in enumerate(zip(ids, texts, strict=True)):
            meta = {"year": year, "source_file": f"{year}.pdf"}
            if j > 0:
                meta["previous_chunk_id"] = ids[j - 1]
            if j < 2:
                meta["next_chunk_id"] = ids[j + 1]
            docs[pid] = StoredDoc(id=pid, text=text, metadata=meta)
        anchors.append(ids[1])
    return docs, anchors


@pytest.fixture
def flow(monkeypatch):
    docs, anchors = _chained_docs()
    ranking = [SearchHit(pid, docs[pid].text, dict(docs[pid].metadata), 1.0 - n / 100)
               for n, pid in enumerate(reversed(anchors))]  # newest letter ranks first: rank order is not year order
    monkeypatch.setattr(ask_flow, "_state", {"retriever": FakeRetriever(ranking), "llm": FakeLLM(), "docs_by_id": docs,
                                              "knowledge": KnowledgeStore.load(FIX)})
    monkeypatch.setattr(token_count, "_tokenizer", lambda: FourCharTokenizer())
    monkeypatch.setenv("FINANCE_PIPELINE", "1")
    monkeypatch.setenv("LLM_CONTEXT_PASSAGES", str(CAP))
    monkeypatch.setenv("FINANCE_K_TREND", str(LIMIT))
    return SimpleNamespace(docs=docs, anchors=anchors)


def _route(monkeypatch, mode: int, window=None):
    """The trend question's route with FINANCE_WINDOW_PACK=mode and (given) a reader window of `window` tokens."""
    monkeypatch.setenv("FINANCE_WINDOW_PACK", str(mode))
    if window is not None:
        monkeypatch.setenv("LLM_N_CTX", str(window + 900 + token_count.WINDOW_MARGIN_TOKENS))
    return ask_flow.finance_route(AskRequest(query=TREND_Q))


def _own_prompt(route) -> str:
    """The prompt of `route`'s passages with every merged neighbour removed."""
    own = [window_pack.own_chunk(h) or h for h in route.hits]
    return finance_answer.reader_prompt(route.query, own, [], trend=True)


def _rank(route):
    return [h.id for h in route.retrieved_hits]


def _sizes(monkeypatch):
    """Tokens of the full prompt and of the prompt without neighbour text, all twelve passages served."""
    full = _route(monkeypatch, 0)
    assert len(full.hits) == LIMIT and full.debug["window_trimmed"] == 0
    return full, token_count.count_tokens(full.prompt), token_count.count_tokens(_own_prompt(full))


# --- the setting and the helpers ---
def test_the_setting_key_exists_and_defaults_to_the_adopted_mode():
    assert FINANCE_DEFAULTS["FINANCE_WINDOW_PACK"] == 2 and window_pack.MIN_SENTENCES == 2  # finance-recall-75-r4, task 4.1


def test_own_chunk_keeps_the_anchor_text_and_drops_the_neighbour_ids(flow, monkeypatch):
    full = _route(monkeypatch, 0)
    hit = full.hits[0]
    assert len(hit.metadata["merged_ids"]) == 3  # before, own, after
    own = window_pack.own_chunk(hit)
    assert own.id == hit.id and own.text == flow.docs[hit.id].text and own.score == hit.score
    assert own.metadata["merged_ids"] == [hit.id] and own.metadata["merged_spans"] == [[hit.id, 0, len(own.text)]]
    assert own.metadata["year"] == hit.metadata["year"]
    assert window_pack.own_chunk(own) is None  # nothing left to remove
    assert window_pack.own_chunk(SearchHit("x", "plain text", {}, 1.0)) is None  # never expanded


def test_shortenings_drop_the_sentences_with_the_fewest_question_words_first_and_keep_two():
    from src.generation.text_relevance import relevance_words
    hit = SearchHit("p", "Bonds were cheap. The weather was cold today. Stocks rose sharply. Bonds then fell over time.",
                    {"year": 1990, "merged_ids": ["p", "q"]}, 0.5)
    steps = window_pack.shortenings(hit, relevance_words("How did bonds change over time?"))
    assert [s.text for s in steps] == ["Bonds were cheap. The weather was cold today. Bonds then fell over time.",
                                       "Bonds were cheap. Bonds then fell over time."]  # a tie drops the later one
    assert all(s.id == "p" and s.metadata["merged_ids"] == ["p"] and s.metadata["year"] == 1990 for s in steps)
    assert window_pack.shortenings(SearchHit("p", "One sentence here. Another one.", {}, 1.0), {"bond"}) == []


def test_pack_drops_a_whole_hit_only_after_the_neighbour_text_and_then_packs_the_kept_hits_again():
    def hit(n):  # a 50-character neighbour, then the 100-character anchor
        text = "N" * 50 + "\n\n" + "A" * 100
        return SearchHit(f"h{n}", text, {"merged_ids": [f"h{n}", f"n{n}"],
                                         "merged_spans": [[f"n{n}", 0, 50], [f"h{n}", 52, 152]]}, 1.0)

    served = [hit(n) for n in range(5)]  # 5 x 152 + 4 separators = 764 characters; without neighbours 504

    def compose(hits):
        return list(hits), "|".join(h.text for h in hits)

    plain = list(served)
    while len(plain) > 2 and len(compose(plain)[1]) > 480:
        plain.pop()
    assert len(plain) == 3  # the plain trim keeps 3 passages
    done = window_pack.pack(served, compose, window_pack.count_chars, 480, 2, "q", 1)
    assert [h.id for h in done.served] == ["h0", "h1", "h2", "h3"] and done.used == len(done.prompt) <= 480
    # four hits fit once their neighbours go; packed again from the full text, the top hit keeps its neighbour
    assert [h.metadata["merged_ids"] for h in done.served] == [["h0", "n0"], ["h1"], ["h2"], ["h3"]]
    assert done.debug == {"window_pack": 1, "pack_stripped": 3, "pack_merged_dropped": 3, "pack_shortened": 0}
    nothing = window_pack.pack(served, compose, window_pack.count_chars, 10, 2, "q", 1)  # nothing fits: the floor
    assert [h.metadata["merged_ids"] for h in nothing.served] == [["h0"], ["h1"]]

    def dense(text):  # neighbour text costs four tokens a character: the per-character estimate saves too little
        return 4 * text.count("N") + len(text) - text.count("N"), "dense"

    full = dense(compose(served)[1])[0]
    assert full == 1514  # one neighbour and its separator are 202 tokens, estimated at about 103
    tight = window_pack.pack(served, compose, dense, full - 300, 2, "q", 1)
    # the first batch strips three hits; the bisection keeps the two that are needed
    assert [len(h.metadata["merged_ids"]) for h in tight.served] == [2, 2, 2, 1, 1]
    assert tight.used == full - 2 * 202 and tight.source == "dense" and tight.debug["pack_stripped"] == 2


# --- mode 1 in the finance route ---
def test_a_window_that_fits_is_left_alone(flow, monkeypatch):
    full, need, _ = _sizes(monkeypatch)
    for mode in (1, 2):
        route = _route(monkeypatch, mode, window=need)
        assert route.prompt == full.prompt and route.served_ids == full.served_ids
        assert "window_pack" not in route.debug and route.debug["window_trimmed"] == 0


def test_neighbour_text_goes_before_whole_passages(flow, monkeypatch):
    full, need, own = _sizes(monkeypatch)
    assert own < need
    plain = _route(monkeypatch, 0, window=own)
    assert CAP <= len(plain.hits) < LIMIT  # the plain trim drops whole passages
    packed = _route(monkeypatch, 1, window=own)
    assert len(packed.hits) == LIMIT and packed.debug["window_trimmed"] == 0
    assert packed.debug["prompt_tokens"] == token_count.count_tokens(packed.prompt) <= packed.debug["window_tokens"]
    assert [h.id for h in packed.hits] == [h.id for h in full.hits]  # same passages, same order, same [n]
    # every neighbour had to go, and its id left served_ids: only the anchors are served now
    assert packed.served_ids == [h.id for h in packed.hits]
    assert all(h.text == flow.docs[h.id].text for h in packed.hits)
    assert packed.debug["pack_stripped"] == LIMIT and packed.debug["pack_merged_dropped"] == 2 * LIMIT
    assert packed.debug["window_pack"] == 1 and packed.debug["pack_shortened"] == 0


def test_the_lowest_ranked_hits_lose_their_neighbours_first(flow, monkeypatch):
    full, need, _ = _sizes(monkeypatch)
    route = _route(monkeypatch, 1, window=need - 1)  # one token short
    assert len(route.hits) == LIMIT and route.debug["prompt_tokens"] <= need - 1
    stripped = [h.id for h in route.hits if h.metadata["merged_ids"] == [h.id]]
    assert 1 <= len(stripped) < LIMIT and route.debug["pack_stripped"] == len(stripped)
    assert set(stripped) == set(_rank(route)[-len(stripped):])  # the tail of the ranking, not of the year order
    gone = [pid for pid in full.served_ids if pid not in route.served_ids]
    assert len(gone) == route.debug["pack_merged_dropped"] == 2 * len(stripped)
    assert all(pid in flow.docs and pid not in flow.anchors for pid in gone)  # only neighbour ids left served_ids


@pytest.mark.parametrize("mode", [1, 2])
def test_packing_never_serves_fewer_passages_than_the_plain_trim_and_always_fits(flow, monkeypatch, mode):
    _, need, own = _sizes(monkeypatch)
    for window in (40, own // 3, own // 2, own - 80, own - 1, own, (own + need) // 2, need - 1):
        plain, packed = _route(monkeypatch, 0, window), _route(monkeypatch, mode, window)
        assert len(packed.hits) >= len(plain.hits) >= CAP
        for route in (plain, packed):  # whole passages still go from the tail of the ranking
            assert {h.id for h in route.hits} == set(_rank(route)[:len(route.hits)])
        if len(packed.hits) > CAP:
            assert token_count.count_tokens(packed.prompt) <= packed.debug["window_tokens"] == window
        assert set(packed.served_ids) == {i for h in packed.hits for i in h.metadata["merged_ids"]}


def test_the_floor_is_todays_budget_and_keeps_the_packed_passages(flow, monkeypatch):
    route = _route(monkeypatch, 1, window=40)  # nothing fits
    assert len(route.hits) == CAP and route.debug["window_trimmed"] == LIMIT - CAP
    assert route.served_ids == [h.id for h in route.hits]  # the floor's passages lost their neighbours too
    assert {h.id for h in route.hits} == set(_rank(route)[:CAP])


def test_the_character_window_packs_too(flow, monkeypatch):
    monkeypatch.setenv("FINANCE_WINDOW_TOKENS", "0")
    full = _route(monkeypatch, 0)
    own = len(_own_prompt(full))
    monkeypatch.setenv("LLM_N_CTX", str(900 + math.ceil(own / 4)))  # (n_ctx - 900) * 4 characters: just the anchors
    plain, packed = _route(monkeypatch, 0), _route(monkeypatch, 1)
    assert len(plain.hits) < LIMIT == len(packed.hits)
    assert len(packed.prompt) <= packed.debug["window_chars"] and "window_tokens" not in packed.debug
    assert packed.served_ids == [h.id for h in packed.hits]


# --- mode 2 ---
def test_mode_2_shortens_the_passages_beyond_todays_budget_instead_of_dropping_them(flow, monkeypatch):
    full, _, own = _sizes(monkeypatch)
    window = own - 160  # even without neighbours, two or more passages are too many
    one, two = _route(monkeypatch, 1, window), _route(monkeypatch, 2, window)
    assert len(one.hits) < LIMIT == len(two.hits) and two.debug["window_trimmed"] == 0
    assert token_count.count_tokens(two.prompt) <= two.debug["window_tokens"] == window
    rank = _rank(two)
    shortened = [h for h in two.hits if len(h.text) < len(flow.docs[h.id].text)]
    assert shortened and len(shortened) == two.debug["pack_shortened"] and two.debug["window_pack"] == 2
    assert {h.id for h in shortened} <= set(rank[CAP:])  # today's budget is never shortened
    assert {h.id for h in shortened} == set(rank[-len(shortened):])  # the lowest ranks first
    for h in shortened:
        own_sentences = evidence_sentences(flow.docs[h.id].text)
        kept = evidence_sentences(h.text)
        assert len(kept) >= window_pack.MIN_SENTENCES
        assert kept == [s for s in own_sentences if s in kept]  # a subset, in the passage's own order
        assert any("bonds" in s for s in kept)  # the question's words survive
        assert h.metadata["merged_ids"] == [h.id] and h.metadata["year"] == flow.docs[h.id].metadata["year"]
    assert [h.id for h in two.hits] == [h.id for h in full.hits]  # the order of the passages


def test_mode_2_falls_back_to_dropping_passages_when_shortening_is_not_enough(flow, monkeypatch):
    for window in (40, 300):
        one, two = _route(monkeypatch, 1, window), _route(monkeypatch, 2, window)
        assert len(two.hits) >= len(one.hits) >= CAP
        if len(two.hits) > CAP:
            assert token_count.count_tokens(two.prompt) <= window


def test_the_claim_validator_finds_a_shortened_passage_by_its_marker(flow, monkeypatch):
    _, _, own = _sizes(monkeypatch)
    route = _route(monkeypatch, 2, own - 160)
    n, hit = next((n, h) for n, h in enumerate(route.hits, 1) if len(h.text) < len(flow.docs[h.id].text))
    sentence = next(s for s in evidence_sentences(hit.text) if "bonds" in s)
    answer, citations = ask_flow._finalize_answer(route.llm, route.prompt, route.hits, f"{sentence} [{n}]", 900)
    assert sentence in answer
    cited = {route.hits[i].id for c in citations for i in c["passage_indices"]}
    assert cited == {hit.id}  # the marker still points at the shortened passage, by its own id


# --- the eval script ---
def test_finance_route_debug_scores_the_packed_served_list(flow, monkeypatch):
    _, _, own = _sizes(monkeypatch)
    anchor = flow.anchors[0]
    neighbour = next(pid for pid in flow.docs if pid.startswith(anchor[:4]) and pid != anchor)
    case = {"qid": "t1", "group": 1, "question_type": "t", "query": TREND_Q, "gold_passage_ids": [anchor, neighbour]}
    monkeypatch.setenv("FINANCE_WINDOW_PACK", "0")
    before = frd.route_one(ask_flow, case)
    assert (before["gold_served"], before["gold_served_merged"]) == (1, 2)  # the neighbour is merged into its anchor
    monkeypatch.setenv("LLM_N_CTX", str(own + 900 + token_count.WINDOW_MARGIN_TOKENS))
    monkeypatch.setenv("FINANCE_WINDOW_PACK", "1")
    after = frd.route_one(ask_flow, case)
    route = ask_flow.finance_route(AskRequest(query=TREND_Q))
    assert after["served_merged_ids"] == route.served_ids and after["served_ids"] == [h.id for h in route.hits]
    assert (after["gold_served"], after["gold_served_merged"]) == (1, 1)  # the dropped neighbour is not counted
