"""The site's Year / Decade scope (`where`) on the finance route.

Live 2026-10-26: the site sent `where={"year": 2018}` with "what happened at yeae 2008?"; the finance route ignored
`where` (it only used the plan's own years), and the site showed the request's filter as "filter year:2018" under an
answer served from other letters. The route now restricts the plan to the picked letters, filters the served passages
by the request's filter and reports that filter as `used_filter`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src.knowledge import KnowledgeStore
from src.query.plan import CORPUS_WIDE, PERSONAL, plan_query, scope_plan, where_year_range

sys.path.insert(0, str(Path(__file__).parent))
from test_ask_flow_finance import DIGEST_Q, TREND_Q, _ask, _stream, env  # noqa: E402,F401


@pytest.mark.parametrize("where, expected", [
    (None, None),
    ({}, None),
    ({"source_file": "2008.pdf"}, None),
    ({"year": 2018}, (2018, 2018)),
    ({"year": {"$eq": 2008}}, (2008, 2008)),
    ({"year": {"$gte": 1990, "$lte": 1995}}, (1990, 1995)),
    ({"year": {"$gt": 1990, "$lt": 1995}}, (1991, 1994)),
    ({"year": {"$in": [1994, 1990, 1997]}}, (1990, 1997)),
    ({"decade": 1990}, (1990, 1999)),
    ({"decade": 1990, "year": 1995}, (1995, 1995)),  # the year wins
    ({"year": {"$gte": 2000, "$lte": 1990}}, None),
])
def test_where_year_range(where, expected):
    assert where_year_range(where) == expected


def test_scope_plan_replaces_the_question_years_with_a_hard_filter():
    plan = plan_query("What is the main event happened in the year 2008?", premise_check=False)
    assert plan.years == (2008,) and not plan.year_hard
    scoped = scope_plan(plan, (2018, 2018), premise_check=False)
    assert scoped.year_range == (2018, 2018) and scoped.year_hard and scoped.years == (2018,)
    assert scoped.year_spans == ((2018, 2018),) and scoped.year_source == "explicit"


def test_scope_plan_turns_a_corpus_wide_question_into_a_period_one():
    plan = plan_query(DIGEST_Q, premise_check=False)
    assert plan.kind == CORPUS_WIDE
    scoped = scope_plan(plan, (1990, 1999), premise_check=False)
    assert scoped.kind != CORPUS_WIDE and scoped.superlative == ""


def test_scope_plan_leaves_a_personal_plan_alone():
    plan = plan_query("How old was Mrs. B when she opened her carpet store?", KnowledgeStore.empty(),
                      premise_check=False)
    if plan.scope == PERSONAL:
        assert scope_plan(plan, (1990, 1990), premise_check=False) is plan


def test_route_without_where_is_unchanged(env):
    body = _ask(env, TREND_Q)
    assert body["used_filter"] is None
    assert len({h["year"] for h in body["hits"]}) > 1


@pytest.mark.parametrize("where, years", [({"year": 1990}, {1990}), ({"decade": 1990}, {1990, 1996})])
def test_route_serves_only_the_picked_letters_on_both_endpoints(env, where, years):
    body = _ask(env, TREND_Q, where=where)
    events, _ = _stream(env, TREND_Q, where=where)
    assert body["hits"] and {h["year"] for h in body["hits"]} <= years
    assert {h["year"] for h in events["meta"]["hits"]} <= years
    assert body["used_filter"] == events["meta"]["used_filter"] == where


def test_route_with_a_scope_that_has_no_passages_refuses_without_the_model(env):
    body = _ask(env, "What happened in 2008?", where={"year": 2018})  # the fixture has no 2018 passage
    assert body["hits"] == [] and body["used_filter"] == {"year": 2018}
    assert env.llm.prompts == []


def test_scope_keeps_the_part_of_the_question_years_inside_it():
    plan = plan_query("How did book value grow from 1985 to 1995?", premise_check=False)
    scoped = scope_plan(plan, (1990, 1999), premise_check=False)
    assert scoped.year_range == (1990, 1995) and scoped.year_hard


def test_scope_outside_the_question_years_drops_the_figure_lookup():
    plan = plan_query("What was Berkshire's book value per share in 1987?", premise_check=False)
    assert plan.kind == "numeric"
    scoped = scope_plan(plan, (1990, 1990), premise_check=False)
    assert scoped.kind != "numeric" and not scoped.numeric and scoped.year_range == (1990, 1990)


def test_numeric_question_inside_the_scope_is_answered_for_the_asked_year(env):
    from test_ask_flow_finance import NUMERIC_Q
    body = _ask(env, NUMERIC_Q, where={"year": 1987})
    assert "$2,477.47" in body["answer"] and body["used_filter"] == {"year": 1987}


def test_numeric_question_outside_the_scope_never_gets_another_year_figure(env):
    """Live probe 2026-10-26: 1987 asked, 1990 picked -> the numbers table answered with 1990's $4,612.06."""
    from test_ask_flow_finance import NUMERIC_Q
    body = _ask(env, NUMERIC_Q, where={"year": 1990})
    assert "as written" not in (body["answer"] or "") and "$2,477.47" not in (body["answer"] or "")
    assert all(h["year"] == 1990 for h in body["hits"]) and body["used_filter"] == {"year": 1990}


def test_the_site_matches_the_backend_refusal_and_scope_reply_text():
    """The site recognises these two answers by exact text (the scope hint under a refusal, the scope-reply style)."""
    from src.generation.finance_answer import SCOPE_REPLY
    from src.generation.prompt import REFUSAL_LINE
    site = (Path(__file__).parents[1] / "frontend" / "src" / "main.jsx").read_text(encoding="utf-8")
    assert f'const REFUSAL_LINE = "{REFUSAL_LINE}";' in site
    assert f'const SCOPE_REPLY = "{SCOPE_REPLY}";' in site


def test_a_question_year_keeps_the_next_letter_inside_a_wider_scope():
    """Live proof run W4: "What happened in 2008?" with the 2000s picked was served the 2008 letter only and refused;
    without a scope the route also reads the 2009 letter, which looks back on 2008."""
    plan = plan_query("What happened in 2008?", premise_check=False)
    scoped = scope_plan(plan, (2000, 2009), premise_check=False)
    assert scoped.year_range == (2008, 2009) and scoped.years == (2008,)
    assert scope_plan(plan, (2008, 2008), premise_check=False).year_range == (2008, 2008)


def test_a_figure_inside_the_scope_from_an_out_of_scope_letter_is_not_served(monkeypatch):
    """Live proof run W2: float figures for 1980-1985 came from a table in the 1990 letter while the 1980s were picked."""
    import json as _json
    from src.services import ask_flow
    from src.services.schemas import AskRequest
    from src.storage import StoredDoc
    chunks = Path(__file__).parents[1] / "data" / "processed" / "chunks_v3_paragraph.jsonl"
    docs = {}
    for line in chunks.read_text(encoding="utf-8").splitlines():
        row = _json.loads(line)
        docs[row["id"]] = StoredDoc(id=row["id"], text=row["text"], metadata=row.get("metadata") or {"year": int(row["id"][:4])})

    class NoSearch:
        reranker = None

        def _hybrid_for_filter(self, *a):
            return []

        def _bm25_search(self, *a, **k):
            return []

    monkeypatch.setenv("FINANCE_PIPELINE", "1")
    monkeypatch.setattr(ask_flow, "_state", {"retriever": NoSearch(), "docs_by_id": docs,
                                             "knowledge": KnowledgeStore.load(), "llm": None})
    q = "What did Buffett say about insurance float?"
    open_route = ask_flow.finance_route(AskRequest(query=q))
    assert open_route.route == "numeric" and any(h.metadata["year"] == 1990 for h in open_route.hits)  # the setup holds
    route = ask_flow.finance_route(AskRequest(query=q, where={"decade": 1980}))
    assert route.used_filter == {"decade": 1980}
    assert all(1980 <= h.metadata["year"] <= 1989 for h in route.hits)
