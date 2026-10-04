"""NLI claim scorer (DeBERTa-v3 MNLI/FEVER/ANLI, local, offline).

Fixed rules: entailed = entailment is the argmax label and its probability >= 0.5.
Premises longer than the window limit are split into overlapping windows (<= 400 tokens); max over windows.
The model is loaded lazily on first use, never at import time.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.evaluation.answer_benchmark import is_unanswerable_case
from src.evaluation.citation_faithfulness import split_sentences
from src.retrieval.context import _walk_neighbors
from src.storage import StoredDoc

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path("/Users/necatifurkancolak/AI-Workplace/Projects/done/BuffettRAG/models/nli-deberta-v3-base")
CHUNKS_FILE = ROOT / "data" / "processed" / "chunks_v3_paragraph.jsonl"
ENTAIL_THRESHOLD = 0.5
WINDOW_TOKENS = 400
WINDOW_STRIDE = 300  # 100-token overlap
_MARKER_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


class NliScorer:
    def __init__(self, model_dir: Path = MODEL_DIR, chunks_file: Path = CHUNKS_FILE):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(self.device).eval()
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.entail_idx = next(i for i, v in id2label.items() if v == "entailment")
        self.chunks_file = chunks_file
        self._passages: Optional[Dict[str, str]] = None
        self._docs: Dict[str, StoredDoc] = {}
        self._cache: Dict[tuple, float] = {}

    def _load(self) -> None:
        if self._passages is None:
            self._passages = {}
            self._docs = {}
            with open(self.chunks_file, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        d = json.loads(line)
                        self._passages[d["id"]] = d["text"]
                        self._docs[d["id"]] = StoredDoc(d["id"], d["text"], {k: v for k, v in d.items() if k != "text"})

    def passage_text(self, pid: str) -> str:
        self._load()
        return self._passages.get(pid, "")

    def block_text(self, pid: str) -> str:
        """Cited passage plus its +/-1 same-year neighbours, as the answer system saw it."""
        self._load()
        doc = self._docs.get(pid)
        if doc is None:
            return ""
        before = _walk_neighbors(doc, self._docs, "previous_chunk_id", 1)
        after = _walk_neighbors(doc, self._docs, "next_chunk_id", 1)
        return "\n\n".join(d.text for d in (*before, doc, *after))

    @staticmethod
    def _norm(text: str) -> str:
        return re.sub(r"\W+", " ", text.lower()).strip()

    def support_prob(self, block: str, sentence: str) -> float:
        """1.0 if the sentence is contained verbatim (normalized) in the block, else max NLI over
        1-3 sentence windows of the block and the whole block."""
        if not block.strip() or not sentence.strip():
            return 0.0
        if self._norm(sentence) and self._norm(sentence) in self._norm(block):
            return 1.0
        best = self.entail_prob(block, sentence)
        bs = [s for s in split_sentences(block) if s.strip()]
        for n in (1, 2, 3):
            for i in range(max(0, len(bs) - n + 1)):
                best = max(best, self.entail_prob(" ".join(bs[i:i + n]), sentence))
        return best

    def _windows(self, premise: str, hypothesis: str) -> List[str]:
        ids = self.tok(premise, add_special_tokens=False)["input_ids"]
        hyp_len = len(self.tok(hypothesis, add_special_tokens=False)["input_ids"])
        limit = max(32, WINDOW_TOKENS - hyp_len)
        if len(ids) <= limit:
            return [premise]
        out, start = [], 0
        stride = min(WINDOW_STRIDE, max(1, limit - 100))
        while True:
            out.append(self.tok.decode(ids[start:start + limit]))
            if start + limit >= len(ids):
                return out
            start += stride

    def entail_prob(self, premise: str, hypothesis: str) -> float:
        """Max over premise windows of P(entailment) and whether it is the argmax label."""
        key = (premise, hypothesis)
        if key in self._cache:
            return self._cache[key]
        best = 0.0
        if premise.strip() and hypothesis.strip():
            torch = self._torch
            wins = self._windows(premise, hypothesis)
            enc = self.tok(wins, [hypothesis] * len(wins), truncation=True, max_length=512,
                           padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self.model(**enc).logits.float(), dim=-1).cpu()
            for row in probs:
                if int(row.argmax()) == self.entail_idx:
                    best = max(best, float(row[self.entail_idx]))
        self._cache[key] = best
        return best

    def entailed(self, premise: str, hypothesis: str) -> bool:
        return self.entail_prob(premise, hypothesis) >= ENTAIL_THRESHOLD

    # ---- answer-level scoring ----
    @staticmethod
    def clean(sentence: str) -> str:
        return re.sub(r"\s+", " ", _MARKER_RE.sub("", sentence)).strip()

    def _cited_ids(self, sentence: str, passage_ids: Sequence[str]) -> List[str]:
        out: List[str] = []
        for raw in _MARKER_RE.findall(sentence):
            for n in raw.split(","):
                i = int(n.strip()) - 1
                if 0 <= i < len(passage_ids) and passage_ids[i] not in out:
                    out.append(passage_ids[i])
        return out

    def score_row(self, row: Dict[str, Any], case: Dict[str, Any]) -> Dict[str, Any]:
        qid = row["qid"]
        if row.get("provider_failures"):
            return {"qid": qid, "unscored": True, "nli_accepted": None}
        lex = row.get("score") or {}
        if is_unanswerable_case(case):
            return {"qid": qid, "unscored": False, "nli_accepted": bool(lex.get("accepted")),
                    "lexical_accepted": bool(lex.get("accepted")), "nli_claims": [],
                    "nli_claim_coverage": None, "sentences": [], "citation_support": None}
        passage_ids = row.get("passage_ids") or []
        raw_sents = split_sentences(row.get("answer") or "")
        sents = []
        for raw in raw_sents:
            text = self.clean(raw)
            cited = self._cited_ids(raw, passage_ids)
            prob = max((self.support_prob(self.block_text(p), text) for p in cited), default=0.0)
            supported = bool(cited) and prob >= ENTAIL_THRESHOLD
            sents.append({"text": text, "cited_passage_ids": cited, "supported": supported,
                          "support_prob": round(prob, 4)})
        lex_claims = {c["claim"]: c.get("met") for c in lex.get("claims", [])}
        claims = []
        for gc in case["gold_claims"]:
            claim = gc["claim"]
            best = None  # (prob, indices)
            cands = [(i,) for i in range(len(sents))] + [(i, i + 1) for i in range(len(sents) - 1)]
            for idx in cands:
                premise = " ".join(sents[i]["text"] for i in idx)
                p = self.entail_prob(premise, claim)
                if p >= ENTAIL_THRESHOLD and (best is None or p > best[0]):
                    best = (p, idx)
            met = best is not None
            used = list(best[1]) if met else []
            claims.append({
                "claim": claim, "nli_met": met, "nli_prob": round(best[0], 4) if met else None,
                "lexical_met": bool(lex_claims.get(claim)),
                "supporting_sentences": [sents[i]["text"] for i in used],
                "supporting_sentence_supported": [sents[i]["supported"] for i in used],
            })
        n = len(claims)
        coverage = sum(c["nli_met"] for c in claims) / n if n else 0.0
        used_ok = all(all(c["supporting_sentence_supported"]) for c in claims if c["nli_met"])
        min_cov = float((case.get("accept") or {}).get("min_claim_coverage", 1.0))
        accepted = coverage >= min_cov and used_ok
        cited_sents = [s for s in sents if s["cited_passage_ids"]]
        support = (sum(s["supported"] for s in cited_sents) / len(cited_sents)) if cited_sents else None
        return {"qid": qid, "unscored": False, "nli_accepted": accepted,
                "lexical_accepted": bool(lex.get("accepted")), "nli_claims": claims,
                "nli_claim_coverage": coverage, "sentences": sents,
                "citation_support": support,
                "cited_sentences": len(cited_sents),
                "supported_cited_sentences": sum(s["supported"] for s in cited_sents)}
