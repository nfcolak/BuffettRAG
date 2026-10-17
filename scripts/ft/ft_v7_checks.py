"""ft_v7 pure helpers for build_ft_v6.py (no corpus, no index): sentence alignment and the citation rule, the serving replay,
question style, refusal terms, the partial-answer gap line, source position mix and exclusion neighbours."""
from __future__ import annotations

from functools import lru_cache
import random
import re

from src.evaluation.claim_validator import evidence_sentences, validate_and_filter_answer
from src.generation.answer_expand import _Passage, _core
from src.generation.prompt import format_answer_markdown, strip_chat_artifacts
from src.generation.text_relevance import relevance_words
from src.storage import SearchHit


def norm(text):
    return re.sub(r'\s+', ' ', text).strip()


# ------------------------------------------------------------------ sentence alignment + citation rule (SP-01, TARGET-02, TPC-01/03, TARGET-06)
@lru_cache(maxsize=8192)
def aligned_set(passage_text):
    """The complete sentences of a served passage, split exactly like the serving validator and answer_expand split it."""
    return frozenset(norm(s) for s in evidence_sentences(passage_text))


def is_aligned(sentence, passage_text):
    """True when `sentence` IS one sentence of the passage (a substring or a half sentence is not)."""
    return norm(sentence) in aligned_set(passage_text)


def lowest_holder(sentence, texts):
    """The citation rule: [n] = the lowest-numbered served passage that contains the whole sentence. 1-based slot or None."""
    for slot, text in enumerate(texts, 1):
        if is_aligned(sentence, text):
            return slot
    return None


def locatable(sentence, passage_text):
    """answer_expand can find the sentence in the passage (EXP-03); a sentence it cannot locate gets no expansion at serving."""
    return _Passage(passage_text).locate(_core(sentence)) is not None


# ------------------------------------------------------------------ sentence quality (TARGET-02, TARGET-03)
DEPENDENT_OPENERS = (
    'This', 'That', 'These', 'Those', 'It', 'They', 'He', 'She', 'Such', 'However', 'But', 'And', 'Also', 'Thus', 'Hence', 'Moreover',
    'Furthermore', 'Yet', 'So', 'Then', 'Still', 'Instead', 'Both', 'Here', 'There', 'Its', 'Their', 'His', 'Her',
    # ft_v7 (target-audit a12.py): connectives that lean on the sentence before
    'Again', 'Clearly', 'Consequently', 'Therefore', 'Additionally', 'Indeed', 'Rather', 'Naturally', 'Finally', 'Similarly', 'Likewise',
    'Nor', 'Neither', 'Either', 'Otherwise', 'Conversely', 'Accordingly', 'Nevertheless', 'Nonetheless', 'In addition', 'As a result',
    'For these reasons', 'For that reason', 'On the other hand', 'In contrast', 'That is', 'In other words', 'Of course', 'Obviously',
    'Fortunately', 'Unfortunately', 'Meanwhile', 'Now', 'Because of this', 'Thereafter', 'Later', 'Today')
DEPENDENT_START = re.compile(r'^(?:' + '|'.join(sorted((re.escape(w) for w in DEPENDENT_OPENERS), key=len, reverse=True)) + r')\b')
MIN_OBLIGATION_WORDS = 8     # a whole obligation shorter than this is a fragment, not an answer (33 ft_v6 rows)
TERMINAL = re.compile(r'[.!?][\"”’\')\]]*$')


def ends_like_a_sentence(text):
    return bool(TERMINAL.search(text.strip()))


# ------------------------------------------------------------------ serving replay (POSTPROCESS-01/02, EXP-03, B14)
def served_answer(target):
    """What /ask does to the raw model text before it validates it."""
    return format_answer_markdown(strip_chat_artifacts(target))


def serving_changes(target, passages, gap_lines=()):
    """Replay `target` through the serving postprocess against its own prompt passages ([(year, text)] by slot).

    -> True when the validator's output differs from the target. `gap_lines` are whole lines that carry no citation and that
    the validator removes on purpose (the gap line of a partial row): they must be the ONLY thing it removes."""
    hits = [SearchHit(id=f'slot{i}', text=text, metadata={'year': year}, score=0.) for i, (year, text) in enumerate(passages, 1)]
    answer = served_answer(target)
    verdict = validate_and_filter_answer(answer, hits)
    skip = {norm(g) for g in gap_lines}
    expected = '\n'.join(line for line in answer.split('\n') if norm(line) not in skip)
    return norm(verdict.safe_answer) != norm(expected)


# ------------------------------------------------------------------ question style (DS-06, PD-05)
QUESTION_OVERLAP_MAX = 0.60   # eval questions overlap their gold passage by 0.50-0.57 (dist-shift DS-06); ft_v6 averaged 0.75


def question_overlap(question, texts):
    """Share of the question's stemmed content words that occur in the source passages (the dist-shift unit's measure)."""
    terms = relevance_words(question)
    if not terms:
        return 0.
    seen = set()
    for text in texts:
        seen |= relevance_words(text)
    return len(terms & seen) / len(terms)


# ------------------------------------------------------------------ refusals (RL-02, REFUSAL-01)
MONTHS_DAYS = {'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november',
               'december', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'}
_NOT_NAMES = {'the', 'what', 'which', 'how', 'who', 'when', 'why', 'where', 'did', 'does', 'was', 'were', 'is', 'are', 'in', 'of',
              'berkshire', 'buffett', 'hathaway', 'us', 'a', 'an', 'ceo', 'cfo', 'coo', 'cio', 'gaap', 'sec', 'usa', 'inc', 'corp', 'chairman', 'president'}
FUTURE_YEAR = re.compile(r'\b20(?:2[5-9]|[3-9]\d)\b')
PRICE_TALK = re.compile(r'\b(?:crypto\w*|bitcoin|ethereum|dogecoin|solana|nft|closing price|stock price|share price|price per share|'
                        r'trading price|market price|ticker)\b', re.I)


def distinctive_missing(term):
    """A `missing` term the served-text check can actually test: a number or a capitalised name, at least 4 characters.

    The teacher once returned 'specific companies', 'exact amount', '1990s' and 'previous year'; those never occur verbatim, so
    the 'absent from the served text' check passed without testing anything (12 of 16 ft_v6 adjacent refusals)."""
    term = (term or '').strip()
    if len(term) < 4:
        return False
    if re.search(r'\d', term):
        return True
    words = re.findall(r"[A-Za-z][A-Za-z.'’-]*", term)
    return any(w[0].isupper() and w.lower() not in _NOT_NAMES and w.lower() not in MONTHS_DAYS and len(w) >= 3 for w in words)


# ------------------------------------------------------------------ partial answers (REFUSAL-05, RL-05, DS-07)
def gap_line(part):
    """The gap sentence. Wording = the coverage note the grounded engine writes (generation/grounded/engine.py `_coverage_note`),
    which is what prompt v3 means by 'write a gap line only for a part that is genuinely absent'.

    It has no [n]: the serving validator removes an uncited line, so the user never sees it and the answer keeps only verified
    sentences. check_row therefore lets the validator drop exactly this line and nothing else."""
    part = norm(part).rstrip('.?!')
    return f'The retrieved passages do not cover {part}.'


GAP_PREFIX = 'The retrieved passages do not cover '


# ------------------------------------------------------------------ source position mix (PD-01, DS-08, PD-04, ASM-08, ASM-09)
POSITION_MODES = ('natural', 'serving', 'uniform')
# gold rank of a claim at serving (position-design PD-01, n=285): rank1 52.6%, 2 18%, 3 11.2%, 4-8 11.5%, >8 6.7%
SERVING_RANK = ((1, 1, .526), (2, 2, .18), (3, 3, .112), (4, 8, .115), (9, 10 ** 6, .067))


def parse_position_mix(text):
    """'natural,serving,uniform=0.5,0.3,0.2' -> {'natural': .5, 'serving': .3, 'uniform': .2}; None/'' -> None (no repositioning)."""
    if not text:
        return None
    names, _, values = str(text).partition('=')
    names = [n.strip() for n in names.split(',') if n.strip()]
    try:
        weights = [float(v) for v in values.split(',') if v.strip()]
    except ValueError:
        raise ValueError(f'bad --position-mix {text!r}; expected e.g. natural,serving,uniform=0.5,0.3,0.2') from None
    if len(names) != len(weights) or not names or any(n not in POSITION_MODES for n in names) or len(set(names)) != len(names) \
            or any(w < 0 for w in weights) or abs(sum(weights) - 1) > 1e-6:
        raise ValueError(f'bad --position-mix {text!r}; modes {POSITION_MODES}, weights must sum to 1')
    return dict(zip(names, weights))


def rank_weights(k):
    """P(source group sits at slot s) for s = 1..k under the serving distribution (ranks beyond k fold into the tail)."""
    weights = [0.] * k
    for lo, hi, mass in SERVING_RANK:
        slots = [s for s in range(lo, min(hi, k) + 1)]
        if not slots:                      # k < 9: the >8 mass joins the 4-8 slots
            slots = [s for s in range(4, k + 1)] or [k]
        for s in slots:
            weights[s - 1] += mass / len(slots)
    return weights


def draw_slots(mode, count, k, rng):
    """`count` distinct 1-based target slots for the source groups, drawn independently without replacement in `mode`."""
    free, out = list(range(1, k + 1)), []
    base = rank_weights(k) if mode == 'serving' else [1.] * k
    for _ in range(count):
        slot = rng.choices(free, weights=[base[s - 1] for s in free])[0]
        out.append(slot)
        free.remove(slot)
    return out


def reposition(hits, group_index, mode, rng):
    """Move the source hits (`group_index`: positions in `hits`, one per source group) to freshly drawn slots.

    The other passages keep their reranker order and fill the remaining slots. mode 'natural' returns the hits untouched."""
    hits = list(hits)
    if mode == 'natural' or not group_index:
        return hits
    k = len(hits)
    targets = draw_slots(mode, len(group_index), k, rng)
    placed = {slot - 1: hits[i] for i, slot in zip(group_index, targets)}
    rest = iter(h for i, h in enumerate(hits) if i not in set(group_index))
    return [placed[s] if s in placed else next(rest) for s in range(k)]


def pick_mode(mix, rng):
    names = list(mix)
    return rng.choices(names, weights=[mix[n] for n in names])[0]


# ------------------------------------------------------------------ exclusions (ASM-07, B15)
def with_neighbours(ids, full):
    """`ids` plus the previous/next chunk of every id that is in the corpus (ids outside it stay as they are)."""
    out = set(ids)
    for ident in ids:
        doc = full.get(ident)
        for key in ('previous_chunk_id', 'next_chunk_id'):
            if doc is not None and doc.metadata.get(key):
                out.add(doc.metadata[key])
    return out


def parse_id_list(text):
    """Comma-separated ids -> set."""
    return {part.strip() for part in str(text or '').split(',') if part.strip()}
