"""Central configuration for the BuffettRAG project.

Kept backward compatible with the original ingestion pipeline (paths,
CHUNK_SIZE, CHUNK_OVERLAP, year range). New sections cover embeddings,
vector store, retrieval, generation, and evaluation.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


if load_dotenv is not None:
    load_dotenv()


def _csv_env(name: str, default: str = "") -> tuple[str, ...]:
    raw = os.getenv(name, default)
    return tuple(item.strip() for item in raw.split(",") if item.strip())

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
OUTPUT_DIR = DATA_DIR / "processed"
CHUNKS_FILE = OUTPUT_DIR / "chunks.jsonl"
CHUNKS_V2_FILE = OUTPUT_DIR / "chunks_v2.jsonl"  # historical compatibility corpus
CHUNKS_V3_PARAGRAPH_FILE = OUTPUT_DIR / "chunks_v3_paragraph.jsonl"  # v3: Berkshire letters 1977-2024 only
# Legacy name kept for every consumer that imports it: it names the ACTIVE corpus (v3, Berkshire letters only).
CHUNKS_V3_FILE = CHUNKS_V3_PARAGRAPH_FILE
METADATA_FILE = OUTPUT_DIR / "metadata.json"
METADATA_V2_FILE = OUTPUT_DIR / "metadata_v2.json"

INDEX_DIR = DATA_DIR / "indices"
FAISS_DIR = INDEX_DIR / os.getenv("FAISS_INDEX_NAME", "faiss_v3")

EVAL_DIR = DATA_DIR / "evaluation"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
FAISS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Ingestion / chunking
# -----------------------------------------------------------------------------
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Improved pipeline uses larger semantic chunks because bge-* handles them well
# and Buffett's letters are paragraph-heavy.
CHUNK_SIZE_V2 = 800
CHUNK_OVERLAP_V2 = 120
CHUNK_MIN_CHARS_V2 = 160

START_YEAR = 1977
END_YEAR = 2024
TOTAL_LETTERS = END_YEAR - START_YEAR + 1

TEXT_FILES_PATTERN = "buffet_*.txt"
PDF_FILES_PATTERN = "buffet_*.pdf"

# -----------------------------------------------------------------------------
# Embeddings
# -----------------------------------------------------------------------------
# Primary -- ~440MB, 768-dim, MIT
EMBEDDING_MODEL_PRIMARY = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
EMBEDDING_DIM_PRIMARY = 768

# Alias kept for get_embedder('base'/'large') compatibility; same model as primary.
EMBEDDING_MODEL_LARGE = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM_LARGE = 768

# bge-* prepends a query instruction for retrieval
BGE_QUERY_INSTRUCTION = (
    "Represent this sentence for searching relevant passages: "
)

EMBEDDING_BATCH_SIZE = 64
def _default_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", _default_device())

# -----------------------------------------------------------------------------
# Vector store
# -----------------------------------------------------------------------------
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "faiss")

# -----------------------------------------------------------------------------
# Retrieval
# -----------------------------------------------------------------------------
DEFAULT_TOP_K = 8           # final passages returned to the LLM
RETRIEVAL_FETCH_K = 30      # candidates fetched before reranking / fusion
RRF_K = 60                  # reciprocal rank fusion constant
ANSWER_CONTEXT_NEIGHBORS = int(os.getenv("ANSWER_CONTEXT_NEIGHBORS", "1"))
# Per-passage context budget. Modern cloud models handle 100k+ token windows;
# 9000 chars (~2250 tokens) per passage keeps neighbor expansion from truncating.
ANSWER_CONTEXT_MAX_CHARS = int(os.getenv("ANSWER_CONTEXT_MAX_CHARS", "9000"))
# Max candidates sent to the cross-encoder (cost is linear in this).
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "30"))  # was 15; retrieval_dev_v8
# Query expansion: "auto" (only when first-pass evidence is weak) | "always" | "off"
# Defaults to "off": a small embedded model produces unreliable expansions.
EXPANSION_MODE = os.getenv("EXPANSION_MODE", "off").strip().lower()

# Reranker
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_TOP_K = 8

# -----------------------------------------------------------------------------
# Generation
# -----------------------------------------------------------------------------
LLM_MAX_NEW_TOKENS = 900
# Serving prompt version (src/generation/prompt.py): v1 = text FT-r1 was trained on, v2 = coverage prompt (FT-r3),
# v3 = v2 for the Berkshire-only corpus (FT-r4 training data). Env wins at call time.
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v1")
# Providers: "llama" (embedded llama.cpp GGUF model) | "remote". The extractive engine was removed.
# No external API is used anywhere.
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "llama").strip().lower()
LLM_MODEL_PATH = Path(os.getenv("LLM_MODEL_PATH", "models/ft/7b_r1/buffett-qwen2.5-7b-ft-r1-q4_k_m.gguf"))
if not LLM_MODEL_PATH.is_absolute():
    LLM_MODEL_PATH = BASE_DIR / LLM_MODEL_PATH
LLM_N_CTX = int(os.getenv("LLM_N_CTX", "16384"))  # was 8192; retrieval_dev_v8
LLM_N_THREADS = int(os.getenv("LLM_N_THREADS", "0"))  # 0 = auto
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_GPU_LAYERS = int(os.getenv("LLM_GPU_LAYERS", "-1"))  # -1 = all (Metal); 0 = CPU only
# A small CPU model cannot take 8 long passages, so cap the prompt at 5. 7B GGUF
# models get 16 (retrieval_dev_v8; 8 before, with n_ctx 8192): the dev check (80 cases) showed fewer unneeded refusals (13 -> 9 of
# 71) with no metric loss.
_DEFAULT_CONTEXT_PASSAGES = "16" if "7b" in LLM_MODEL_PATH.name.lower() else "5"
LLM_CONTEXT_PASSAGES = int(os.getenv("LLM_CONTEXT_PASSAGES", _DEFAULT_CONTEXT_PASSAGES))
LLM_PASSAGE_MAX_CHARS = int(os.getenv("LLM_PASSAGE_MAX_CHARS", "1800"))
# Neighbour-sentence expansion (src/generation/answer_expand.py): passage sentences added after / before every
# answer sentence that is a verbatim span of its cited passage. 0/0 = off. Read from the environment at call time.
ANSWER_NEIGHBOR_AFTER = int(os.getenv("ANSWER_NEIGHBOR_AFTER", "0"))
ANSWER_NEIGHBOR_BEFORE = int(os.getenv("ANSWER_NEIGHBOR_BEFORE", "0"))
ANSWER_MAX_SENTENCES = int(os.getenv("ANSWER_MAX_SENTENCES", "12"))  # a model answer this long is not expanded
# Cap on the expanded answer (a separate budget, so a long cited answer does not starve the later parts); 0 = max(12, 4 x raw sentences).
ANSWER_EXPANDED_MAX_SENTENCES = int(os.getenv("ANSWER_EXPANDED_MAX_SENTENCES", "12"))
# Retry once with the top 3 passages when the soft evidence gate let evidence through and the model still refused.
ANSWER_REFUSAL_RETRY = os.getenv("ANSWER_REFUSAL_RETRY", "1")

# -----------------------------------------------------------------------------
# Retrieval / served-context settings, read from the environment AT CALL TIME
# (data/evaluation/retrieval_dev_v7/comparison.md documents the measurement).
# The module constants above stay for import compatibility; serving code calls
# retrieval_setting() / llm_context_passages() so an env var set later still applies.
# -----------------------------------------------------------------------------
RETRIEVAL_DEFAULTS: dict = {
    "RERANK_CANDIDATES": 30,          # candidates sent to the cross-encoder (was 15)
    "RRF_K": 60,                      # reciprocal-rank-fusion constant
    "HYBRID_VECTOR_WEIGHT": 1.0,      # dense ranking weight in the hybrid fusion
    "HYBRID_BM25_WEIGHT": 1.0,        # BM25 ranking weight in the hybrid fusion
    "ANSWER_CONTEXT_NEIGHBORS": 1,    # neighbour chunks merged into each served passage
    "RETRIEVAL_MULTIPART": 1,         # 1: split a multi-part question, retrieve per part (+2.1pp served recall)
    "RETRIEVAL_PART_SLOTS": 2,        # reserved slots per question part (multipart only)
    "RETRIEVAL_FOLLOWUP_TURNS": 1,    # prior user turns whose terms extend a follow-up query
    "RETRIEVAL_FOLLOWUP_ALWAYS": 0,   # 1: extend every query that has history, not only short/referential ones
    "RETRIEVAL_PERIOD_RESERVE": 1,    # reserved hits per period in a temporal comparison
    "RETRIEVAL_NEIGHBOR_ANCHORS": 0,  # neighbours of the top-N hits appended as own passages
    "RETRIEVAL_MMR_LAMBDA": 1.0,      # <1: MMR diversity over the reranked pool (1 = off)
    "LLM_CHARS_PER_TOKEN": 4.0,       # prompt-budget estimate used to drop passages that would not fit n_ctx
    "RETRIEVAL_QUERY_STRIP": 0,       # 1: add a candidate ranking from the query without question scaffolding
    "RETRIEVAL_YEAR_FILTER_MIN": 1970,  # earliest year the auto year filter recognises
    "RETRIEVAL_YEAR_FILTER_WEIGHT": 2,  # weight of the year-filtered ranking against the unfiltered one in the boost
    "RETRIEVAL_DEDUP": 1,             # 0: no near-duplicate removal of the reranked pool
    "COMPARE_ROUTING_V2": 0,          # 1: pv8 comparison routing (more true comparisons, fewer fact questions routed)
    "RETRIEVAL_RERANK_FETCH": 0,      # 1: fetch at least RERANK_CANDIDATES candidates (deeper rerank pool than fetch_k)
    # pv8-retrieval-validator (why-v7 RT-02/03/04, TEMPORAL-01/02); every default keeps the old behaviour.
    "RETRIEVAL_PERIOD_POOL": 0,       # >0: candidates per period sent to the reranker (total cap 60); 0: max(depth // n, 4)
    "RETRIEVAL_PERIOD_FOCUS_RERANK": 0,  # 1: rerank each period's pool against its period-focused query
    "RETRIEVAL_YEAR_FILTER_KEEP": 0,  # K>0: top-K unfiltered hits stay inside the rerank window under the auto year filter
    "RETRIEVAL_YEAR_REF_SKIP": 0,     # 1: reference years ("since 1990", "end of 2001") do not set the auto year filter
    "VALIDATOR_SPLIT_V2": 0,          # 1: split_claims makes no part under 3 content tokens
    "VALIDATOR_VERBATIM_KEEP": 0,     # 1: keep a sentence that is a verbatim span (>= 6 tokens) of its cited passage
    "LLM_PASSAGE_MAX_CHARS": LLM_PASSAGE_MAX_CHARS,  # per-passage char cap of the served context (call-time copy)
}


def retrieval_setting(name: str):
    """Typed value of a retrieval setting; the environment wins, read on every call."""
    default = RETRIEVAL_DEFAULTS[name]
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return type(default)(float(raw)) if isinstance(default, int) else float(raw)


# -----------------------------------------------------------------------------
# Finance pipeline settings (openspec change finance-query-pipeline), read from the environment AT CALL TIME.
# Nothing reads them unless FINANCE_PIPELINE=1, so the defaults leave flag-off behaviour unchanged.
# -----------------------------------------------------------------------------
FINANCE_DEFAULTS: dict = {
    "FINANCE_K": 0,                    # passages finance_search returns before the neighbour merge; 0 = llm_context_passages()
    "FINANCE_OVERFETCH": 3,            # candidates fetched per ranking = max(RETRIEVAL_FETCH_K, k * this)
    "FINANCE_YEAR_BOOST_WEIGHT": 2,    # weight of the year-filtered ranking against the unfiltered one
    "FINANCE_YEAR_KEEP": 3,            # year boost only: the top-N unfiltered hits stay served even outside the range
    "FINANCE_ENTITY_BOOST_WEIGHT": 1,  # weight of the ranking of passages tagged with the plan's entities
    "FINANCE_TAG_DEPTH": 200,          # depth of the ranking the entity-tagged candidates are drawn from
    "FINANCE_TAG_POOL": 24,            # entity-tagged candidates the reranker always sees, beyond RERANK_CANDIDATES
    "FINANCE_MAX_PER_LETTER": 6,       # questions spanning several letters: passages per letter at most (finance-recall-75-r2: 2 -> 3; finance-answer-quality: 3 -> 6)
    "FINANCE_MIN_LETTERS": 4,          # questions spanning several letters: different letters at least, when available
    # finance-recall-75: candidate sources (src/retrieval/finance_policy.py registry) and served budget. Group 5 adopted the
    # ones that raised dev served gold without lowering heldout_v6 and still add gold in the combination; the rest stay
    # off or at today's value. finance-recall-75-r2 group 4 applied the same rule to the round-2 settings below (and
    # FINANCE_MAX_PER_LETTER / FINANCE_K_CORPUS); summary fusion, source slots and the stance source stay off.
    # finance-recall-75-r3 group 4 adopted the token-counted window, K_SPECIFIC 32, K_PERIOD 40, summary fusion 1 and
    # one multi-query search (dev anchors 65.4 -> 72.6, heldout_v6 71.5 -> 72.6); the digest query and the superlative
    # tagged passages stay off.
    # finance-recall-75-r4 group 4 adopted window packing 2, K_TREND / K_CORPUS / K_SPECIFIC 80 and 50 rerank candidates
    # (dev anchors / merged 72.6 / 85.9 -> 81.3 / 85.3, heldout_v6 72.6 / 76.2 -> 74.3 / 74.9); packing frees the window
    # the larger budgets need. The decade blocks and the superlative rerank stay off; K_PERIOD stays 40.
    # finance-answer-quality group 4 (round 5) adopted 6 passages per letter, tables for figure questions, the follow-up
    # plan carry-over (2) and the gentler follow-up vote (7) (dev anchors / merged 81.3 / 85.3 -> 82.4 / 85.3, heldout_v6
    # 74.3 / 74.9 -> 79.3 / 80.0, heldout_v4+v5 72.2 / 76.1 -> 78.7 / 81.5); the plan-query rewrite and the per-letter
    # topic quotas stay off.
    "FINANCE_SUMMARY_SOURCE": 1,       # 1: candidates from the annotation-summary index (summary_source.py)
    "FINANCE_TAGGED_SOURCE": 1,        # 1: passages tagged with the plan's / inferred sectors, companies, events, themes (knowledge_source.py)
    "FINANCE_DIGEST_SOURCE": 1,        # 1: digest timeline source passages for trend / corpus-wide questions (knowledge_source.py)
    "FINANCE_STANCE_SOURCE": 0,        # 1: passages with a stance on the plan's sector or company (knowledge_source.py)
    "FINANCE_K_TREND": 80,             # served passages for a trend plan; 0 = today's budget (FINANCE_K / llm_context_passages()) (finance-recall-75-r4: 32 -> 80)
    "FINANCE_K_CORPUS": 80,            # served passages for a corpus-wide plan (and the superlative route); 0 = today's budget (finance-recall-75-r4: 32 -> 80)
    "FINANCE_K_PERIOD": 40,            # served passages for a period / comparison plan; 0 = today's budget (finance-recall-75-r3: 32 -> 40)
    "FINANCE_RERANK_CANDIDATES": 50,   # candidates sent to the cross-encoder on the finance path (finance-recall-75-r4: 30 -> 50)
    "FINANCE_FOLLOWUP_HISTORY": 0,     # 1: a short / referential follow-up to a financial question is planned on that question, never scope-replied for being short (ask_flow.py)
    # served passages for a specific / numeric plan, capped by the reader window like the trend budget; 0 = today's budget.
    # 32 since finance-recall-75-r3: the token-counted window (FINANCE_WINDOW_TOKENS) bounds number-dense prompts
    # (~3.1 chars per Qwen token), which the character window did not (round 2 kept 21 for that reason; 24 overflowed it).
    # 80 since finance-recall-75-r4: window packing fits the larger budget into the same token window.
    "FINANCE_K_SPECIFIC": 80,
    "FINANCE_SUMMARY_FUSE_WEIGHT": 1,  # RRF weight of the summary ranking fused into the candidate order (0 = off; finance-recall-75-r3: 1)
    "FINANCE_THEME_CENTROIDS": 1,      # 1: theme inference also matches tagged-passage summary centroids
    "FINANCE_SOURCE_SLOTS": 0,         # served slots reserved for source candidates below the rerank cut
    "FINANCE_PLAN_TERM_BOOST": 5,      # integer, hundredths (5 = +0.05): rerank score bonus for candidates tagged with the plan's targets (0 = off)
    "FINANCE_WINDOW_TOKENS": 1,        # 1: finance reader trim counts tokens (server /tokenize, else tokenizer.json, else chars/4); 0: characters / 4
    "FINANCE_MULTI_QUERY": 1,          # extra theme searches for view/opinion plans (0 = off; N = 1, 2 and 3 measured the same)
    "FINANCE_DIGEST_QUERY": 0,         # 1: one extra search with the matching digest's summary as the query
    "FINANCE_SUPERLATIVE_TAGGED": 0,   # superlative route: serve up to N positive-stance passages of the winner (0 = off)
    "FINANCE_WINDOW_PACK": 2,          # 1: drop merged neighbours of low-ranked hits before whole hits; 2: also shorten them (finance-recall-75-r4: 0 -> 2)
    "FINANCE_DECADE_BLOCKS": 0,        # corpus-wide/trend plans: top M per decade block as extra candidates (0 = off)
    "FINANCE_SUPERLATIVE_RERANK": 0,   # superlative route: serve the top N winner passages by cross-encoder score (0 = off)
    # finance-answer-quality group 4 (retrieval round 5); the adopted ones are noted "round 5", the other two stay off.
    "FINANCE_PLAN_QUERY": 0,           # N: searches rewritten from the plan (topic keywords for non-view plans, event aliases + real years)
    "FINANCE_LETTER_QUOTA": 0,         # 1: diverse plans give each letter places in proportion to its passages on the plan's topic
    "FINANCE_FOLLOWUP_PLAN": 2,        # 1: a follow-up without years / entities carries the previous financial question's; 2: also adds its letters (round 5: 0 -> 2)
    "FINANCE_FOLLOWUP_VOTE": 7,        # N: a short follow-up is scope-voted personal / mixed only at a vote share >= N / 10 (0 = the planner's 0.6; round 5: 0 -> 7)
    "FINANCE_TABLE_CUE": 1,            # 1: table passages also join a non-numeric plan that asks for figures (numeric intent or a figure cue; round 5: 0 -> 1)
    "FINANCE_PREMISE_CHECK": 0,        # 1: plan.premise_note when the asked period misses the event / topic years
    "FINANCE_ANSWER_V3": 0,            # 1: year-grouped evidence pack, premise sentence, dedupe, partial retry
    "FINANCE_MIXED_STRICT": 0,         # 1: mixed plans drop answer sentences cited only to personal passages that carry no business figure (FT-r4 fix round)
}


def finance_setting(name: str):
    """Typed value of a finance pipeline setting; the environment wins, read on every call."""
    default = FINANCE_DEFAULTS[name]
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return type(default)(float(raw))


def finance_pipeline_enabled() -> bool:
    """FINANCE_PIPELINE=1 at call time (default off: today's query path, byte for byte)."""
    return (os.environ.get("FINANCE_PIPELINE") or "").strip() == "1"


def finance_debug_enabled() -> bool:
    """FINANCE_DEBUG=1 at call time (staging only; default off: /ask responses keep today's schema)."""
    return (os.environ.get("FINANCE_DEBUG") or "").strip() == "1"


def llm_n_ctx() -> int:
    """LLM_N_CTX at call time (default 16384): the llama window and the prompt budget of the served context."""
    raw = os.environ.get("LLM_N_CTX")
    return int(raw) if raw and raw.strip() else 16384


def llm_context_passages() -> int:
    """LLM_CONTEXT_PASSAGES at call time (default: 16 for 7B GGUF models, else 5)."""
    raw = os.environ.get("LLM_CONTEXT_PASSAGES")
    return int(raw) if raw and raw.strip() else int(_DEFAULT_CONTEXT_PASSAGES)


# Remote provider (DEFAULT_LLM_PROVIDER=remote): OpenAI-compatible endpoint, all read at call time.
REMOTE_LLM_DEFAULT_MODEL = "buffett-qwen2.5-7b-ft-r1-q4_k_m.gguf"


def remote_llm_url() -> str:
    """REMOTE_LLM_URL at call time (base URL; empty when unset)."""
    return (os.environ.get("REMOTE_LLM_URL") or "").strip()


def remote_llm_model() -> str:
    """REMOTE_LLM_MODEL at call time (default: the FT-r1 GGUF file name)."""
    return (os.environ.get("REMOTE_LLM_MODEL") or "").strip() or REMOTE_LLM_DEFAULT_MODEL


def remote_llm_token() -> str:
    """REMOTE_LLM_TOKEN at call time (bearer secret; never log it)."""
    return (os.environ.get("REMOTE_LLM_TOKEN") or "").strip()


def remote_llm_timeout_s() -> float:
    """REMOTE_LLM_TIMEOUT_S at call time (default 300: Modal cold start)."""
    raw = os.environ.get("REMOTE_LLM_TIMEOUT_S")
    return float(raw) if raw and raw.strip() else 300.0


def remote_llm_retries() -> int:
    """REMOTE_LLM_RETRIES at call time (default 2; connect errors and 502/503/504 only)."""
    raw = os.environ.get("REMOTE_LLM_RETRIES")
    return int(raw) if raw and raw.strip() else 2

# -----------------------------------------------------------------------------
# Service URLs
# -----------------------------------------------------------------------------
# Backend FastAPI URL -- Frontend uses this to reach the RAG service.
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
# -----------------------------------------------------------------------------
# API security
# -----------------------------------------------------------------------------
# Leave API_KEYS empty for local-only development. In shared or public
# deployments, set comma-separated keys and send one as X-API-Key.
API_KEYS = _csv_env("API_KEYS")

CORS_ORIGINS = _csv_env(
    "CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,"
    "http://localhost:5173,http://127.0.0.1:5173",
)
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "60"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
MAX_REQUEST_BODY_BYTES = min(max(int(os.getenv("MAX_REQUEST_BODY_BYTES", "65536")), 1024), 1_048_576)
PUBLIC_DEMO_MODE = os.getenv("PUBLIC_DEMO_MODE", "0") == "1"
# Only trust X-Forwarded-For for rate-limit client identity when the service
# actually runs behind a reverse proxy that sets it; otherwise the header is
# client-controlled and lets callers reset their own rate-limit bucket.
TRUST_PROXY_HEADERS = os.getenv("TRUST_PROXY_HEADERS", "0") == "1"
EXPOSE_DEBUG_STATUS = os.getenv("EXPOSE_DEBUG_STATUS", "0") == "1"

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
EVAL_TOP_K_VALUES = [1, 3, 5, 10]
