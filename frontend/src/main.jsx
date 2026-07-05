import React from "react";
import { createRoot } from "react-dom/client";
import { BookOpen, ChevronRight, Paperclip, Send, Settings, Trash2 } from "lucide-react";

import "./styles.css";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "/api";
const BACKEND_API_KEY = import.meta.env.VITE_BACKEND_API_KEY || "";
const HISTORY_KEY = "buffettrag.messages.v1";
const LLM_SETTINGS_KEY = "buffettrag.llmSettings.v1";

const LLM_PROVIDERS = [
  { value: "openai", label: "OpenAI", defaultModel: "gpt-4.1-mini" },
  { value: "anthropic", label: "Anthropic", defaultModel: "claude-haiku-4-5-20251001" },
  { value: "openrouter", label: "OpenRouter", defaultModel: "openrouter/free" },
];

const EXAMPLE_QUERIES = [
  "How did Buffett react to the 2008 financial crisis?",
  "What did Buffett say about GEICO and the insurance float?",
  "Buffett on succession planning at Berkshire",
  "What did Buffett say about derivatives?",
  "Buffett on inflation and purchasing power",
  "How has Buffett's view on technology stocks changed from the 1990s to the 2020s?",
];

const DECADES = [
  { value: 1970, label: "'70s" },
  { value: 1980, label: "'80s" },
  { value: 1990, label: "'90s" },
  { value: 2000, label: "'00s" },
  { value: 2010, label: "'10s" },
  { value: 2020, label: "'20s" },
];

const YEARS = Array.from({ length: 48 }, (_, i) => 1977 + i);

function nowLabel() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function makeId(prefix) {
  return `${prefix}_${Date.now()}_${Math.random().toString(16).slice(2, 8)}`;
}

function readHistory() {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function persistHistory(messages) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(messages.slice(-200)));
}

function defaultModelForProvider(provider) {
  return LLM_PROVIDERS.find((item) => item.value === provider)?.defaultModel || "";
}

function labelForProvider(provider) {
  return LLM_PROVIDERS.find((item) => item.value === provider)?.label || provider;
}

function readLlmSettings() {
  try {
    const raw = localStorage.getItem(LLM_SETTINGS_KEY);
    const parsed = raw ? JSON.parse(raw) : {};
    const provider = parsed.provider || "openai";
    return {
      llmProvider: provider,
      llmModel: parsed.model || defaultModelForProvider(provider),
      llmApiKey: parsed.remember ? parsed.apiKey || "" : "",
      rememberLlmSettings: Boolean(parsed.remember),
    };
  } catch {
    return {
      llmProvider: "openai",
      llmModel: defaultModelForProvider("openai"),
      llmApiKey: "",
      rememberLlmSettings: false,
    };
  }
}

function persistLlmSettings(settings) {
  const payload = {
    provider: settings.llmProvider,
    model: settings.llmModel,
    remember: Boolean(settings.rememberLlmSettings),
    ...(settings.rememberLlmSettings ? { apiKey: settings.llmApiKey } : {}),
  };
  localStorage.setItem(LLM_SETTINGS_KEY, JSON.stringify(payload));
}

function buildWhere(config) {
  if (config.scope === "year") return { year: Number(config.year) };
  if (config.scope === "decade" && config.decade !== null) return { decade: config.decade };
  return null;
}

async function postBackend(path, payload) {
  const response = await fetch(`${BACKEND_URL}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(BACKEND_API_KEY ? { "X-API-Key": BACKEND_API_KEY } : {}),
    },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Backend returned ${response.status}`);
  }
  return response.json();
}

async function askBackend(query, config) {
  const payload = {
    query,
    strategy: config.strategy,
    top_k: config.topK,
    rerank: config.rerank,
    where: buildWhere(config),
    auto_year_filter: true,
    ...(config.useLlm ? { llm_provider: config.llmProvider } : {}),
    ...(config.useLlm && config.llmApiKey ? { llm_api_key: config.llmApiKey } : {}),
    ...(config.useLlm && config.llmModel ? { llm_model: config.llmModel } : {}),
  };
  return postBackend(config.useLlm ? "/ask" : "/search", payload);
}

async function getBackendHealth() {
  try {
    const response = await fetch(`${BACKEND_URL}/health`);
    if (!response.ok) throw new Error("offline");
    return response.json();
  } catch {
    return null;
  }
}

function TopBar({ health, provider, onOpenSettings }) {
  return (
    <header className="topbar">
      <div className="brand">
        <div className="brand-mark">
          <img src={`${import.meta.env.BASE_URL}assets/chatbot_avatar.png`} alt="BuffettRAG" />
        </div>
        <div className="brand-text">
          <div className="brand-name">BuffettRAG</div>
          <div className="brand-sub">research console</div>
        </div>
      </div>

      <div className="topbar-divider" />

      <div className="scope-info">
        <div className="scope-label">Corpus</div>
        <div className="scope-value">
          48 letters <span className="muted">/ 1977-2024</span>
        </div>
      </div>

      <div className="topbar-spacer" />

      <div className="status-cluster">
        <span className="status-chip">
          <span className={`dot ${health ? "" : "idle"}`} />
          <span>backend</span>
          <strong>{health ? "online" : "offline"}</strong>
        </span>
        <span className="status-chip">
          <span>chunks</span>
          <strong>{health?.indexed_count ?? "n/a"}</strong>
        </span>
        <span className="status-chip">
          <span>embed</span>
          <strong>bge-small</strong>
        </span>
        <span className="status-chip">
          <span>llm</span>
          <strong>{labelForProvider(provider)}</strong>
        </span>
        <button className="icon-btn" title="Settings" onClick={onOpenSettings}>
          <Settings />
        </button>
      </div>
    </header>
  );
}

function LeftRail({ config, setConfig, onPickExample, onClear }) {
  const setKey = (key, value) => setConfig((current) => ({ ...current, [key]: value }));

  return (
    <aside className="rail-left">
      <div className="rail-section">
        <h3 className="rail-heading"><span className="hash">#</span> Retrieval</h3>

        <div className="field">
          <div className="field-label">Strategy</div>
          <div className="segmented cols-4">
            {["hybrid", "vector", "metadata", "naive"].map((strategy) => (
              <button
                key={strategy}
                className={`seg-btn ${config.strategy === strategy ? "active" : ""}`}
                onClick={() => setKey("strategy", strategy)}
              >
                {strategy}
              </button>
            ))}
          </div>
        </div>

        <div className="field">
          <div className="field-label">
            <span>Top-K passages</span>
            <span className="field-value">{config.topK}</span>
          </div>
          <input
            type="range"
            className="slider"
            min={3}
            max={15}
            value={config.topK}
            onChange={(event) => setKey("topK", Number(event.target.value))}
          />
        </div>

        <Toggle
          label="Rerank (cross-encoder)"
          hint="bge-reranker-base"
          on={config.rerank}
          onChange={(value) => setKey("rerank", value)}
        />
        <Toggle
          label="Generate answer (LLM)"
          hint="off = retrieval only"
          on={config.useLlm}
          onChange={(value) => setKey("useLlm", value)}
        />
      </div>

      <div className="rail-section">
        <h3 className="rail-heading"><span className="hash">#</span> Scope</h3>
        <div className="segmented cols-3">
          {[
            ["all", "All"],
            ["year", "Year"],
            ["decade", "Decade"],
          ].map(([value, label]) => (
            <button
              key={value}
              className={`seg-btn ${config.scope === value ? "active" : ""}`}
              onClick={() => setKey("scope", value)}
            >
              {label}
            </button>
          ))}
        </div>

        {config.scope === "year" && (
          <select
            className="rail-select"
            value={config.year}
            onChange={(event) => setKey("year", Number(event.target.value))}
          >
            {YEARS.map((year) => <option key={year} value={year}>{year}</option>)}
          </select>
        )}

        {config.scope === "decade" && (
          <div className="year-strip">
            {DECADES.map((decade) => (
              <button
                key={String(decade.value)}
                className={`year-cell ${config.decade === decade.value ? "active" : ""}`}
                onClick={() => setKey("decade", decade.value)}
              >
                {decade.label}
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="rail-section examples">
        <h3 className="rail-heading"><span className="hash">#</span> Example prompts</h3>
        <div className="example-list">
          {EXAMPLE_QUERIES.map((query, index) => (
            <button key={query} className="example-card" onClick={() => onPickExample(query)}>
              <span className="example-num">{String(index + 1).padStart(2, "0")}</span>
              <span>{query}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="rail-section rail-footer">
        <div className="backend-url">Backend<br /><code>{BACKEND_URL}</code></div>
        <button className="clear-btn" onClick={onClear}><Trash2 /> Clear chat</button>
      </div>
    </aside>
  );
}

function Toggle({ label, hint, on, onChange }) {
  return (
    <button className="toggle-row" onClick={() => onChange(!on)}>
      <span className="toggle-meta">
        <span className="toggle-label">{label}</span>
        {hint && <span className="toggle-hint">{hint}</span>}
      </span>
      <span className={`toggle-switch ${on ? "on" : ""}`} />
    </button>
  );
}

function LlmSettingsModal({ open, config, setConfig, onClose }) {
  const [draft, setDraft] = React.useState(config);

  React.useEffect(() => {
    if (open) setDraft(config);
  }, [open, config]);

  if (!open) return null;

  const setDraftKey = (key, value) => {
    setDraft((current) => ({ ...current, [key]: value }));
  };

  const chooseProvider = (provider) => {
    setDraft((current) => ({
      ...current,
      llmProvider: provider,
      llmModel:
        !current.llmModel || current.llmModel === defaultModelForProvider(current.llmProvider)
          ? defaultModelForProvider(provider)
          : current.llmModel,
    }));
  };

  const save = () => {
    setConfig((current) => ({ ...current, ...draft }));
    persistLlmSettings(draft);
    onClose();
  };

  const clearKey = () => {
    const next = { ...draft, llmApiKey: "", rememberLlmSettings: false };
    setDraft(next);
    localStorage.removeItem(LLM_SETTINGS_KEY);
    setConfig((current) => ({ ...current, llmApiKey: "", rememberLlmSettings: false }));
  };

  return (
    <div className="modal-backdrop" onMouseDown={onClose}>
      <div className="settings-modal" onMouseDown={(event) => event.stopPropagation()}>
        <div className="modal-head">
          <div>
            <div className="modal-title">LLM settings</div>
            <div className="modal-kicker">Choose the answer engine for this browser.</div>
          </div>
          <button className="icon-btn" onClick={onClose} title="Close">x</button>
        </div>

        <div className="field">
          <div className="field-label">Provider</div>
          <div className="segmented cols-3">
            {LLM_PROVIDERS.map((provider) => (
              <button
                key={provider.value}
                className={`seg-btn ${draft.llmProvider === provider.value ? "active" : ""}`}
                onClick={() => chooseProvider(provider.value)}
              >
                {provider.label}
              </button>
            ))}
          </div>
        </div>

        <div className="field">
          <div className="field-label">Model</div>
          <input
            className="settings-input"
            value={draft.llmModel}
            placeholder={defaultModelForProvider(draft.llmProvider)}
            onChange={(event) => setDraftKey("llmModel", event.target.value)}
          />
        </div>

        <div className="field">
          <div className="field-label">API key</div>
          <input
            className="settings-input"
            type="password"
            value={draft.llmApiKey}
            placeholder={`${labelForProvider(draft.llmProvider)} API key`}
            onChange={(event) => setDraftKey("llmApiKey", event.target.value)}
          />
        </div>

        <Toggle
          label="Remember on this device"
          hint="stores provider settings locally"
          on={draft.rememberLlmSettings}
          onChange={(value) => setDraftKey("rememberLlmSettings", value)}
        />

        <div className="modal-actions">
          <button className="clear-btn" onClick={clearKey}>Clear key</button>
          <button className="save-btn" onClick={save}>Save settings</button>
        </div>
      </div>
    </div>
  );
}

function CiteText({ text }) {
  const chunks = String(text || "").split(/(\[\d+(?:,\s*\d+)*\])/g);
  return chunks.map((chunk, index) => {
    if (/^\[\d+(?:,\s*\d+)*\]$/.test(chunk)) {
      return <span key={index} className="cite-mark">{chunk}</span>;
    }
    return <React.Fragment key={index}>{chunk}</React.Fragment>;
  });
}

function renderBlock(block, index) {
  const lines = block.split("\n").filter(l => l.trim() !== "");
  const isBullet = lines.length >= 2 && lines.every(l => /^\s*[-*]\s+/.test(l));
  const isNumbered = lines.length >= 2 && lines.every(l => /^\s*\d+\.\s+/.test(l));

  if (isBullet) {
    return (
      <ul key={index} className="answer-list">
        {lines.map((l, i) => (
          <li key={i}><CiteText text={l.replace(/^\s*[-*]\s+/, "")} /></li>
        ))}
      </ul>
    );
  }
  if (isNumbered) {
    return (
      <ol key={index} className="answer-list">
        {lines.map((l, i) => (
          <li key={i}><CiteText text={l.replace(/^\s*\d+\.\s+/, "")} /></li>
        ))}
      </ol>
    );
  }
  return <p key={index}><CiteText text={block} /></p>;
}

function Message({ msg, isSelected, onSelectSources }) {
  const isUser = msg.role === "user";
  const avatar = isUser ? `${import.meta.env.BASE_URL}assets/user_avatar.png` : `${import.meta.env.BASE_URL}assets/assistant_avatar.png`;
  const blocks = String(msg.content || "").split(/\n{2,}/).filter(Boolean);

  return (
    <div className={`msg ${isUser ? "user" : "assistant"}`}>
      <div className="msg-avatar"><img src={avatar} alt={msg.role} /></div>
      <div className="msg-body">
        <div className="msg-head">
          <span className={`msg-name ${isUser ? "user" : "assistant"}`}>
            {isUser ? "You" : "BuffettRAG"}
          </span>
          <span className="msg-time">{msg.created_at}</span>
        </div>
        <div className="msg-bubble">
          {blocks.length ? blocks.map((block, index) => renderBlock(block, index)) : <p />}
        </div>

        {!isUser && (
          <div className="msg-meta">
            <span className="meta-pill accent"><span className="k">strat</span><span className="v">{msg.meta?.strategy ?? "n/a"}</span></span>
            <span className="meta-pill"><span className="k">rerank</span><span className="v">{msg.meta?.reranked ? "yes" : "no"}</span></span>
            <span className="meta-pill"><span className="k">filter</span><span className="v">{formatFilter(msg.meta?.used_filter)}</span></span>
            {msg.sources?.length > 0 && (
              <button
                className={`meta-pill action ${isSelected ? "active" : ""}`}
                onClick={() => onSelectSources(msg.id)}
              >
                <Paperclip />
                <span className="v">{msg.sources.length} sources</span>
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function TypingMessage() {
  return (
    <div className="msg assistant">
      <div className="msg-avatar"><img src={`${import.meta.env.BASE_URL}assets/assistant_avatar.png`} alt="assistant" /></div>
      <div className="msg-body">
        <div className="msg-head">
          <span className="msg-name assistant">BuffettRAG</span>
          <span className="msg-time">...</span>
        </div>
        <div className="typing-bubble">
          <span className="typing-dot" />
          <span className="typing-dot" />
          <span className="typing-dot" />
        </div>
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="empty-state">
      <div className="empty-emblem">
        <img src={`${import.meta.env.BASE_URL}assets/chatbot_avatar.png`} alt="" />
      </div>
      <h2 className="empty-title">Ask anything about Buffett's letters</h2>
      <p className="empty-sub">
        Retrieval-augmented search across 48 Berkshire Hathaway shareholder letters.
        Try a question, or pick one from the prompts panel.
      </p>
      <div className="empty-stats">
        <Stat value="48" label="Letters" />
        <Stat value="1977" label="From" />
        <Stat value="2024" label="To" />
      </div>
    </div>
  );
}

function Stat({ value, label }) {
  return (
    <div>
      <div className="empty-stat-value">{value}</div>
      <div className="empty-stat-label">{label}</div>
    </div>
  );
}

function ChatColumn({
  messages,
  isTyping,
  selectedId,
  onSelectSources,
  onSend,
  onClear,
  draft,
  setDraft,
  inputRef,
}) {
  const scrollRef = React.useRef(null);

  React.useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [messages.length, isTyping]);

  const canSend = draft.trim().length > 0 && !isTyping;

  return (
    <main className="chat-col">
      <div className="chat-header">
        <div>
          <div className="chat-title">Research chat</div>
          <div className="chat-sub">retrieval {"->"} rerank {"->"} grounded answer</div>
        </div>
        <div className="chat-header-spacer" />
        <button className="icon-btn" onClick={onClear} title="Clear chat"><Trash2 /></button>
      </div>

      <div className="chat-scroll" ref={scrollRef}>
        {messages.length === 0 && !isTyping ? <EmptyState /> : (
          <>
            {messages.map((message) => (
              <Message
                key={message.id}
                msg={message}
                isSelected={message.id === selectedId}
                onSelectSources={onSelectSources}
              />
            ))}
            {isTyping && <TypingMessage />}
          </>
        )}
      </div>

      <div className="composer-wrap">
        <div className="composer">
          <input
            ref={inputRef}
            className="composer-input"
            placeholder="Ask about Buffett's shareholder letters..."
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                onSend();
              }
            }}
          />
          <button className="composer-send" onClick={onSend} disabled={!canSend} aria-label="Send">
            <Send />
          </button>
        </div>
        <div className="composer-hint">
          <span>Press <kbd>Enter</kbd> to send</span>
          <span>Answers cite passages with <span className="cite-mark">[n]</span></span>
        </div>
      </div>
    </main>
  );
}

function SourceCard({ source, index }) {
  const [open, setOpen] = React.useState(false);
  const score = Number(source.score || 0);
  const topics = String(source.topics || "")
    .split(",")
    .map((topic) => topic.trim())
    .filter(Boolean)
    .slice(0, 4);
  const text = String(source.text || "");
  const preview = text.replace(/\s+/g, " ").slice(0, 320);

  return (
    <div className="source-card">
      <div className="source-card-rail" />
      <div className="source-head">
        <span className="source-year">{source.year ?? "n/a"}</span>
        <span className="source-rank">[{index + 1}]</span>
        <div className="source-head-spacer" />
        <div className="source-score-mini">
          <div className="score-bar">
            <div className="score-bar-fill" style={{ width: `${Math.min(100, Math.max(0, score * 100))}%` }} />
          </div>
          <span>{Number.isFinite(score) ? score.toFixed(3) : "n/a"}</span>
        </div>
      </div>
      <div className="source-file">{source.source_file || "Unknown source"}</div>
      {topics.length > 0 && (
        <div className="source-topics">
          {topics.map((topic) => <span key={topic} className="topic-chip">{topic}</span>)}
        </div>
      )}
      <p className="source-preview">{preview}{text.length > 320 ? "..." : ""}</p>
      <button className={`source-expand ${open ? "open" : ""}`} onClick={() => setOpen((value) => !value)}>
        <span className="chev"><ChevronRight /></span>
        {open ? "Hide full passage" : "View full passage"}
      </button>
      {open && <div className="source-full">{text}</div>}
    </div>
  );
}

function RightRail({ selectedMsg, question }) {
  if (!selectedMsg || !selectedMsg.sources?.length) {
    return (
      <aside className="rail-right">
        <div className="sources-header">
          <div className="sources-title">Sources</div>
          <div className="sources-kicker">Passages retrieved for the selected answer.</div>
        </div>
        <div className="sources-scroll">
          <div className="source-empty">
            <div className="source-empty-icon"><BookOpen /></div>
            <div className="source-empty-title">No answer selected</div>
            <div className="source-empty-text">
              Send a question, then click sources on any answer to inspect retrieved passages here.
            </div>
          </div>
        </div>
      </aside>
    );
  }

  const meta = selectedMsg.meta || {};
  return (
    <aside className="rail-right">
      <div className="sources-header">
        <div className="sources-title">Sources <span className="count-badge">{selectedMsg.sources.length}</span></div>
        {question && <div className="sources-q">"{question}"</div>}
        <div className="sources-summary">
          <span className="summary-pill"><span className="k">strat</span><span className="v">{meta.strategy ?? "n/a"}</span></span>
          <span className="summary-pill"><span className="k">rerank</span><span className="v">{meta.reranked ? "on" : "off"}</span></span>
          <span className="summary-pill"><span className="k">filter</span><span className="v">{formatFilter(meta.used_filter)}</span></span>
          <span className="summary-pill"><span className="k">cites</span><span className="v">{selectedMsg.citations?.length ?? 0}</span></span>
        </div>
      </div>
      <div className="sources-scroll">
        {selectedMsg.sources.map((source, index) => (
          <SourceCard key={source.id || `${source.source_file}-${index}`} source={source} index={index} />
        ))}
      </div>
    </aside>
  );
}

// function formatFilter(filter) {
//   if (!filter) return "none";
//   if (typeof filter === "string") return filter;
//   if (filter.year) return `year:${filter.year}`;
//   if (filter.decade) return `decade:${filter.decade}s`;
//   return "custom";
// }
function formatFilter(filter) {
  if (!filter) return "none";
  if (typeof filter === "string") return filter;

  // range filter: { year: { $gte: 1990, $lte: 2024 } }
  if (filter.year && typeof filter.year === "object") {
    const gte = filter.year["$gte"] ?? "?";
    const lte = filter.year["$lte"] ?? "?";
    return `year:${gte}–${lte}`;
  }

  // single year: { year: 2008 }
  if (filter.year) return `year:${filter.year}`;

  // decade: { decade: 1990 }
  if (filter.decade) return `decade:${filter.decade}s`;

  return "custom";
}

function normalizeAnswer(response, config) {
  const hasAnswer = config.useLlm && response.answer;
  return hasAnswer
    ? response.answer
    : "I found the most relevant passages for this query. Review the source panel for the supporting excerpts.";
}

function App() {
  const [messages, setMessages] = React.useState(readHistory);
  const [selectedId, setSelectedId] = React.useState(() => {
    const lastAssistant = readHistory().filter((msg) => msg.role === "assistant").at(-1);
    return lastAssistant?.id ?? null;
  });
  const [draft, setDraft] = React.useState("");
  const [isTyping, setIsTyping] = React.useState(false);
  const [error, setError] = React.useState("");
  const [health, setHealth] = React.useState(null);
  const [settingsOpen, setSettingsOpen] = React.useState(false);
  const inputRef = React.useRef(null);

  const [config, setConfig] = React.useState({
    strategy: "hybrid",
    topK: 8,
    rerank: true,
    useLlm: true,
    scope: "all",
    year: 2024,
    decade: 1970,
    ...readLlmSettings(),
  });

  React.useEffect(() => {
    getBackendHealth().then(setHealth);
  }, []);

  React.useEffect(() => {
    persistHistory(messages);
  }, [messages]);

  const selectedMsg = React.useMemo(
    () => messages.find((msg) => msg.id === selectedId && msg.role === "assistant"),
    [messages, selectedId]
  );

  const selectedQuestion = React.useMemo(() => {
    if (!selectedMsg) return "";
    const idx = messages.findIndex((msg) => msg.id === selectedMsg.id);
    if (idx <= 0) return "";
    const previous = messages[idx - 1];
    return previous?.role === "user" ? previous.content : "";
  }, [selectedMsg, messages]);

  const send = async () => {
    const query = draft.trim();
    if (!query || isTyping) return;

    setError("");
    const userMessage = {
      id: makeId("u"),
      role: "user",
      content: query,
      created_at: nowLabel(),
    };
    setMessages((current) => [...current, userMessage]);
    setDraft("");
    setIsTyping(true);

    try {
      const response = await askBackend(query, config);
      const assistantMessage = {
        id: makeId("a"),
        role: "assistant",
        created_at: nowLabel(),
        content: normalizeAnswer(response, config),
        sources: response.hits || [],
        citations: response.citations || [],
        meta: {
          strategy: response.strategy ?? config.strategy,
          reranked: response.reranked ?? config.rerank,
          used_filter: response.used_filter ?? buildWhere(config),
          provider: config.llmProvider,
        },
      };
      setMessages((current) => [...current, assistantMessage]);
      setSelectedId(assistantMessage.id);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(`Backend did not respond: ${message}`);
    } finally {
      setIsTyping(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
    setSelectedId(null);
    setError("");
  };

  const pickExample = (query) => {
    setDraft(query);
    inputRef.current?.focus();
  };

  return (
    <>
      <TopBar
        health={health}
        provider={config.llmProvider}
        onOpenSettings={() => setSettingsOpen(true)}
      />
      <LlmSettingsModal
        open={settingsOpen}
        config={config}
        setConfig={setConfig}
        onClose={() => setSettingsOpen(false)}
      />
      {error && <div className="error-banner">{error}</div>}
      <div className="workspace">
        <LeftRail config={config} setConfig={setConfig} onPickExample={pickExample} onClear={clearChat} />
        <ChatColumn
          messages={messages}
          isTyping={isTyping}
          selectedId={selectedId}
          onSelectSources={setSelectedId}
          onSend={send}
          onClear={clearChat}
          draft={draft}
          setDraft={setDraft}
          inputRef={inputRef}
        />
        <RightRail selectedMsg={selectedMsg} question={selectedQuestion} />
      </div>
    </>
  );
}

createRoot(document.getElementById("root")).render(<App />);
