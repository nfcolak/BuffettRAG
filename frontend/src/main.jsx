import React from "react";
import { createRoot } from "react-dom/client";
import { BookOpen, ChevronRight, Paperclip, Send, Settings, Trash2 } from "lucide-react";

import "./styles.css";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";
const HISTORY_KEY = "buffettrag.messages.v1";

const EXAMPLE_QUERIES = [
  "How did Buffett react to the 2008 financial crisis?",
  "What does Buffett look for when evaluating a company for acquisition?",
  "How has Buffett's view on technology stocks changed from the 1990s to the 2020s?",
  "What did Buffett say about derivatives?",
  "Buffett on inflation and purchasing power",
];

const DECADES = [
  { value: null, label: "All" },
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

function buildWhere(config) {
  if (config.scope === "year") return { year: Number(config.year) };
  if (config.scope === "decade" && config.decade !== null) return { decade: config.decade };
  return null;
}

async function postBackend(path, payload) {
  const response = await fetch(`${BACKEND_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
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

function TopBar({ health }) {
  return (
    <header className="topbar">
      <div className="brand">
        <div className="brand-mark">
          <img src="/assets/header_logo.png" alt="BuffettRAG" />
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
        <button className="icon-btn" title="Settings">
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

function CiteText({ text }) {
  const chunks = String(text || "").split(/(\[\d+(?:,\s*\d+)*\])/g);
  return chunks.map((chunk, index) => {
    if (/^\[\d+(?:,\s*\d+)*\]$/.test(chunk)) {
      return <span key={index} className="cite-mark">{chunk}</span>;
    }
    return <React.Fragment key={index}>{chunk}</React.Fragment>;
  });
}

function Message({ msg, isSelected, onSelectSources }) {
  const isUser = msg.role === "user";
  const avatar = isUser ? "/assets/user_avatar.png" : "/assets/chatbot_avatar.png";
  const paragraphs = String(msg.content || "").split(/\n{2,}/).filter(Boolean);

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
          {paragraphs.length ? paragraphs.map((paragraph, index) => (
            <p key={index}><CiteText text={paragraph} /></p>
          )) : <p />}
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
      <div className="msg-avatar"><img src="/assets/chatbot_avatar.png" alt="assistant" /></div>
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
        <img src="/assets/chatbot_avatar.png" alt="" />
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

function formatFilter(filter) {
  if (!filter) return "none";
  if (typeof filter === "string") return filter;
  if (filter.year) return `year:${filter.year}`;
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
  const inputRef = React.useRef(null);

  const [config, setConfig] = React.useState({
    strategy: "hybrid",
    topK: 8,
    rerank: true,
    useLlm: true,
    scope: "all",
    year: 2024,
    decade: null,
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
      <TopBar health={health} />
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
