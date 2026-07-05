/* ===== BuffettRAG redesigned UI — main app ===== */

const { useState, useRef, useEffect, useMemo } = React;
const { EXAMPLE_QUERIES, DECADES, SEED_MESSAGES } = window.BR_DATA;

// ───── tiny inline icons ─────
const Icon = {
  Send: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round">
      <path d="M22 2L11 13" /><path d="M22 2l-7 20-4-9-9-4 20-7z" />
    </svg>
  ),
  Sparkle: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3z" />
    </svg>
  ),
  Trash: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 6h18" /><path d="M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2" /><path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6" />
    </svg>
  ),
  Settings: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.6 1.6 0 00.3 1.8l.1.1a2 2 0 01-2.8 2.8l-.1-.1a1.6 1.6 0 00-1.8-.3 1.6 1.6 0 00-1 1.5V21a2 2 0 11-4 0v-.1a1.6 1.6 0 00-1-1.5 1.6 1.6 0 00-1.8.3l-.1.1a2 2 0 11-2.8-2.8l.1-.1a1.6 1.6 0 00.3-1.8 1.6 1.6 0 00-1.5-1H3a2 2 0 110-4h.1a1.6 1.6 0 001.5-1 1.6 1.6 0 00-.3-1.8l-.1-.1a2 2 0 112.8-2.8l.1.1a1.6 1.6 0 001.8.3H9a1.6 1.6 0 001-1.5V3a2 2 0 114 0v.1a1.6 1.6 0 001 1.5 1.6 1.6 0 001.8-.3l.1-.1a2 2 0 112.8 2.8l-.1.1a1.6 1.6 0 00-.3 1.8V9a1.6 1.6 0 001.5 1H21a2 2 0 110 4h-.1a1.6 1.6 0 00-1.5 1z" />
    </svg>
  ),
  Doc: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" /><path d="M14 2v6h6" /><path d="M9 13h6" /><path d="M9 17h6" />
    </svg>
  ),
  Paperclip: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 11l-9 9a5 5 0 01-7-7l9-9a3.5 3.5 0 015 5l-9 9a2 2 0 01-3-3l8-8" />
    </svg>
  ),
  Chev: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" style={{ width: 10, height: 10 }}>
      <path d="M9 18l6-6-6-6" />
    </svg>
  ),
  Books: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 19.5A2.5 2.5 0 016.5 17H20" /><path d="M6.5 2H20v20H6.5A2.5 2.5 0 014 19.5v-15A2.5 2.5 0 016.5 2z" />
    </svg>
  ),
};

// ───── Top bar ─────
function TopBar({ scope }) {
  return (
    <header className="topbar">
      <div className="brand">
        <div className="brand-mark">
          <img src="assets/header_logo.png" alt="BuffettRAG" />
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
          48 letters <span className="muted">/ 1977–2024</span>
        </div>
      </div>

      <div className="topbar-spacer" />

      <div className="status-cluster">
        <span className="status-chip">
          <span className="dot" />
          <span>backend</span>
          <strong>online</strong>
        </span>
        <span className="status-chip">
          <span>model</span>
          <strong>openai</strong>
        </span>
        <span className="status-chip">
          <span>embed</span>
          <strong>bge-small</strong>
        </span>
        <button className="icon-btn" title="Settings"><Icon.Settings /></button>
      </div>
    </header>
  );
}

// ───── Left rail ─────
function LeftRail({ config, setConfig, onPickExample }) {
  const setKey = (k, v) => setConfig(c => ({ ...c, [k]: v }));

  return (
    <aside className="rail-left">
      <div className="rail-section">
        <h3 className="rail-heading"><span className="hash">#</span> Retrieval</h3>

        <div className="field">
          <div className="field-label">Strategy</div>
          <div className="segmented cols-4">
            {["hybrid", "vector", "bm25", "naive"].map(s => (
              <button
                key={s}
                className={"seg-btn " + (config.strategy === s ? "active" : "")}
                onClick={() => setKey("strategy", s)}
              >
                {s === "bm25" ? "BM25" : s}
              </button>
            ))}
          </div>
        </div>

        <div className="field">
          <div className="field-label">
            <span>Top-K passages</span>
            <span className="field-value">{config.topK}</span>
          </div>
          <div className="slider-row">
            <input
              type="range"
              className="slider"
              min={3}
              max={15}
              value={config.topK}
              onChange={e => setKey("topK", parseInt(e.target.value, 10))}
            />
          </div>
        </div>

        <div className="field" style={{ marginBottom: 0 }}>
          <Toggle
            label="Rerank (cross-encoder)"
            hint="bge-reranker-base"
            on={config.rerank}
            onChange={v => setKey("rerank", v)}
          />
          <Toggle
            label="Generate answer (LLM)"
            hint="off = retrieval only"
            on={config.useLlm}
            onChange={v => setKey("useLlm", v)}
          />
        </div>
      </div>

      <div className="rail-section">
        <h3 className="rail-heading"><span className="hash">#</span> Decade filter</h3>
        <div className="year-strip">
          {DECADES.map(d => (
            <button
              key={String(d.value)}
              className={"year-cell " + (config.decade === d.value ? "active" : "")}
              onClick={() => setKey("decade", d.value)}
              title={d.value === null ? "All decades" : `${d.value}s`}
            >
              {d.label}
            </button>
          ))}
        </div>
      </div>

      <div className="rail-section" style={{ flex: 1 }}>
        <h3 className="rail-heading"><span className="hash">#</span> Example prompts</h3>
        <div className="example-list">
          {EXAMPLE_QUERIES.map((q, i) => (
            <button key={i} className="example-card" onClick={() => onPickExample(q)}>
              <span className="example-num">{String(i + 1).padStart(2, "0")}</span>
              <span>{q}</span>
            </button>
          ))}
        </div>
      </div>
    </aside>
  );
}

function Toggle({ label, hint, on, onChange }) {
  return (
    <div className="toggle-row" onClick={() => onChange(!on)}>
      <div className="toggle-meta">
        <div className="toggle-label">{label}</div>
        {hint && <div className="toggle-hint">{hint}</div>}
      </div>
      <div className={"toggle-switch " + (on ? "on" : "")} />
    </div>
  );
}

// ───── Message renderer ─────
function renderAssistantContent(content) {
  return content.map((block, i) => {
    if (block.kind === "spacer") return <div key={i} style={{ height: 4 }} />;
    if (block.kind === "list") {
      return (
        <ul key={i} style={{ margin: "4px 0 4px 4px", paddingLeft: 18, listStyle: "none" }}>
          {block.items.map((it, j) => (
            <li key={j} style={{ position: "relative", paddingLeft: 14, marginBottom: 6, color: "var(--text-primary)" }}>
              <span style={{ position: "absolute", left: 0, top: 8, width: 6, height: 6, borderRadius: "50%", background: "var(--accent)", opacity: 0.7 }} />
              <strong>{it.lead}</strong>
              {it.body}
              {it.cite && <CiteMark idx={it.cite} />}
            </li>
          ))}
        </ul>
      );
    }
    return (
      <p key={i}>
        {block.text}
        {block.cite && <CiteMark idx={block.cite} />}
      </p>
    );
  });
}

function CiteMark({ idx }) {
  return <span className="cite-mark" title={`Source [${idx}]`}>[{idx}]</span>;
}

function Message({ msg, isSelected, onSelectSources }) {
  const isUser = msg.role === "user";
  const avatar = isUser ? "assets/user_avatar.png" : "assets/chatbot_avatar.png";
  return (
    <div className={`msg ${isUser ? "user" : "assistant"}`}>
      <div className="msg-avatar"><img src={avatar} alt={msg.role} /></div>
      <div className="msg-body">
        <div className="msg-head">
          <span className={"msg-name " + (isUser ? "user" : "assistant")}>
            {isUser ? "You" : "BuffettRAG"}
          </span>
          <span className="msg-time">{msg.created_at}</span>
        </div>
        <div className="msg-bubble">
          {Array.isArray(msg.content) ? renderAssistantContent(msg.content) : <p>{msg.content}</p>}
        </div>

        {!isUser && msg.meta && (
          <div className="msg-meta">
            <span className="meta-pill accent">
              <span className="k">strat</span><span className="v">{msg.meta.strategy}</span>
            </span>
            <span className="meta-pill">
              <span className="k">rerank</span><span className="v">{msg.meta.reranked ? "yes" : "no"}</span>
            </span>
            <span className="meta-pill">
              <span className="k">k</span><span className="v">{msg.meta.top_k}</span>
            </span>
            {msg.meta.used_filter && (
              <span className="meta-pill">
                <span className="k">filter</span><span className="v">{msg.meta.used_filter}</span>
              </span>
            )}
            {msg.sources && msg.sources.length > 0 && (
              <button
                className={"meta-pill action " + (isSelected ? "active" : "")}
                onClick={() => onSelectSources(msg.id)}
              >
                <Icon.Paperclip />
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
      <div className="msg-avatar"><img src="assets/chatbot_avatar.png" alt="assistant" /></div>
      <div className="msg-body">
        <div className="msg-head">
          <span className="msg-name assistant">BuffettRAG</span>
          <span className="msg-time">…</span>
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

// ───── Empty state ─────
function EmptyState() {
  return (
    <div className="empty-state">
      <div className="empty-emblem">
        <img src="assets/chatbot_avatar.png" alt="" />
      </div>
      <h2 className="empty-title">Ask anything about Buffett's letters</h2>
      <p className="empty-sub">
        Retrieval-augmented search across 48 Berkshire Hathaway shareholder letters.
        Try a question, or pick one from the prompts panel.
      </p>
      <div className="empty-stats">
        <div>
          <div className="empty-stat-value">48</div>
          <div className="empty-stat-label">Letters</div>
        </div>
        <div>
          <div className="empty-stat-value">1977</div>
          <div className="empty-stat-label">From</div>
        </div>
        <div>
          <div className="empty-stat-value">2024</div>
          <div className="empty-stat-label">To</div>
        </div>
        <div>
          <div className="empty-stat-value">~12k</div>
          <div className="empty-stat-label">Chunks</div>
        </div>
      </div>
    </div>
  );
}

// ───── Center chat ─────
function ChatColumn({ messages, isTyping, selectedId, onSelectSources, onSend, onClear, draft, setDraft, inputRef }) {
  const scrollRef = useRef(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [messages.length, isTyping]);

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
  };

  const canSend = draft.trim().length > 0;

  return (
    <div className="chat-col">
      <div className="chat-header">
        <div>
          <div className="chat-title">Research chat</div>
          <div className="chat-sub">retrieval → rerank → grounded answer</div>
        </div>
        <div className="chat-header-spacer" />
        <div className="chat-actions">
          <button className="icon-btn" onClick={onClear} title="Clear chat"><Icon.Trash /></button>
        </div>
      </div>

      <div className="chat-scroll" ref={scrollRef}>
        {messages.length === 0 && !isTyping ? (
          <EmptyState />
        ) : (
          <>
            {messages.map(m => (
              <div key={m.id} className="msg-group">
                <Message
                  msg={m}
                  isSelected={m.id === selectedId}
                  onSelectSources={onSelectSources}
                />
              </div>
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
            placeholder="Ask about Buffett's shareholder letters…"
            value={draft}
            onChange={e => setDraft(e.target.value)}
            onKeyDown={handleKey}
          />
          <button className="composer-send" onClick={onSend} disabled={!canSend} aria-label="Send">
            <Icon.Send />
          </button>
        </div>
        <div className="composer-hint">
          <span>
            Press <kbd>↵</kbd> to send · <kbd>Shift</kbd>+<kbd>↵</kbd> for newline
          </span>
          <span>Answers cite passages with <span className="cite-mark" style={{ marginLeft: 2 }}>[n]</span></span>
        </div>
      </div>
    </div>
  );
}

// ───── Right rail: sources ─────
function SourceCard({ source }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="source-card">
      <div className="source-card-rail" />
      <div className="source-head">
        <span className="source-year">{source.year}</span>
        <span className="source-rank">[{source.rank}]</span>
        <div className="source-head-spacer" />
        <div className="source-score-mini">
          <div className="score-bar">
            <div className="score-bar-fill" style={{ width: `${Math.min(100, source.score * 100)}%` }} />
          </div>
          <span>{source.score.toFixed(3)}</span>
        </div>
      </div>
      <div className="source-file">{source.source_file}</div>
      <div className="source-topics">
        {source.topics.map((t, i) => (
          <span key={i} className="topic-chip">{t}</span>
        ))}
      </div>
      <p className="source-preview">{source.preview}</p>
      <button className={"source-expand " + (open ? "open" : "")} onClick={() => setOpen(o => !o)}>
        <span className="chev"><Icon.Chev /></span>
        {open ? "Hide full passage" : "View full passage"}
      </button>
      {open && <div className="source-full">{source.full}</div>}
    </div>
  );
}

function RightRail({ selectedMsg, question }) {
  if (!selectedMsg || !selectedMsg.sources || selectedMsg.sources.length === 0) {
    return (
      <aside className="rail-right">
        <div className="sources-header">
          <div className="sources-title-row">
            <div className="sources-title">
              Sources
            </div>
          </div>
          <div style={{ fontSize: 11.5, color: "var(--text-muted)", marginTop: 2 }}>
            Passages retrieved for the selected answer.
          </div>
        </div>
        <div className="sources-scroll">
          <div className="source-empty">
            <div className="source-empty-icon"><Icon.Books /></div>
            <div className="source-empty-title">No answer selected</div>
            <div className="source-empty-text">
              Send a question — then click <strong>sources</strong> on any answer to inspect the retrieved passages here.
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
        <div className="sources-title-row">
          <div className="sources-title">
            Sources <span className="count-badge">{selectedMsg.sources.length}</span>
          </div>
        </div>
        {question && <div className="sources-q">"{question}"</div>}
        <div className="sources-summary">
          <span className="summary-pill"><span className="k">strat</span><span className="v">{meta.strategy}</span></span>
          <span className="summary-pill"><span className="k">rerank</span><span className="v">{meta.reranked ? "on" : "off"}</span></span>
          <span className="summary-pill"><span className="k">filter</span><span className="v">{meta.used_filter || "none"}</span></span>
          <span className="summary-pill"><span className="k">cites</span><span className="v">{(selectedMsg.citations || []).length}</span></span>
        </div>
      </div>
      <div className="sources-scroll">
        {selectedMsg.sources.map(s => <SourceCard key={s.rank} source={s} />)}
      </div>
    </aside>
  );
}

// ───── Mock answer generator ─────
function mockAnswer(query, config) {
  const lc = query.toLowerCase();
  if (lc.includes("derivative")) {
    return {
      content: [
        { text: "Buffett famously called derivatives 'financial weapons of mass destruction' in the 2002 letter — language he later said he stood by despite Berkshire's own (selective) use of equity-index put contracts." , cite: 1 },
        { text: null, kind: "spacer" },
        { text: "His core objections were threefold: counterparty risk that compounds in a crisis, mark-to-model valuations that 'devolve into mark-to-myth,' and pay structures that reward originators while embedding tail risk in the balance sheet.", cite: 2 },
      ],
      sources: [
        { rank: 1, year: 2002, source_file: "buffet_2002.txt", score: 0.91, topics: ["derivatives", "risk"], preview: "We view derivatives as time bombs, both for the parties that deal in them and the economic system. In our view, derivatives are financial weapons of mass destruction…", full: "We view derivatives as time bombs, both for the parties that deal in them and the economic system. In our view, derivatives are financial weapons of mass destruction, carrying dangers that, while now latent, are potentially lethal. Charlie and I are of one mind in how we feel about derivatives and the trading activities that go with them." },
        { rank: 2, year: 2008, source_file: "buffet_2008.txt", score: 0.84, topics: ["derivatives", "mark-to-model"], preview: "I should also mention that the carrying value of these contracts is calculated using the Black-Scholes formula… in extreme cases mark-to-model can devolve into mark-to-myth.", full: "I should also mention that the carrying value of these contracts is calculated using the Black-Scholes formula. The qualifier 'subject to the limitations of the formula' applies, however; in extreme cases mark-to-model can devolve into mark-to-myth. Black-Scholes was not designed for the long-dated contracts we have written." },
      ],
    };
  }
  if (lc.includes("inflation") || lc.includes("purchasing power")) {
    return {
      content: [
        { text: "Buffett's writing on inflation centers on a single idea: the real enemy of long-term investors is not market volatility but the steady erosion of purchasing power." , cite: 1 },
        { text: null, kind: "spacer" },
        { text: "The defensive playbook he describes across decades emphasizes:" },
        { kind: "list", items: [
          { lead: "Pricing power", body: " — businesses able to raise prices without losing customers (See's Candies is the canonical example).", cite: 2 },
          { lead: "Low capital intensity", body: " — companies that don't need to constantly reinvest at inflated replacement costs." , cite: 3 },
          { lead: "Ownership stakes in productive assets", body: " over long-dated bonds whose coupons are paid in depreciating currency." },
        ]},
      ],
      sources: [
        { rank: 1, year: 1980, source_file: "buffet_1980.txt", score: 0.88, topics: ["inflation"], preview: "Inflation acts as a gigantic corporate tapeworm. That tapeworm preemptively consumes its requisite daily diet of investment dollars regardless of the health of the host organism…", full: "Inflation acts as a gigantic corporate tapeworm. That tapeworm preemptively consumes its requisite daily diet of investment dollars regardless of the health of the host organism. Whatever the level of reported profits, more dollars must be put into receivables, inventory, and fixed assets to do the same amount of physical business." },
        { rank: 2, year: 1983, source_file: "buffet_1983.txt", score: 0.82, topics: ["pricing-power", "moat"], preview: "The single most important decision in evaluating a business is pricing power. If you've got the power to raise prices without losing business to a competitor, you've got a very good business…", full: "The single most important decision in evaluating a business is pricing power. If you've got the power to raise prices without losing business to a competitor, you've got a very good business. And if you have to have a prayer session before raising the price by 10 percent, then you've got a terrible business." },
        { rank: 3, year: 1981, source_file: "buffet_1981.txt", score: 0.78, topics: ["capital", "inflation"], preview: "In an inflationary world, a business with a return on equity well above the cost of capital is enormously valuable — and the lower its capital needs, the better…", full: "In an inflationary world, a business with a return on equity well above the cost of capital is enormously valuable — and the lower its capital needs, the better. The opposite is true of capital-hungry businesses, which must continually reinvest at ever-higher replacement costs simply to stay in place." },
      ],
    };
  }
  if (lc.includes("acquisition") || lc.includes("evaluating") || lc.includes("acquire")) {
    return {
      content: [
        { text: "Berkshire's acquisition criteria have been published almost verbatim in every annual letter since the early 1980s. They are deliberately simple and biased toward filtering out far more deals than they admit." , cite: 1 },
        { kind: "list", items: [
          { lead: "Size", body: " — large purchases (pre-tax earnings of at least $75M unless they fit an existing subsidiary).", cite: 1 },
          { lead: "Demonstrated consistent earning power", body: " — projections are of no interest.", cite: 1 },
          { lead: "Good returns on equity", body: " while employing little or no debt.", cite: 1 },
          { lead: "Management in place", body: " — Berkshire does not supply operators.", cite: 1 },
          { lead: "Simple business", body: " — if there's lots of technology, we won't understand it.", cite: 1 },
          { lead: "An offering price", body: " — discussions are a waste of time without one.", cite: 1 },
        ]},
      ],
      sources: [
        { rank: 1, year: 2014, source_file: "buffet_2014.txt", score: 0.93, topics: ["acquisitions", "criteria"], preview: "Here's what we're looking for: (1) Large purchases (at least $75 million of pre-tax earnings unless the business will fit into one of our existing units)…", full: "Here's what we're looking for: (1) Large purchases (at least $75 million of pre-tax earnings unless the business will fit into one of our existing units), (2) demonstrated consistent earning power (future projections are of no interest to us, nor are 'turnaround' situations), (3) businesses earning good returns on equity while employing little or no debt, (4) management in place (we can't supply it), (5) simple businesses (if there's lots of technology, we won't understand it), and (6) an offering price (we don't want to waste our time or that of the seller by talking, even preliminarily, about a transaction when price is unknown)." },
        { rank: 2, year: 1996, source_file: "buffet_1996.txt", score: 0.79, topics: ["circle-of-competence"], preview: "What an investor needs is the ability to correctly evaluate selected businesses. Note that word 'selected': you don't have to be an expert on every company, or even many. You only have to be able to evaluate companies within your circle of competence…", full: "What an investor needs is the ability to correctly evaluate selected businesses. Note that word 'selected': you don't have to be an expert on every company, or even many. You only have to be able to evaluate companies within your circle of competence. The size of that circle is not very important; knowing its boundaries, however, is vital." },
      ],
    };
  }
  if (lc.includes("technology") || lc.includes("tech")) {
    return {
      content: [
        { text: "Buffett's stance on technology evolved markedly. Through the 1990s he openly avoided the sector, citing a 'circle of competence' he didn't intend to expand." , cite: 1 },
        { text: "From roughly 2011 onward — beginning with IBM and, much more consequentially, the Apple position established in 2016 — he reframed certain tech companies as consumer-products businesses whose economics he could evaluate.", cite: 2 },
        { text: "By 2023 the Apple stake had become Berkshire's largest single investment, which he described as fitting his definition of an 'inevitable' — a brand-led business with deep customer captivity rather than a bet on a particular technology cycle.", cite: 3 },
      ],
      sources: [
        { rank: 1, year: 1999, source_file: "buffet_1999.txt", score: 0.84, topics: ["circle-of-competence", "tech"], preview: "We have no insights into which participants in the tech field possess a truly durable competitive advantage… we don't have the foggiest notion who the winners will be.", full: "We have no insights into which participants in the tech field possess a truly durable competitive advantage. Our problem — which we can't solve by studying up — is that we have no insights into which participants in the tech field possess a truly durable competitive advantage." },
        { rank: 2, year: 2016, source_file: "buffet_2016.txt", score: 0.88, topics: ["apple", "consumer-brands"], preview: "Berkshire's investment in Apple has been substantial. We see Apple as a consumer-products company with extraordinary customer loyalty rather than a technology investment…", full: "Berkshire's investment in Apple has been substantial. We see Apple as a consumer-products company with extraordinary customer loyalty rather than a technology investment in the conventional sense. The 'ecosystem' around the iPhone produces the kind of recurring, sticky relationship that we've always prized in any business." },
        { rank: 3, year: 2023, source_file: "buffet_2023.txt", score: 0.81, topics: ["apple", "concentration"], preview: "Apple remains our largest investment by a wide margin. Tim Cook and his team have continued to compound the franchise…", full: "Apple remains our largest investment by a wide margin. Tim Cook and his team have continued to compound the franchise. It is, by my reckoning, the second-best business we own — only somewhat behind the family of insurance operations." },
      ],
    };
  }
  // generic answer
  return {
    content: [
      { text: `Here is a synthesis of the most relevant passages from Buffett's letters on this question.`},
      { text: null, kind: "spacer" },
      { text: "The retrieval surfaced material concentrated in a handful of letters — the most prominent themes are summarized in the response panel on the right.", cite: 1 },
      { text: "Use the controls in the left rail to widen the decade filter, increase top-K, or re-run with reranking off if you'd like to compare retrieval quality side-by-side." },
    ],
    sources: [
      { rank: 1, year: 1996, source_file: "buffet_1996.txt", score: 0.82, topics: ["philosophy", "long-term"], preview: "Our favorite holding period is forever. We are just the opposite of those who hurry to sell and book profits when companies perform well…", full: "Our favorite holding period is forever. We are just the opposite of those who hurry to sell and book profits when companies perform well but who tenaciously hang on to businesses that disappoint. Peter Lynch aptly likens such behavior to cutting the flowers and watering the weeds." },
      { rank: 2, year: 2007, source_file: "buffet_2007.txt", score: 0.74, topics: ["moat", "durable-advantage"], preview: "A truly great business must have an enduring 'moat' that protects excellent returns on invested capital…", full: "A truly great business must have an enduring 'moat' that protects excellent returns on invested capital. The dynamics of capitalism guarantee that competitors will repeatedly assault any business 'castle' that is earning high returns." },
      { rank: 3, year: 2018, source_file: "buffet_2018.txt", score: 0.69, topics: ["intrinsic-value"], preview: "Charlie and I urge you to focus on operating earnings, which were little changed in 2018, and to ignore both quarterly and annual gains or losses from investments…", full: "Charlie and I urge you to focus on operating earnings, which were little changed in 2018, and to ignore both quarterly and annual gains or losses from investments, whether realized or unrealized. Our advice in no way diminishes the importance of these investments to Berkshire." },
    ],
  };
}

// ───── App ─────
function App() {
  const [messages, setMessages] = useState(SEED_MESSAGES);
  const [selectedId, setSelectedId] = useState(SEED_MESSAGES[SEED_MESSAGES.length - 1].id);
  const [draft, setDraft] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const inputRef = useRef(null);

  const [config, setConfig] = useState({
    strategy: "hybrid",
    topK: 8,
    rerank: true,
    useLlm: true,
    decade: null,
  });

  const selectedMsg = useMemo(
    () => messages.find(m => m.id === selectedId && m.role === "assistant"),
    [messages, selectedId]
  );

  const selectedQuestion = useMemo(() => {
    if (!selectedMsg) return "";
    const idx = messages.findIndex(m => m.id === selectedMsg.id);
    if (idx <= 0) return "";
    const prev = messages[idx - 1];
    return prev && typeof prev.content === "string" ? prev.content : "";
  }, [selectedMsg, messages]);

  const send = () => {
    const q = draft.trim();
    if (!q) return;
    const now = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    const userMsg = {
      id: "u_" + Date.now(),
      role: "user",
      content: q,
      created_at: now,
    };
    setMessages(prev => [...prev, userMsg]);
    setDraft("");
    setIsTyping(true);

    setTimeout(() => {
      const mock = mockAnswer(q, config);
      const filterStr = config.decade ? `decade:${config.decade}s` : null;
      const a = {
        id: "a_" + Date.now(),
        role: "assistant",
        created_at: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        content: mock.content,
        sources: mock.sources,
        citations: mock.sources.map(s => ({ idx: s.rank, year: s.year })),
        meta: {
          strategy: config.strategy,
          reranked: config.rerank,
          top_k: config.topK,
          used_filter: filterStr,
        },
      };
      setMessages(prev => [...prev, a]);
      setSelectedId(a.id);
      setIsTyping(false);
    }, 950);
  };

  const onPickExample = (q) => {
    setDraft(q);
    if (inputRef.current) inputRef.current.focus();
  };

  const onClear = () => {
    setMessages([]);
    setSelectedId(null);
  };

  return (
    <>
      <TopBar />
      <div className="workspace">
        <LeftRail config={config} setConfig={setConfig} onPickExample={onPickExample} />
        <ChatColumn
          messages={messages}
          isTyping={isTyping}
          selectedId={selectedId}
          onSelectSources={setSelectedId}
          onSend={send}
          onClear={onClear}
          draft={draft}
          setDraft={setDraft}
          inputRef={inputRef}
        />
        <RightRail selectedMsg={selectedMsg} question={selectedQuestion} />
      </div>
    </>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
