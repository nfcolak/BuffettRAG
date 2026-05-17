/* ===== Seed data for BuffettRAG redesigned UI ===== */

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

const SEED_MESSAGES = [
  {
    id: "u_seed_1",
    role: "user",
    content: "How did Buffett react to the 2008 financial crisis?",
    created_at: "10:42",
  },
  {
    id: "a_seed_1",
    role: "assistant",
    created_at: "10:42",
    content: [
      { text: "Buffett's response to the 2008 crisis was famously contrarian — he framed it as a generational opportunity to deploy capital while others were fearful." },
      { text: null, kind: "spacer" },
      { text: "Three concrete moves stand out from the letters of 2008–2009:" },
      { kind: "list", items: [
        { lead: "Goldman Sachs ($5B)", body: " preferred stock at 10% with warrants, struck during the panic week of September 2008." , cite: 1 },
        { lead: "GE ($3B)", body: " on similar preferred-plus-warrant terms, signalling support for blue-chip industrials.", cite: 2 },
        { lead: "BNSF acquisition ($26B)", body: " announced 2009 — described as an 'all-in wager on the economic future of the United States.'", cite: 3 },
      ]},
      { text: null, kind: "spacer" },
      { text: "He repeatedly rejected the idea of 'waiting for the bottom,' arguing that cash-rich investors who hesitate during panics forfeit the rare moments when great businesses are priced as merely good ones.", cite: 4 },
    ],
    meta: { strategy: "hybrid", reranked: true, used_filter: "decade:2000s", top_k: 8 },
    citations: [
      { idx: 1, year: 2008 }, { idx: 2, year: 2008 }, { idx: 3, year: 2009 }, { idx: 4, year: 2008 },
    ],
    sources: [
      {
        rank: 1,
        year: 2008,
        source_file: "buffet_2008.txt",
        score: 0.892,
        topics: ["financial-crisis", "preferred-stock", "goldman-sachs"],
        preview: "In September we agreed to invest $5 billion in Goldman Sachs preferred stock, with warrants to purchase common at $115 a share. We knew the world was in turmoil, but we also knew that Goldman was a premier institution that would emerge stronger…",
        full: "In September we agreed to invest $5 billion in Goldman Sachs preferred stock, with warrants to purchase common at $115 a share. We knew the world was in turmoil, but we also knew that Goldman was a premier institution that would emerge stronger from the events that were then unfolding. By October the financial system was on the brink, and Berkshire's willingness to step forward with capital — at terms that we believed were fair to both sides — provided some of the public reassurance that was so badly needed.",
      },
      {
        rank: 2,
        year: 2008,
        source_file: "buffet_2008.txt",
        score: 0.864,
        topics: ["fear", "opportunity", "cash"],
        preview: "A simple rule dictates my buying: Be fearful when others are greedy, and be greedy when others are fearful. And most certainly, fear is now widespread, gripping even seasoned investors…",
        full: "A simple rule dictates my buying: Be fearful when others are greedy, and be greedy when others are fearful. And most certainly, fear is now widespread, gripping even seasoned investors. To be sure, investors are right to be wary of highly leveraged entities or businesses in weak competitive positions. But fears regarding the long-term prosperity of the nation's many sound companies make no sense.",
      },
      {
        rank: 3,
        year: 2009,
        source_file: "buffet_2009.txt",
        score: 0.812,
        topics: ["bnsf", "acquisition", "railroads"],
        preview: "Our purchase of Burlington Northern Santa Fe — completed in February — is the largest acquisition in Berkshire's history. It is a major bet on the long-term economic future of the United States…",
        full: "Our purchase of Burlington Northern Santa Fe — completed in February — is the largest acquisition in Berkshire's history. It is a major bet on the long-term economic future of the United States. I love such bets. Our country's social and economic problems are real and serious; I have no doubt we will overcome them as we have so many times before.",
      },
      {
        rank: 4,
        year: 2008,
        source_file: "buffet_2008.txt",
        score: 0.781,
        topics: ["cash-on-hand", "ge", "preferred"],
        preview: "We also committed $3 billion to a similar preferred-and-warrant package for General Electric. In both cases the terms reflected the scarcity of capital at the moment, not any belief that the businesses had become weak…",
        full: "We also committed $3 billion to a similar preferred-and-warrant package for General Electric. In both cases the terms reflected the scarcity of capital at the moment, not any belief that the businesses had become weak. Berkshire entered the crisis with a fortress balance sheet for precisely this reason — to be the buyer of choice when the cost of capital exploded.",
      },
      {
        rank: 5,
        year: 2010,
        source_file: "buffet_2010.txt",
        score: 0.654,
        topics: ["retrospective", "crisis"],
        preview: "Looking back, the 2008 panic offered the kind of pricing dislocation that comes along once in a generation. Many great businesses traded for less than their intrinsic value would justify…",
        full: "Looking back, the 2008 panic offered the kind of pricing dislocation that comes along once in a generation. Many great businesses traded for less than their intrinsic value would justify even in a pessimistic scenario. Investors who waited for the all-clear signal — there is never such a signal — missed the period in which the heaviest lifting could be done at the most attractive prices.",
      },
    ],
  },
];

window.BR_DATA = { EXAMPLE_QUERIES, DECADES, SEED_MESSAGES };
