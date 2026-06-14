# Agentic Engineering, From Scratch — A Study Plan

A build-first path from "I can call an LLM API" to "I understand agents, RAG, cost,
and model selection well enough to design my own systems." No LangGraph, no
LlamaIndex, no agent frameworks. Just Python, `numpy`, an HTTP client, and one or
two model APIs. Frameworks are added *only after* you've built the thing they
abstract — so you understand what they're hiding.

**Who this is for:** a working software engineer new to AI/ML. You already know how
to write code, read docs, and debug. What's new is the model and everything around it.

**The one rule:** every phase ends with something you *built*, not something you read.
Reading is for unblocking the build, not for replacing it.

**Pacing:** the week estimates assume ~6–10 focused hours/week. Go slower if needed.
The phases are ordered so each one is rewarding before the next one gets hard — that's
deliberate. Don't skip ahead; the later phases reuse code and intuition from earlier ones.

---

## Phase 0 — Talk to a model (afternoon win)

**Goal:** demystify "the model." It's an HTTP endpoint that takes a list of messages
and returns text. That's it.

**Concepts to internalize**
- The request/response shape: a `messages` array of `{role, content}` where role is
  `system`, `user`, or `assistant`. The model is *stateless* — you resend the whole
  conversation every turn. This single fact explains most of what comes later (cost,
  context limits, memory).
- System prompt vs user prompt vs assistant turns.
- Streaming vs. non-streaming responses.

**Build**
- A CLI chat loop in ~50 lines. Keep an in-memory list of messages, append the user
  turn, call the API, append the response, repeat. Add streaming so tokens print as
  they arrive.
- Deliberately let the conversation get long and watch what happens to latency and
  (later) cost. Notice you're resending everything each turn.

**You've got it when** you can explain why the model "forgets" earlier turns if you
don't resend them, and why a long conversation costs more *per turn* over time.

---

## Phase 1 — Tokenization

**Goal:** stop treating tokens as a billing mystery. Understand what the model
actually sees.

**Concepts to internalize**
- Models don't see characters or words — they see tokens (sub-word units). Byte-Pair
  Encoding (BPE) is the common scheme.
- Why this explains real behavior: why models are bad at character-level tasks
  (spelling, counting letters, reversing strings), why numbers tokenize weirdly, why
  some languages cost more tokens than others, why a "100-page PDF" might be 60k tokens.
- Context window = a hard token budget shared by input + output.

**Build**
- Implement BPE from scratch: train a merge table on a corpus, then write `encode` and
  `decode`. Start at the byte level. Karpathy's `minbpe` is the canonical reference —
  watch his "Let's build the GPT Tokenizer" video, then write your own before looking
  at his.
- Write a `count_tokens(text)` helper using the real tokenizer for your chosen API
  (e.g. `tiktoken` for OpenAI-family, the SDK's counting utility for Anthropic). Use it
  to print the token count of every message in your Phase 0 chat app.

**You've got it when** you can predict, roughly, whether a given prompt will be cheap or
expensive *before* you send it, and explain why "strawberry has how many r's" is hard.

---

## Phase 2 — Sampling and generation control

**Goal:** understand the knobs, because you'll tune them constantly later.

**Concepts to internalize**
- The model outputs a probability distribution over the next token; *sampling* picks
  from it. `temperature`, `top_p`, `top_k` reshape that distribution.
- `max_tokens`, `stop` sequences, and why output tokens usually cost more than input.
- Determinism: why temperature 0 is "more deterministic" but not guaranteed identical,
  and why this matters for testing.

**Build**
- Take one prompt, sweep temperature from 0 to 1.5, and log the outputs side by side.
  Do the same for `top_p`. Build a tiny intuition table for yourself.
- Add a `stop` sequence to your chat app to cut off generation cleanly.

**You've got it when** you can choose sampling settings for a task on purpose
(extraction → low temp, brainstorming → higher) instead of by superstition.

---

## Phase 3 — Cost and model selection / routing

**Goal:** this is one of the three things your friend asked about. Treat cost and model
choice as a *design decision*, not an afterthought.

**Concepts to internalize**
- Pricing is per-token, split into input and output, and output is usually several times
  more expensive. Cached/"prompt caching" tokens are cheaper still.
- The quality / latency / cost triangle. Big frontier models are smart, slow, expensive;
  small models are fast and cheap and fine for easy subtasks. Most production systems
  are a *mix*.
- **Routing**: send easy requests to a cheap model, hard ones to an expensive one.
  Classification/extraction → small model; multi-step reasoning → large model.
- Prompt caching (reuse a long static system prompt cheaply) and batching.

**Build**
- Wrap your API client in a `CostTracker` that logs input tokens, output tokens, and
  computes dollar cost per call from a pricing table you maintain. Print a running total.
- Build a dead-simple **router**: a cheap model first classifies the task ("simple" vs
  "complex"), then you dispatch to the appropriate model. Measure cost and quality
  difference on a batch of 20 mixed prompts.
- Experiment with prompt caching on a long shared system prompt and confirm the cost drop.

**You've got it when** you can take a feature spec and sketch which model handles which
step and why, with a rough cost estimate per request.

---

## Phase 4 — Embeddings and semantic search

**Goal:** the foundation under RAG. Build it *before* RAG so RAG isn't magic.

**Concepts to internalize**
- An embedding maps text to a vector; similar meanings land near each other. Cosine
  similarity measures "nearness."
- A vector store is, at its core, a list of vectors plus a nearest-neighbor search. You
  do not need Pinecone/FAISS to understand it — `numpy` is enough at small scale.
- Chunking: documents must be split before embedding. Chunk size and overlap are
  consequential and underrated.

**Build**
- Write `embed(texts)` calling an embeddings API. Store vectors in a `numpy` array.
- Implement cosine similarity and a `search(query, k)` that returns the top-k chunks —
  by hand, no library.
- Take a handful of your own documents (notes, docs, a wiki dump), chunk them a few
  different ways (fixed size, sentence-aware, with/without overlap), and *eyeball how
  retrieval quality changes*. This is the lesson, not the code.

**You've got it when** you can explain why a bad chunking strategy silently wrecks a RAG
system even when the model and embeddings are good.

---

## Phase 5 — RAG from scratch

**Goal:** the headline request. Build the whole pipeline yourself, end to end, no
framework. This is where the previous four phases pay off.

**Concepts to internalize — the pipeline**
1. **Ingest** raw documents.
2. **Chunk** them (Phase 4).
3. **Embed** the chunks (Phase 4) and store vectors + the original text + metadata.
4. **Retrieve**: embed the user's query, find top-k chunks (Phase 4).
5. **Augment**: build a prompt that injects the retrieved chunks as context.
6. **Generate**: call the model (Phases 0–2) and return the answer, ideally with
   citations back to the source chunks.

**Build (do these in order, each is a layer)**
- **v1 — naive RAG:** the six steps above, minimal. Get it working end to end on your
  own corpus. Celebrate; this already feels like magic.
- **v2 — make it honest:** add source citations, and make the model say "I don't know"
  when retrieval returns nothing relevant. Add a similarity-score threshold.
- **v3 — better retrieval:**
  - *Hybrid search*: combine dense (embedding) search with keyword search (implement
    BM25 yourself — it's a tractable, illuminating formula) and merge the rankings.
  - *Reranking*: over-retrieve (say top 20), then use a reranker model (or a cheap LLM
    call) to reorder to the best 5.
  - *Query rewriting*: have a cheap model rewrite the user's question into a better
    search query before retrieving.
- **v4 — evaluate it (critical, usually skipped):** build a small eval set of
  question→expected-answer pairs over your corpus. Measure retrieval quality
  (did the right chunk get retrieved?) separately from answer quality. You cannot
  improve what you don't measure, and RAG fails in two distinct places — retrieval and
  generation — so measure them separately.

**You've got it when** you can diagnose a wrong RAG answer as either "retrieval pulled the
wrong context" or "right context, model reasoned badly" — and know they need different fixes.

---

## Phase 6 — Tool use and structured output (the bridge to agents)

**Goal:** see that "function calling" is not special model magic — it's structured
generation plus a loop you write.

**Concepts to internalize**
- *Structured output / JSON mode*: constraining the model to emit parseable JSON.
- *Tool calling*: you describe available tools (name, params as a schema); the model
  responds with a request to call one; **your code executes it** and feeds the result
  back; the model continues. The model never runs anything itself — you do.
- This is just a loop: model says "call `get_weather(city=...)`" → you run it → you
  append the result as a new message → you call the model again.

**Build**
- Define 2–3 real tools as plain Python functions: a calculator, a web search, a
  "read file" function. Write JSON schemas for them.
- Implement the tool-calling loop by hand: parse the model's tool request, dispatch to
  the function, append the result, re-call. No framework.
- Handle the messy parts yourself: malformed JSON from the model, a tool that errors, the
  model trying to call a tool that doesn't exist. This error-handling *is* the real work.

**You've got it when** you realize an "agent" is mostly this loop with a stopping
condition, and you could have guessed how every agent framework works internally.

---

## Phase 7 — Agents from scratch

**Goal:** the second headline. Build a real agent loop with no orchestration library.
Read Anthropic's "Building Effective Agents" *now* — you'll finally have the context to
get it, and its whole point is "prefer simple composable patterns over frameworks,"
which is exactly what you've been doing.

**Concepts to internalize**
- The distinction Anthropic draws: **workflows** (you orchestrate the steps in code,
  predictable) vs. **agents** (the model decides the next step dynamically, in a loop).
  Most useful systems are workflows; reach for a true agent only when the path can't be
  predicted in advance.
- The **ReAct** loop: *reason → act (call a tool) → observe (get the result) → repeat*
  until done. It's Phase 6's loop plus a planning/reflection step and a stop condition.
- State and memory: what to keep in context, what to summarize, what to store externally.
- The composable patterns from the Anthropic guide: prompt chaining, routing (you built
  this in Phase 3!), parallelization, orchestrator-workers, evaluator-optimizer.

**Build**
- A **ReAct agent** that takes a goal, loops (think → pick tool → run → observe), and
  stops when it has an answer or hits a max-step budget. Reuse your Phase 6 tools.
- Add a *cost and step budget* so a confused agent can't loop forever burning money —
  this connects directly back to Phase 3.
- Build one **workflow** version of the same task (you orchestrate the steps) and compare
  it to the agent version on reliability, cost, and latency. Feel the tradeoff yourself.
- Then read the Anthropic claude-cookbooks `patterns/agents` reference implementations
  and compare them to what you built. Note what they do that you didn't (and why).

**You've got it when** you can argue, for a given problem, whether it should be a fixed
workflow or a true agent — and defend the choice on cost and reliability, not vibes.

---

## Phase 8 — Evaluation, observability, and production reality

**Goal:** the difference between a demo and a system. Weave this in from Phase 5 onward;
formalize it here.

**Concepts to internalize**
- Why evaluating LLM systems is genuinely hard (no single right answer) and the common
  approaches: golden datasets, LLM-as-judge (and its pitfalls), task-specific metrics.
- Tracing: logging every model call, tool call, token count, and cost in a multi-step
  run so you can debug *which* step failed.
- Guardrails, retries with backoff, timeouts, and graceful degradation.
- Failure modes: hallucination, prompt injection (especially once tools touch real
  data/the web), runaway loops, silent retrieval failures.

**Build**
- Add structured tracing to your agent: a log/JSON per run capturing each step, its
  tokens, and its cost. This is your homemade version of LangSmith/Langfuse — build it
  once before you adopt one.
- Write an eval harness for your Phase 7 agent: a set of tasks with success criteria,
  run them, score pass/fail, track cost-per-task and average steps.

**You've got it when** you instinctively ask "how will I know if this is working?" *before*
building the next feature.

---

## Phase 9 — Advanced depth (pick what's relevant)

By now you can learn these as needed rather than in strict order.

- **Prompting vs. RAG vs. fine-tuning** — when each is the right tool. Most people reach
  for fine-tuning too early; understand why RAG or better prompting usually wins first.
- **Context engineering** — managing what's in the window: summarization, context
  compression, sliding windows, and why "just stuff more in" degrades quality.
- **Caching** — prompt caching (revisit Phase 3) and *semantic* caching (serve a cached
  answer when a new query is semantically close to a past one). Build a small semantic
  cache using your Phase 4 embedding search.
- **Multi-agent systems** — orchestrator + specialized sub-agents; when the coordination
  cost is worth it (often it isn't — measure).
- **Local/open models** — running a small model locally (Ollama/llama.cpp) to feel the
  cost/quality/latency tradeoff from the other end, and for tasks where data can't leave
  your machine.
- **Optional deep cut:** train a tiny transformer (Karpathy's nanoGPT / "Let's build
  GPT"). You don't need this to be a great agentic engineer, but it permanently removes
  the "it's magic" feeling. Worth a weekend once the above is solid.

---

## Anchor resources (lean on these, don't drown in them)

- **Andrej Karpathy** — "Let's build the GPT Tokenizer" + the `minbpe` repo (Phase 1);
  "Deep Dive into LLMs" for end-to-end intuition; nanoGPT / "Let's build GPT" (Phase 9).
- **Anthropic, "Building Effective Agents"** (anthropic.com/research/building-effective-agents)
  and the **claude-cookbooks `patterns/agents`** reference code (Phases 7–8). Read the
  essay *after* Phase 6, not before.
- **Your chosen model provider's API docs** — read the messages, tool-use, streaming, and
  prompt-caching sections directly. Primary sources beat tutorials.
- Pick **one** embeddings provider and **one** chat provider to start. Don't comparison-shop
  models until Phase 3.

## Anti-goals (things to deliberately *not* do yet)
- Don't adopt an agent framework until after Phase 7. The point is to understand what
  they abstract.
- Don't stand up a managed vector DB until you've felt the limits of your `numpy` store.
- Don't fine-tune anything until Phase 9. It's rarely the first right answer.
- Don't optimize cost before you can measure it (Phase 3 gives you the tools).

## How to know you're "as proficient as a senior engineer"
You can take a vague product idea, decide whether it needs an agent or a workflow, choose
models per step with a cost estimate, design the retrieval and evaluation strategy, and
name the three most likely ways it'll fail in production — before writing a line of code.
