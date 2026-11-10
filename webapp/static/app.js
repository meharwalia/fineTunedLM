(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  };

  const prompt = $("prompt"), btn = $("compare"), banner = $("banner");
  const cards = { base: $("card-base"), tuned: $("card-tuned") };
  const REWARD_RANGE = 3; // bar covers -3..+3, clipped

  async function loadExamples() {
    try {
      const r = await fetch("/api/examples");
      const data = await r.json();
      const box = $("chips");
      const items = Array.isArray(data.selected) && data.selected.length
        ? data.selected : (data.examples || []).map((q) => ({ question: q }));
      if (Array.isArray(data.selected) && data.selected.length) $("ex-caption").hidden = false;
      items.forEach((it) => {
        const q = it.question;
        const b = el("button", "chip", it.label || q);
        if (it.label) b.title = q;
        b.type = "button";
        b.setAttribute("role", "listitem");
        b.addEventListener("click", () => { prompt.value = q; updateCount(); prompt.focus(); });
        box.appendChild(b);
      });
    } catch (e) { /* examples are optional */ }
  }

  function body(card) { return card.querySelector("[data-body]"); }

  function renderIdle(card) {
    const b = body(card); b.replaceChildren();
    const s = el("div", "state");
    s.appendChild(el("div", "title", "Waiting for a question"));
    s.appendChild(el("div", null, "The answer will appear here."));
    b.appendChild(s);
    card.setAttribute("aria-busy", "false");
    card.classList.remove("win");
  }

  function renderLoading(card) {
    const b = body(card); b.replaceChildren();
    const s = el("div", "state");
    s.appendChild(el("div", "spinner"));
    s.appendChild(el("div", null, "Generating…"));
    [100, 90, 60].forEach((w) => { const k = el("div", "skeleton"); k.style.width = w + "%"; s.appendChild(k); });
    b.appendChild(s);
    card.setAttribute("aria-busy", "true");
    card.classList.remove("win");
  }

  function renderError(card, res) {
    const b = body(card); b.replaceChildren();
    const titles = { not_connected: "Model not connected yet", not_available: "Model not trained yet", failed: "Generation failed" };
    const s = el("div", "state" + (res.error === "failed" ? " err" : ""));
    s.appendChild(el("div", "title", titles[res.error] || "Generation failed"));
    if (res.error === "failed" && res.message) s.appendChild(el("div", null, res.message));
    else if (res.error === "not_available") s.appendChild(el("div", null, "Train the model first, then reload."));
    else if (res.error === "not_connected") s.appendChild(el("div", null, "The model interface has not been connected to the server yet."));
    b.appendChild(s);
    card.setAttribute("aria-busy", "false");
    card.classList.remove("win");
  }

  // Minimal, safe markdown: **bold** and `code` become elements (built with
  // textContent, never innerHTML). Line breaks and "1." lists already read well
  // because .response uses white-space: pre-wrap.
  function renderMarkdown(target, text) {
    if ((text.match(/\*\*/g) || []).length % 2 === 1) { // a "**" left open by the token limit
      const i = text.lastIndexOf("**");
      text = text.slice(0, i) + text.slice(i + 2);
    }
    const re = /\*\*([^*\n]+?)\*\*|`([^`\n]+?)`/g;
    let last = 0, m;
    while ((m = re.exec(text)) !== null) {
      if (m.index > last) target.appendChild(document.createTextNode(text.slice(last, m.index)));
      target.appendChild(m[1] !== undefined ? el("strong", null, m[1]) : el("code", null, m[2]));
      last = re.lastIndex;
    }
    if (last < text.length) target.appendChild(document.createTextNode(text.slice(last)));
  }

  // better: "low" (lower is better) or "high"
  const BADGES = [
    { key: "avg_sentence_words", label: (v) => "avg sentence " + round1(v) + " words", better: "low",
      tip: "Average number of words per sentence. STE asks for short sentences (20 words or fewer for steps)." },
    { key: "pct_passive", label: (v) => "passive " + Math.round(v) + "%", better: "low",
      tip: "Share of sentences in the passive voice. STE prefers the active voice." },
    { key: "list_items", label: (v) => "list items " + Math.round(v), better: "high",
      tip: "Number of numbered steps or bullets. STE uses numbered steps for procedures and bullets for lists." },
  ];
  function round1(v) { return (Math.round(v * 10) / 10).toString(); }
  function badgeValue(ste, key) {
    if (!ste) return null;
    const v = key === "filler" ? ste.filler_phrases : ste[key];
    if (Array.isArray(v)) return v.length;
    return typeof v === "number" ? v : null;
  }

  function renderBadges(card, res, other) {
    const box = card.querySelector("[data-badges]");
    if (!box) return;
    box.replaceChildren();
    if (!res || !res.ste) return;
    BADGES.forEach((d) => {
      const v = badgeValue(res.ste, d.key);
      if (v === null) return;
      const o = other && other.ok ? badgeValue(other.ste, d.key) : null;
      const best = o !== null && o !== v && (d.better === "high" ? v > o : v < o);
      const t = el("span", "badge" + (best ? " badge-best" : ""), d.label(v));
      t.title = d.tip + (best ? " (better than the other answer)" : "");
      t.tabIndex = 0;
      box.appendChild(t);
    });
  }

  function renderOk(card, res, limit) {
    const b = body(card); b.replaceChildren();
    const head = el("div", "name-row");
    head.appendChild(el("div", "model-name", res.model_name));
    b.appendChild(head);
    const resp = el("div", "response");
    if (res.response) renderMarkdown(resp, res.response.trimEnd());
    else resp.textContent = "(empty response)";
    if (limit && res.finished === false) {
      resp.appendChild(document.createTextNode(" …"));
      resp.appendChild(el("div", "cutoff", "Stopped at the " + limit + "-token limit."));
    }
    resp.tabIndex = 0;
    resp.setAttribute("aria-label", "Model response");
    b.appendChild(resp);

    const bd = el("div", "badges");
    bd.dataset.badges = "1";
    bd.setAttribute("aria-label", "Simplified Technical English checks");
    b.appendChild(bd);
    renderBadges(card, res, null);

    card.setAttribute("aria-busy", "false");
  }

  function renderResult(card, res, limit) { res.ok ? renderOk(card, res, limit) : renderError(card, res); }

  function highlight(base, tuned) {
    if (base.ok && tuned.ok) { renderBadges(cards.base, base, tuned); renderBadges(cards.tuned, tuned, base); }
    cards.base.classList.remove("win"); cards.tuned.classList.remove("win");
    if (base.ok && tuned.ok && typeof base.reward === "number" && typeof tuned.reward === "number"
        && base.reward !== tuned.reward) {
      (tuned.reward > base.reward ? cards.tuned : cards.base).classList.add("win");
    }
  }

  function showBanner(msg) { banner.textContent = msg; banner.hidden = false; }

  let lastResult = null;

  async function compare() {
    const text = prompt.value.trim();
    if (!text) { showBanner("Please type a question first."); prompt.focus(); return; }
    banner.hidden = true;
    btn.disabled = true; btn.textContent = "Comparing…";
    renderLoading(cards.base); renderLoading(cards.tuned);
    const payload = { prompt: text };
    try {
      const r = await fetch("/api/compare", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
      });
      if (!r.ok) {
        let msg = "Request failed (" + r.status + ")";
        if (r.status === 422) msg = "Invalid input: check the question length (1–500 characters).";
        throw new Error(msg);
      }
      const data = await r.json();
      lastResult = data;
      const limit = data.settings && data.settings.max_new_tokens;
      renderResult(cards.base, data.base, limit);
      renderResult(cards.tuned, data.tuned, limit);
      highlight(data.base, data.tuned);
    } catch (e) {
      showBanner(e.message && e.message !== "Failed to fetch" ? e.message : "Could not reach the server.");
      if (lastResult) {
        const limit = lastResult.settings && lastResult.settings.max_new_tokens;
        renderResult(cards.base, lastResult.base, limit);
        renderResult(cards.tuned, lastResult.tuned, limit);
        highlight(lastResult.base, lastResult.tuned);
      } else { renderIdle(cards.base); renderIdle(cards.tuned); }
    } finally {
      btn.disabled = false; btn.textContent = "Compare";
    }
  }

  function updateCount() { $("count").textContent = prompt.value.length + " / 500"; }
  prompt.addEventListener("input", updateCount);
  prompt.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); if (!btn.disabled) compare(); }
  });
  $("form").addEventListener("submit", (e) => { e.preventDefault(); if (!btn.disabled) compare(); });

  renderIdle(cards.base); renderIdle(cards.tuned);
  loadExamples();
})();
