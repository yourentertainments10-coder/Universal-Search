const $ = (id) => document.getElementById(id);
const PAGE = 20;
const MARK_START = "\x02";
const MARK_END = "\x03";
const STATUS_LABEL = { done: "text ready", pending: "waiting", failed: "failed", needs_ocr: "needs OCR", skipped: "skipped" };

let activeType = "";
let searchSeq = 0;      // ignore answers to outdated searches
let shown = 0;
let debounce = null;
let pollTimer = null;
let wasRunning = false;

async function api(path, options) {
  const res = await fetch(path, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
}

function formatSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / 1024 / 1024).toFixed(1) + " MB";
}

// Put text into an element, turning the server's match markers into <mark>. No innerHTML.
function setMarked(el, text) {
  el.replaceChildren();
  let inMark = false;
  let buf = "";
  const flush = () => {
    if (!buf) return;
    if (inMark) {
      const m = document.createElement("mark");
      m.textContent = buf;
      el.appendChild(m);
    } else {
      el.appendChild(document.createTextNode(buf));
    }
    buf = "";
  };
  for (const ch of text) {
    if (ch === MARK_START) { flush(); inMark = true; }
    else if (ch === MARK_END) { flush(); inMark = false; }
    else buf += ch;
  }
  flush();
}

function typeLabel(f) {
  return { image: "Image", pdf: "PDF", document: "Document" }[f.type] || f.type;
}

async function act(path, label) {
  try {
    await api(path, { method: "POST" });
  } catch (e) {
    $("resultInfo").textContent = `${label}: ${e.message}`;
    $("resultInfo").className = "status error";
  }
}

function buildResult(f) {
  const item = document.createElement("article");
  item.className = "result";

  const thumb = document.createElement("div");
  thumb.className = "thumb";
  const fallback = () => { thumb.textContent = f.ext.replace(".", "").toUpperCase(); };
  if (f.type === "image") {
    const img = document.createElement("img");
    img.loading = "lazy";
    img.alt = "";
    img.src = "/api/thumb?path=" + encodeURIComponent(f.path);
    img.addEventListener("error", () => { img.remove(); fallback(); });
    thumb.appendChild(img);
  } else {
    fallback();
  }

  const body = document.createElement("div");
  body.className = "body";
  const title = document.createElement("div");
  title.className = "title";
  title.title = "Open original";
  setMarked(title, f.name_hl);
  title.addEventListener("click", () => act("/api/open?path=" + encodeURIComponent(f.path), "Open"));
  const badge = document.createElement("span");
  badge.className = "badge " + f.status;
  badge.textContent = STATUS_LABEL[f.status] || f.status;
  title.appendChild(badge);

  const path = document.createElement("div");
  path.className = "path";
  path.textContent = f.path;
  path.title = f.path;
  const sub = document.createElement("div");
  sub.className = "sub";
  sub.textContent = `${typeLabel(f)} · ${formatSize(f.size)} · ${f.modified.slice(0, 10)}`;
  body.append(title, path, sub);
  if (f.snippet) {
    const snip = document.createElement("div");
    snip.className = "snippet";
    setMarked(snip, f.snippet);
    body.appendChild(snip);
  }

  const actions = document.createElement("div");
  actions.className = "actions";
  const mk = (label, fn) => {
    const b = document.createElement("button");
    b.className = "secondary";
    b.textContent = label;
    b.addEventListener("click", fn);
    return b;
  };
  actions.append(
    mk("Open", () => act("/api/open?path=" + encodeURIComponent(f.path), "Open")),
    mk("Show in folder", () => act("/api/reveal?path=" + encodeURIComponent(f.path), "Show in folder")),
    mk("Text", () => showText(f))
  );

  item.append(thumb, body, actions);
  return item;
}

function searchUrl(offset) {
  const p = new URLSearchParams({ q: $("q").value, sort: $("sort").value, limit: PAGE, offset });
  if (activeType) p.set("type", activeType);
  if ($("dateFrom").value) p.set("date_from", $("dateFrom").value);
  if ($("dateTo").value) p.set("date_to", $("dateTo").value);
  return "/api/search?" + p.toString();
}

async function runSearch(append = false) {
  const seq = ++searchSeq;
  const info = $("resultInfo");
  try {
    const data = await api(searchUrl(append ? shown : 0));
    if (seq !== searchSeq) return data; // a newer search already started
    const list = $("results");
    if (!append) { list.replaceChildren(); shown = 0; }
    for (const f of data.results) list.appendChild(buildResult(f));
    shown += data.results.length;
    $("moreBtn").hidden = shown >= data.total;

    const hasQuery = $("q").value.trim() !== "";
    let msg;
    if (data.total === 0) {
      msg = hasQuery ? "No results." : "Nothing indexed yet. Open Folders and indexing below and scan a folder.";
    } else if (hasQuery) {
      msg = `${data.total} result${data.total === 1 ? "" : "s"}`;
    } else {
      msg = `${data.total} files, newest first. Type above to search.`;
    }
    if (data.pending > 0 && data.total >= 0 && (hasQuery || data.total === 0)) {
      msg += ` ${data.pending} file${data.pending === 1 ? " is" : "s are"} still being indexed, so more results may appear.`;
    }
    info.textContent = msg;
    info.className = "status";
    return data;
  } catch (e) {
    if (seq === searchSeq) { info.textContent = e.message; info.className = "status error"; }
  }
}

function scheduleSearch() {
  clearTimeout(debounce);
  debounce = setTimeout(() => runSearch(false), 220);
}

async function showText(f) {
  $("dlgTitle").textContent = f.name;
  $("dlgMeta").textContent = "Loading...";
  $("dlgText").textContent = "";
  $("textDialog").showModal();
  try {
    const data = await api("/api/content?path=" + encodeURIComponent(f.path));
    const c = data.content;
    if (!c) {
      $("dlgMeta").textContent = `Status: ${STATUS_LABEL[data.status] || data.status}. No text extracted yet.`;
      return;
    }
    const bits = [`Method: ${c.method || "none"}`];
    if (c.engine) bits.push(`engine: ${c.engine}`);
    if (c.pages) bits.push(`${c.pages} page(s)`);
    bits.push(`${c.length.toLocaleString()} characters`);
    if (c.note) bits.push(c.note);
    if (c.error) bits.push("Error: " + c.error);
    $("dlgMeta").textContent = bits.join(" · ");
    $("dlgText").textContent = c.text || "(no text found)";
  } catch (e) {
    $("dlgMeta").textContent = e.message;
  }
}

/* ---------- scanning and indexing ---------- */

function setScanStatus(msg, isError = false) {
  $("status").textContent = msg;
  $("status").className = "status" + (isError ? " error" : "");
}

function renderIndex(s) {
  const c = s.counts;
  const processed = c.done + c.failed + c.needs_ocr + c.skipped;
  $("barFill").style.width = (s.total ? (processed / s.total) * 100 : 0) + "%";
  let text = `${processed} of ${s.total} files processed: ${c.done} with text, ${c.failed} failed, ${c.needs_ocr} need OCR, ${c.skipped} skipped, ${c.pending} waiting.`;
  if (s.running && s.pausing) text += " Pausing after the current file...";
  else if (s.running && s.current) text += ` Working on ${s.current}`;
  else if (!s.running && c.pending > 0) text += " Paused.";
  if (s.last_error) text += ` Stopped by an error: ${s.last_error}`;
  $("indexStatus").textContent = text;
  $("indexStatus").className = "status" + (s.last_error ? " error" : "");
  $("ocrStatus").textContent = s.ocr.available ? `OCR engine: ${s.ocr.engine}` : s.ocr.hint;
  $("startBtn").disabled = s.running || c.pending === 0;
  $("pauseBtn").disabled = !s.running || s.pausing;
  $("retryBtn").disabled = s.running || c.failed + c.needs_ocr === 0;
}

async function refreshIndex() {
  clearTimeout(pollTimer);
  try {
    const s = await api("/api/index/status");
    renderIndex(s);
    if (s.running) pollTimer = setTimeout(refreshIndex, 1500);
    else if (wasRunning) await runSearch(false); // indexing finished: show final results
    wasRunning = s.running;
  } catch (e) {
    $("indexStatus").textContent = e.message;
  }
}

async function indexAction(path) {
  try { await api(path, { method: "POST" }); } catch (e) { setScanStatus(e.message, true); }
  wasRunning = true;
  setTimeout(refreshIndex, 300);
}

$("scanBtn").addEventListener("click", async () => {
  const folder = $("folder").value.trim();
  if (!folder) return setScanStatus("Enter a folder path.", true);
  $("scanBtn").disabled = true;
  setScanStatus("Scanning...");
  try {
    const d = await api("/api/scan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ folder }),
    });
    setScanStatus(
      `${d.count} files in this folder: ${d.added} new, ${d.updated} changed, ${d.removed} removed, ${d.unchanged} unchanged.` +
        (d.skipped_folders ? ` ${d.skipped_folders} folder(s) could not be read and were left as they were.` : "")
    );
    await runSearch(false);
    wasRunning = true;
    refreshIndex();
  } catch (e) {
    setScanStatus(e.message, true);
  } finally {
    $("scanBtn").disabled = false;
  }
});

/* ---------- wiring ---------- */

$("folder").addEventListener("keydown", (e) => { if (e.key === "Enter") $("scanBtn").click(); });
$("startBtn").addEventListener("click", () => indexAction("/api/index/start"));
$("pauseBtn").addEventListener("click", () => indexAction("/api/index/pause"));
$("retryBtn").addEventListener("click", () => indexAction("/api/index/retry"));
$("dlgClose").addEventListener("click", () => $("textDialog").close());
$("moreBtn").addEventListener("click", () => runSearch(true));
$("q").addEventListener("input", scheduleSearch);
$("q").addEventListener("keydown", (e) => { if (e.key === "Enter") { clearTimeout(debounce); runSearch(false); } });
for (const id of ["dateFrom", "dateTo", "sort"]) $(id).addEventListener("change", () => runSearch(false));
$("clearBtn").addEventListener("click", () => {
  $("q").value = ""; $("dateFrom").value = ""; $("dateTo").value = ""; $("sort").value = "relevance";
  document.querySelectorAll(".chip").forEach((c) => c.classList.toggle("active", c.dataset.type === ""));
  activeType = "";
  $("q").focus();
  runSearch(false);
});
document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    document.querySelectorAll(".chip").forEach((c) => c.classList.remove("active"));
    chip.classList.add("active");
    activeType = chip.dataset.type;
    runSearch(false);
  });
});

(async () => {
  try {
    const roots = await api("/api/files?limit=1");
    if (roots.root && !$("folder").value) $("folder").value = roots.root;
  } catch (e) { /* the search below shows any error */ }
  const data = await runSearch(false);
  if (data && data.total > 0) $("manage").open = false; // already indexed: keep the search front and centre
  refreshIndex();
})();
