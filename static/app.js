const $ = (id) => document.getElementById(id);
const folderInput = $("folder");
const scanBtn = $("scanBtn");
const statusEl = $("status");
const resultsEl = $("results");
let activeType = "";
let pollTimer = null;
let wasRunning = false;

const STATUS_LABEL = { done: "text ready", pending: "waiting", failed: "failed", needs_ocr: "needs OCR", skipped: "skipped" };

async function api(path, options) {
  const res = await fetch(path, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
}

function setStatus(msg, isError = false) {
  statusEl.textContent = msg;
  statusEl.className = "status" + (isError ? " error" : "");
}

function formatSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / 1024 / 1024).toFixed(1) + " MB";
}

function renderFiles(files, total) {
  resultsEl.replaceChildren();
  if (!files.length) {
    resultsEl.textContent = "No files to show yet. Scan a folder first.";
    return;
  }
  for (const f of files) {
    const card = document.createElement("div");
    card.className = "card";
    card.title = f.path;

    const thumb = document.createElement("div");
    thumb.className = "thumb";
    if (f.type === "image") {
      const img = document.createElement("img");
      img.loading = "lazy";
      img.src = "/api/preview?path=" + encodeURIComponent(f.path);
      img.alt = f.name;
      thumb.appendChild(img);
    } else {
      thumb.textContent = f.ext.replace(".", "").toUpperCase();
    }

    const meta = document.createElement("div");
    meta.className = "meta";
    const name = document.createElement("div");
    name.className = "name";
    name.textContent = f.name;
    const sub = document.createElement("div");
    sub.className = "sub";
    sub.append(`${formatSize(f.size)} · ${f.modified.slice(0, 10)}`);
    const badge = document.createElement("span");
    badge.className = "badge " + f.status;
    badge.textContent = STATUS_LABEL[f.status] || f.status;
    sub.appendChild(badge);

    const textBtn = document.createElement("button");
    textBtn.className = "mini";
    textBtn.textContent = "Text";
    textBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      showText(f);
    });

    meta.append(name, sub, textBtn);
    card.append(thumb, meta);
    card.addEventListener("click", () => openFile(f.path));
    resultsEl.appendChild(card);
  }
  if (total > files.length) {
    const more = document.createElement("p");
    more.className = "more";
    more.style.gridColumn = "1 / -1";
    more.textContent = `Showing the newest ${files.length} of ${total} files. Search arrives in the next milestone.`;
    resultsEl.appendChild(more);
  }
}

async function loadFiles() {
  const q = activeType ? "&type=" + activeType : "";
  const data = await api("/api/files?limit=300" + q);
  renderFiles(data.files, data.count);
  return data;
}

async function openFile(path) {
  try {
    await api("/api/open?path=" + encodeURIComponent(path), { method: "POST" });
  } catch (e) {
    setStatus(e.message, true);
  }
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
    if (s.running) {
      pollTimer = setTimeout(refreshIndex, 1500);
    } else if (wasRunning) {
      await loadFiles(); // show final statuses
    }
    wasRunning = s.running;
  } catch (e) {
    $("indexStatus").textContent = e.message;
  }
}

async function indexAction(path) {
  try {
    await api(path, { method: "POST" });
  } catch (e) {
    setStatus(e.message, true);
  }
  wasRunning = true;
  setTimeout(refreshIndex, 300);
}

scanBtn.addEventListener("click", async () => {
  const folder = folderInput.value.trim();
  if (!folder) return setStatus("Enter a folder path.", true);
  scanBtn.disabled = true;
  setStatus("Scanning...");
  try {
    const data = await api("/api/scan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ folder }),
    });
    setStatus(
      `${data.count} files in this folder: ${data.added} new, ${data.updated} changed, ${data.removed} removed, ${data.unchanged} unchanged.` +
        (data.skipped_folders ? ` ${data.skipped_folders} folder(s) could not be read and were left as they were.` : "")
    );
    await loadFiles();
    wasRunning = true;
    refreshIndex();
  } catch (e) {
    setStatus(e.message, true);
  } finally {
    scanBtn.disabled = false;
  }
});

folderInput.addEventListener("keydown", (e) => { if (e.key === "Enter") scanBtn.click(); });
$("startBtn").addEventListener("click", () => indexAction("/api/index/start"));
$("pauseBtn").addEventListener("click", () => indexAction("/api/index/pause"));
$("retryBtn").addEventListener("click", () => indexAction("/api/index/retry"));
$("dlgClose").addEventListener("click", () => $("textDialog").close());

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", async () => {
    document.querySelectorAll(".chip").forEach((c) => c.classList.remove("active"));
    chip.classList.add("active");
    activeType = chip.dataset.type;
    await loadFiles();
  });
});

loadFiles().then((data) => { if (data.root && !folderInput.value) folderInput.value = data.root; });
refreshIndex();
