const folderInput = document.getElementById("folder");
const scanBtn = document.getElementById("scanBtn");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
let activeType = "";

function setStatus(msg, isError = false) {
  statusEl.textContent = msg;
  statusEl.className = "status" + (isError ? " error" : "");
}

function formatSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / 1024 / 1024).toFixed(1) + " MB";
}

function renderFiles(files) {
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
    sub.textContent = `${formatSize(f.size)} · ${f.modified.slice(0, 10)}`;
    meta.append(name, sub);

    card.append(thumb, meta);
    card.addEventListener("click", () => openFile(f.path));
    resultsEl.appendChild(card);
  }
}

async function loadFiles() {
  const q = activeType ? "?type=" + activeType : "";
  const res = await fetch("/api/files" + q);
  const data = await res.json();
  renderFiles(data.files);
  return data;
}

async function openFile(path) {
  const res = await fetch("/api/open?path=" + encodeURIComponent(path), { method: "POST" });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    setStatus(err.detail || "Could not open file", true);
  }
}

scanBtn.addEventListener("click", async () => {
  const folder = folderInput.value.trim();
  if (!folder) return setStatus("Enter a folder path.", true);
  scanBtn.disabled = true;
  setStatus("Scanning...");
  try {
    const res = await fetch("/api/scan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ folder }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Scan failed");
    setStatus(`Found ${data.count} supported files.`);
    await loadFiles();
  } catch (e) {
    setStatus(e.message, true);
  } finally {
    scanBtn.disabled = false;
  }
});

folderInput.addEventListener("keydown", (e) => { if (e.key === "Enter") scanBtn.click(); });

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", async () => {
    document.querySelectorAll(".chip").forEach((c) => c.classList.remove("active"));
    chip.classList.add("active");
    activeType = chip.dataset.type;
    await loadFiles();
  });
});

loadFiles();
