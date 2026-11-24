// DAW Studio in the browser: Pyodide runs the real openenv_daw simulator, tasks and rubric
// (server/ui/static_bridge.py); this file renders its HTML views and makes the model call.
import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.mjs";

const $ = (id) => document.getElementById(id);
const statusEl = $("status");

let studio = null;
let meta = null;
let tools = [];
let currentTool = null;
let state = { started: false, done: false, step_count: 0, max_steps: 30 };

// ---- helpers -------------------------------------------------------------------------------
const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

/** Tiny markdown: ``` fences, `inline code`, paragraphs. */
function md(text) {
  const parts = text.split(/```(?:\w+)?\n([\s\S]*?)```/g);
  return parts
    .map((chunk, i) =>
      i % 2
        ? `<pre>${esc(chunk)}</pre>`
        : esc(chunk).replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\n/g, "<br>"),
    )
    .join("");
}

function showMessage(el, text, ok = true) {
  el.className = ok ? "msg" : "msg bad";
  el.innerHTML = text ? md(text) : "";
}

function applyViews(out) {
  const v = out.views;
  $("task").innerHTML = v.task;
  $("arrangement").innerHTML = v.arrangement;
  $("reward").innerHTML = v.reward;
  $("checks").innerHTML = v.checks;
  $("log").innerHTML = v.log;
  state = out.state;
  if (out.seed !== undefined) $("seed").value = out.seed;
}

function call(fn, ...args) {
  return JSON.parse(studio[fn](...args));
}

// ---- tools -----------------------------------------------------------------------------------
function pickTool(name) {
  currentTool = tools.find((t) => t.name === name);
  $("args").value = currentTool.skeleton;
  $("tool-doc").textContent = currentTool.description;
  for (const b of $("tools").children) b.classList.toggle("on", b.dataset.name === name);
}

// ---- wiring ----------------------------------------------------------------------------------
function resetEpisode(randomize) {
  const out = call(
    "reset",
    $("split").value,
    $("family").value,
    parseInt($("seed").value, 10) || 0,
    randomize,
  );
  applyViews(out);
  showMessage($("result"), "");
}

async function main() {
  const pyodide = await loadPyodide();
  await pyodide.loadPackage(["pydantic", "pyyaml"]);
  const [bundle, toolsJson] = await Promise.all([
    fetch("bundle.zip").then((r) => r.arrayBuffer()),
    fetch("tools.json").then((r) => r.text()),
  ]);
  pyodide.unpackArchive(bundle, "zip", { extractDir: "/bundle" });
  pyodide.globals.set("TOOLS_JSON", toolsJson);
  studio = pyodide.runPython(`
import json, sys
sys.path.insert(0, "/bundle")
from openenv_daw.server.ui.static_bridge import BrowserStudio
BrowserStudio(json.loads(TOOLS_JSON))
`);
  meta = JSON.parse(studio.meta());
  document.head.insertAdjacentHTML("beforeend", meta.css);

  for (const f of meta.families) $("family").add(new Option(f, f));
  tools = meta.tools;
  for (const t of tools) {
    const b = document.createElement("button");
    b.textContent = t.name;
    b.dataset.name = t.name;
    b.onclick = () => pickTool(t.name);
    $("tools").appendChild(b);
  }
  pickTool(tools[0].name);

  $("reset").onclick = () => resetEpisode(false);
  $("random").onclick = () => resetEpisode(true);
  $("family").onchange = () => resetEpisode(true);
  $("run").onclick = () => {
    const out = call("manual", currentTool.name, $("args").value);
    applyViews(out);
    showMessage($("result"), out.message, out.ok);
  };
  $("oracle").onclick = () => {
    const out = call("oracle_step");
    applyViews(out);
    showMessage($("result"), out.message, out.ok);
  };
  $("log").addEventListener("click", (e) => {
    const li = e.target.closest("[data-step]");
    if (li) applyViews(call("select_step", parseInt(li.dataset.step, 10)));
  });

  applyViews(call("init"));
  statusEl.remove();
}

main().catch((e) => {
  statusEl.textContent = `Failed to start: ${e}`;
  console.error(e);
});
