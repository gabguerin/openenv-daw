// DAW Studio in the browser: Pyodide runs the real openenv_daw simulator, tasks and rubric
// (server/ui/static_bridge.py); this file renders its HTML views and makes the model call.
import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.mjs";

const $ = (id) => document.getElementById(id);
const statusEl = $("status");
const OAUTH_KEY = "openenv-daw-oauth";

let studio = null;
let meta = null;
let tools = [];
let currentTool = null;
let state = { started: false, done: false, step_count: 0, max_steps: 30 };
let stopRequested = false;
let oauth = null;

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

// ---- Hugging Face OAuth (Static Spaces with hf_oauth: true) ----------------------------------
async function setupOAuth() {
  const hasOAuth = Boolean(window.huggingface?.variables?.OAUTH_CLIENT_ID);
  $("mode").querySelector('[value="hf"]').disabled = !hasOAuth;
  if (!hasOAuth) {
    $("mode").value = "custom";
    return;
  }
  const hub = await import("https://esm.sh/@huggingface/hub@2");
  oauth = (await hub.oauthHandleRedirectIfPresent()) ||
    JSON.parse(localStorage.getItem(OAUTH_KEY) || "null");
  if (oauth) localStorage.setItem(OAUTH_KEY, JSON.stringify(oauth));
  const box = $("hf-auth");
  box.innerHTML = "";
  const btn = document.createElement("button");
  if (oauth) {
    btn.textContent = `Signed in as ${oauth.userInfo?.preferred_username ?? "you"} (sign out)`;
    btn.onclick = () => {
      localStorage.removeItem(OAUTH_KEY);
      location.reload();
    };
  } else {
    btn.textContent = "Sign in with Hugging Face";
    btn.onclick = async () => {
      window.location.href = await hub.oauthLoginUrl();
    };
  }
  box.appendChild(btn);
}

// ---- model call ------------------------------------------------------------------------------
function modelConfig() {
  const maxTokens = Math.max(parseInt($("max-tokens").value, 10) || meta.max_tokens, 1);
  const model = $("model").value.trim();
  if ($("mode").value === "hf") {
    if (!oauth) return { error: "Sign in with Hugging Face first." };
    return {
      url: meta.hf_router_url,
      key: oauth.accessToken,
      model: model || meta.space_model,
      maxTokens,
    };
  }
  return {
    url: $("base-url").value.trim() || meta.local_url,
    key: $("api-key").value.trim(),
    model: model || meta.local_model,
    maxTokens,
  };
}

async function modelStep() {
  const cfg = modelConfig();
  if (cfg.error) {
    showMessage($("model-out"), cfg.error, false);
    return false;
  }
  const built = call("build_chat_payload", cfg.model, cfg.maxTokens);
  if (built.error) {
    showMessage($("model-out"), built.error, false);
    return false;
  }
  const headers = { "Content-Type": "application/json" };
  if (cfg.key) headers.Authorization = `Bearer ${cfg.key}`;
  let status = 0;
  let text = "";
  try {
    const resp = await fetch(cfg.url.replace(/\/$/, "") + "/chat/completions", {
      method: "POST",
      headers,
      body: JSON.stringify(built.payload),
    });
    status = resp.status;
    text = await resp.text();
  } catch (e) {
    text = String(e);
  }
  const out = call("apply_chat_response", text, status, cfg.maxTokens);
  applyViews(out);
  showMessage($("model-out"), out.message, out.ok);
  return out.ok;
}

async function runToEnd() {
  stopRequested = false;
  $("model-stop").hidden = false;
  setBusy(true);
  try {
    for (let i = 0; i < state.max_steps && !state.done && !stopRequested; i++) {
      if (!(await modelStep())) break;
    }
  } finally {
    $("model-stop").hidden = true;
    setBusy(false);
  }
}

function setBusy(busy) {
  for (const id of ["model-step", "model-run", "run", "oracle", "reset", "random"]) {
    $(id).disabled = busy;
  }
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
  showMessage($("model-out"), "");
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

  $("base-url").value = meta.local_url;
  $("model").value = meta.space_model;
  $("max-tokens").value = meta.max_tokens;
  $("mode").onchange = () => {
    const hf = $("mode").value === "hf";
    $("custom-fields").hidden = hf;
    $("hf-auth").hidden = !hf;
    $("model").value = hf ? meta.space_model : meta.local_model;
  };
  await setupOAuth();
  $("mode").onchange();

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
  $("model-step").onclick = async () => {
    setBusy(true);
    try {
      await modelStep();
    } finally {
      setBusy(false);
    }
  };
  $("model-run").onclick = runToEnd;
  $("model-stop").onclick = () => {
    stopRequested = true;
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
