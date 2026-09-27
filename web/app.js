/* =====================================================
   INDRA Solver Dashboard — JavaScript
   ===================================================== */

const API_BASE = "http://localhost:5050";

// ============================================================
// UTILITY
// ============================================================

function scrollToSection(id) {
  document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
}

function showLoading(text = "Solving…", sub = "") {
  document.getElementById("loading-overlay").style.display = "flex";
  document.getElementById("loading-text").textContent = text;
  document.getElementById("loading-sub").textContent = sub;
}

function hideLoading() {
  document.getElementById("loading-overlay").style.display = "none";
}

async function apiGet(path) {
  const r = await fetch(API_BASE + path);
  return r.json();
}

async function apiPost(path, body) {
  const r = await fetch(API_BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return r.json();
}

function fmtNum(v, digits = 4) {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return v.toPrecision(digits);
  return String(v);
}

function fmtTime(s) {
  if (s < 0.001) return `${(s * 1000).toFixed(2)} ms`;
  if (s < 1) return `${(s * 1000).toFixed(1)} ms`;
  return `${s.toFixed(3)} s`;
}

// ============================================================
// SERVER STATUS
// ============================================================

async function checkServerStatus() {
  try {
    const data = await apiGet("/api/status");
    const dot = document.getElementById("server-dot");
    const txt = document.getElementById("server-status-text");
    const gpuBadge = document.getElementById("gpu-badge");
    const gpuText = document.getElementById("gpu-text");

    dot.classList.add("online");
    txt.textContent = "INDRA Online";

    if (data.gpu_available) {
      gpuBadge.classList.add("active");
      gpuText.textContent = "GPU Active";
    } else {
      gpuText.textContent = "CPU Mode";
    }
  } catch (e) {
    const dot = document.getElementById("server-dot");
    const txt = document.getElementById("server-status-text");
    dot.classList.add("offline");
    txt.textContent = "Server Offline";
  }
}

// ============================================================
// HERO CANVAS ANIMATION
// ============================================================

function initHeroCanvas() {
  const canvas = document.getElementById("hero-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;

  // Simplex polytope visualization
  const nodes = [];
  const edges = [];

  function genGraph() {
    nodes.length = 0; edges.length = 0;

    // Create polytope nodes (vertices of feasible region)
    const cx = W / 2, cy = H / 2;
    const R = 160;
    for (let i = 0; i < 8; i++) {
      const a = (i / 8) * Math.PI * 2;
      nodes.push({
        x: cx + R * Math.cos(a) + (Math.random() - 0.5) * 40,
        y: cy + R * Math.sin(a) + (Math.random() - 0.5) * 40,
        r: Math.random() * 4 + 3,
        active: Math.random() > 0.6,
        vx: (Math.random() - 0.5) * 0.3,
        vy: (Math.random() - 0.5) * 0.3,
      });
    }

    // Add inner nodes
    for (let i = 0; i < 6; i++) {
      const a = Math.random() * Math.PI * 2;
      const r = R * 0.5 * Math.random();
      nodes.push({
        x: cx + r * Math.cos(a), y: cy + r * Math.sin(a),
        r: Math.random() * 3 + 2,
        active: Math.random() > 0.5,
        vx: (Math.random() - 0.5) * 0.2,
        vy: (Math.random() - 0.5) * 0.2,
      });
    }

    // Edges
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dx = nodes[i].x - nodes[j].x;
        const dy = nodes[i].y - nodes[j].y;
        const d = Math.sqrt(dx * dx + dy * dy);
        if (d < 200 && Math.random() > 0.5) {
          edges.push({ i, j, alpha: 0.08 + Math.random() * 0.15 });
        }
      }
    }
  }

  genGraph();

  // Optimal path
  let pathNodes = [];
  let pathT = 0;
  let pathDir = 1;

  function buildPath() {
    const start = nodes[Math.floor(Math.random() * nodes.length)];
    pathNodes = [start];
    let current = start;
    for (let k = 0; k < 5; k++) {
      const neighbors = edges
        .filter(e => nodes[e.i] === current || nodes[e.j] === current)
        .map(e => nodes[e.i] === current ? nodes[e.j] : nodes[e.i]);
      if (neighbors.length === 0) break;
      current = neighbors[Math.floor(Math.random() * neighbors.length)];
      pathNodes.push(current);
    }
  }
  buildPath();

  let t = 0;
  function draw() {
    ctx.clearRect(0, 0, W, H);

    // Draw edges
    edges.forEach(e => {
      const ni = nodes[e.i], nj = nodes[e.j];
      ctx.beginPath();
      ctx.moveTo(ni.x, ni.y);
      ctx.lineTo(nj.x, nj.y);
      ctx.strokeStyle = `rgba(255,107,53,${e.alpha})`;
      ctx.lineWidth = 0.8;
      ctx.stroke();
    });

    // Draw optimal path (animated)
    if (pathNodes.length > 1) {
      ctx.beginPath();
      ctx.moveTo(pathNodes[0].x, pathNodes[0].y);
      for (let k = 1; k < pathNodes.length; k++) {
        ctx.lineTo(pathNodes[k].x, pathNodes[k].y);
      }
      ctx.strokeStyle = "rgba(255,215,0,0.6)";
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.stroke();
      ctx.setLineDash([]);

      // Moving particle along path
      const totalLen = pathNodes.length - 1;
      const seg = Math.floor(pathT * totalLen);
      const frac = pathT * totalLen - seg;
      if (seg < pathNodes.length - 1) {
        const px = pathNodes[seg].x + frac * (pathNodes[seg + 1].x - pathNodes[seg].x);
        const py = pathNodes[seg].y + frac * (pathNodes[seg + 1].y - pathNodes[seg].y);
        ctx.beginPath();
        ctx.arc(px, py, 6, 0, Math.PI * 2);
        ctx.fillStyle = "#FFD700";
        ctx.fill();
        ctx.beginPath();
        ctx.arc(px, py, 12, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(255,215,0,0.2)";
        ctx.fill();
      }
    }

    pathT += 0.004;
    if (pathT > 1) { pathT = 0; buildPath(); }

    // Draw nodes
    nodes.forEach((n, idx) => {
      // Update position (gentle drift)
      n.x += n.vx;
      n.y += n.vy;
      if (n.x < 20 || n.x > W - 20) n.vx *= -1;
      if (n.y < 20 || n.y > H - 20) n.vy *= -1;

      const grd = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, n.r * 3);
      if (n.active) {
        grd.addColorStop(0, "rgba(255,107,53,0.9)");
        grd.addColorStop(1, "rgba(255,107,53,0)");
      } else {
        grd.addColorStop(0, "rgba(124,58,237,0.7)");
        grd.addColorStop(1, "rgba(124,58,237,0)");
      }

      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r * 3, 0, Math.PI * 2);
      ctx.fillStyle = grd;
      ctx.fill();

      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
      ctx.fillStyle = n.active ? "#FF6B35" : "#7C3AED";
      ctx.fill();
    });

    // Draw branch-and-bound tree schematic (bottom left)
    drawBBTree(ctx, 40, 340, t);

    // Labels
    ctx.font = "600 11px 'Outfit', sans-serif";
    ctx.fillStyle = "rgba(255,107,53,0.6)";
    ctx.fillText("FEASIBLE REGION", W/2 - 55, 24);
    ctx.fillStyle = "rgba(255,215,0,0.8)";
    ctx.fillText("▶ OPTIMAL PATH", 80, 368);

    t += 0.02;
    requestAnimationFrame(draw);
  }
  draw();
}

function drawBBTree(ctx, x, y, t) {
  const nodes = [
    { x: 0, y: 0, label: "Root" },
    { x: -60, y: 50, label: "L" },
    { x: 60, y: 50, label: "R" },
    { x: -90, y: 100, label: "LL" },
    { x: -30, y: 100, label: "LR" },
    { x: 30, y: 100, label: "RL" },
    { x: -120, y: 150, label: "★" },
  ];
  const edges = [[0,1],[0,2],[1,3],[1,4],[2,5],[3,6]];
  const colors = ["#FF6B35","#7C3AED","#7C3AED","#06B6D4","#10B981","#7C3AED","#FFD700"];

  edges.forEach(([a, b]) => {
    ctx.beginPath();
    ctx.moveTo(x + nodes[a].x, y + nodes[a].y);
    ctx.lineTo(x + nodes[b].x, y + nodes[b].y);
    ctx.strokeStyle = "rgba(255,255,255,0.15)";
    ctx.lineWidth = 1;
    ctx.stroke();
  });

  nodes.forEach((n, i) => {
    const pulse = i === 6 ? 0.5 + 0.5 * Math.sin(t * 3) : 1;
    ctx.beginPath();
    ctx.arc(x + n.x, y + n.y, 12 * pulse, 0, Math.PI * 2);
    ctx.fillStyle = colors[i] + "33";
    ctx.fill();
    ctx.beginPath();
    ctx.arc(x + n.x, y + n.y, 6, 0, Math.PI * 2);
    ctx.fillStyle = colors[i];
    ctx.fill();
    ctx.font = "bold 7px 'JetBrains Mono'";
    ctx.fillStyle = "rgba(255,255,255,0.7)";
    ctx.textAlign = "center";
    ctx.fillText(n.label, x + n.x, y + n.y + 2);
    ctx.textAlign = "left";
  });
}

// ============================================================
// PROBLEM EXAMPLES
// ============================================================

const EXAMPLES = {
  simple_lp: {
    name: "Simple_LP_2D",
    type: "LP",
    c: "-1, -2",
    aub: "1, 1 | 4\n1, 0 | 3\n0, 1 | 3",
    aeq: "",
    lb: "0, 0",
    ub: "",
  },
  diet_lp: {
    name: "Diet_Problem",
    type: "LP",
    c: "3.19, 2.59, 2.19, 2.89, 1.89",
    aub: "-60,-8,-8,-40,-15 | -55\n-20,-0,-10,-40,-35 | -33\n-10,-15,-15,-35,-70 | -70",
    aeq: "",
    lb: "0,0,0,0,0",
    ub: "",
  },
  knapsack: {
    name: "01_Knapsack",
    type: "MILP",
    c: "-10,-6,-4,-12,-5,-8,-3,-9",
    aub: "6,4,3,8,4,5,2,6 | 20",
    aeq: "",
    lb: "0,0,0,0,0,0,0,0",
    ub: "1,1,1,1,1,1,1,1",
    intvars: "0,1,2,3,4,5,6,7",
  },
  portfolio_qp: {
    name: "Portfolio_QP",
    type: "QP",
    c: "-0.12,-0.10,-0.08,-0.15,-0.09",
    q: "2,0.5,0.2,0.3,0.1;0.5,1.5,0.3,0.2,0.15;0.2,0.3,1.2,0.4,0.25;0.3,0.2,0.4,1.8,0.2;0.1,0.15,0.25,0.2,1.1",
    aub: "",
    aeq: "1,1,1,1,1 | 1",
    lb: "0,0,0,0,0",
    ub: "0.5,0.5,0.5,0.5,0.5",
  },
  transport_lp: {
    name: "Transport_LP",
    type: "LP",
    c: "2,3,1,5,4,8,5,6,8,8",
    aub: "",
    aeq: "1,1,0,0,0,0,0,0,0,0 | 120\n0,0,1,1,1,0,0,0,0,0 | 80\n0,0,0,0,0,1,1,0,0,0 | 80\n0,0,0,0,0,0,0,1,1,1 | 100\n1,0,1,0,0,1,0,1,0,0 | 150\n0,1,0,1,0,0,1,0,1,0 | 130",
    lb: "0,0,0,0,0,0,0,0,0,0",
    ub: "",
  },
};

function loadExample(key) {
  if (!key) return;
  const ex = EXAMPLES[key];
  if (!ex) return;

  document.getElementById("prob-name").value = ex.name;
  document.getElementById("prob-type").value = ex.type;
  document.getElementById("vec-c").value = ex.c;
  document.getElementById("mat-aub").value = ex.aub || "";
  document.getElementById("mat-aeq").value = ex.aeq || "";
  document.getElementById("vec-lb").value = ex.lb || "";
  document.getElementById("vec-ub").value = ex.ub || "";
  if (ex.q) document.getElementById("mat-q").value = ex.q;
  if (ex.intvars) document.getElementById("vec-intvars").value = ex.intvars;

  onTypeChange();
  document.getElementById("example-select").value = "";
}

function onTypeChange() {
  const type = document.getElementById("prob-type").value;
  document.getElementById("q-group").style.display = type === "QP" ? "block" : "none";
  document.getElementById("intvar-group").style.display = type === "MILP" ? "block" : "none";

  const algoSelect = document.getElementById("algo-select");
  if (type === "MILP") algoSelect.value = "bnb";
  else if (type === "QP") algoSelect.value = "qp_admm";
  else algoSelect.value = "auto";
}

// ============================================================
// PLAYGROUND SOLVE
// ============================================================

function parseMatrixInput(text) {
  if (!text.trim()) return { A: null, b: null };
  const rows = text.trim().split("\n").filter(r => r.trim());
  const A = [], b = [];
  for (const row of rows) {
    const [coeffs, rhs] = row.split("|");
    if (!rhs) continue;
    A.push(coeffs.trim().split(",").map(Number));
    b.push(Number(rhs.trim()));
  }
  return { A: A.length ? A : null, b: b.length ? b : null };
}

function parseVec(text, defaultInf = false) {
  if (!text.trim()) return defaultInf ? null : null;
  return text.split(",").map(v => {
    v = v.trim();
    if (v === "inf" || v === "Inf" || v === "INF") return 1e30;
    return Number(v);
  });
}

function parseQ(text) {
  if (!text.trim()) return null;
  return text.trim().split(";").map(row => row.split(",").map(Number));
}

async function solveProblem() {
  const btn = document.getElementById("solve-btn");
  btn.disabled = true;
  btn.innerHTML = `<div class="spinner" style="width:20px;height:20px;margin:0;border-width:2px;"></div> Solving…`;

  const logBox = document.getElementById("solver-log");
  logBox.innerHTML = '<span class="log-line-info">[INDRA] Submitting problem…</span>';

  try {
    const c = parseVec(document.getElementById("vec-c").value);
    if (!c) throw new Error("Objective vector c is required");

    const { A: A_ub, b: b_ub } = parseMatrixInput(document.getElementById("mat-aub").value);
    const { A: A_eq, b: b_eq } = parseMatrixInput(document.getElementById("mat-aeq").value);
    const lb = parseVec(document.getElementById("vec-lb").value) || new Array(c.length).fill(0);
    const ub = parseVec(document.getElementById("vec-ub").value, true);
    const Q = parseQ(document.getElementById("mat-q").value);

    const ivText = document.getElementById("vec-intvars").value.trim();
    const integer_vars = ivText ? ivText.split(",").map(Number) : undefined;

    const payload = {
      name: document.getElementById("prob-name").value || "playground",
      c, A_ub, b_ub, A_eq, b_eq, lb, ub, Q,
      integer_vars,
      maximize: document.getElementById("prob-sense").value === "maximize",
      algorithm: document.getElementById("algo-select").value,
      params: { tol: parseFloat(document.getElementById("param-tol").value) || 1e-8 },
    };

    const result = await apiPost("/api/solve", payload);
    displayResult(result);

  } catch (e) {
    logBox.innerHTML = `<span class="log-line-err">[ERROR] ${e.message}</span>`;
    console.error(e);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<span class="btn-icon">▶</span> Solve with INDRA`;
  }
}

function displayResult(result) {
  const badge = document.getElementById("result-badge");
  const metricsRow = document.getElementById("metrics-row");
  const solutionViz = document.getElementById("solution-viz");
  const logBox = document.getElementById("solver-log");

  const status = result.status || "UNKNOWN";
  const isOptimal = status === "OPTIMAL";

  badge.textContent = status;
  badge.className = "result-badge " + (isOptimal ? "optimal" : status === "INFEASIBLE" ? "infeasible" : "");

  if (isOptimal) {
    metricsRow.style.display = "grid";
    document.getElementById("res-obj").textContent = fmtNum(result.objective_value, 6);
    document.getElementById("res-time").textContent = fmtTime(result.solve_time || 0);
    document.getElementById("res-iter").textContent = result.iterations ?? "—";
    document.getElementById("res-gap").textContent =
      result.optimality_gap != null ? `${(result.optimality_gap * 100).toFixed(4)}%` : "—";
  }

  // Solution vector visualization
  if (isOptimal && result.x && result.x.length > 0) {
    solutionViz.style.display = "block";
    const xBars = document.getElementById("x-bars");
    const xVals = result.x.slice(0, 12);
    const xMax = Math.max(...xVals.map(Math.abs), 1e-10);
    xBars.innerHTML = xVals.map((v, i) => `
      <div class="x-bar-row">
        <div class="x-bar-label">x${i}</div>
        <div class="x-bar-track">
          <div class="x-bar-fill" style="width:${(Math.abs(v)/xMax*100).toFixed(1)}%"></div>
        </div>
        <div class="x-bar-val">${v.toFixed(4)}</div>
      </div>
    `).join("");
  }

  // Log
  const logs = result.logs || result.log || [];
  if (Array.isArray(logs) && logs.length > 0) {
    logBox.innerHTML = logs.map(l => {
      const line = typeof l === "object" ? l.msg : l;
      let cls = "log-line-info";
      if (line.includes("✓") || line.includes("Optimal") || line.includes("optimal")) cls = "log-line-ok";
      if (line.includes("[WARN]")) cls = "log-line-warn";
      if (line.includes("[ERROR]")) cls = "log-line-err";
      return `<div class="${cls}">${escHtml(line)}</div>`;
    }).join("");
  } else {
    logBox.innerHTML = `<div class="log-line-ok">Status: ${status} | Obj: ${fmtNum(result.objective_value, 6)} | Time: ${fmtTime(result.solve_time || 0)}</div>`;
  }

  logBox.scrollTop = logBox.scrollHeight;
}

function escHtml(s) {
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

// ============================================================
// DEMO PROBLEMS
// ============================================================

const DEMO_META = {
  crude_blending: {
    icon: "🛢️",
    title: "Crude Oil Blending",
    tagline: "MRPL Core Use Case",
  },
  cdu_scheduling: {
    icon: "🏭",
    title: "CDU Tank Scheduling",
    tagline: "Distillation Unit Operations",
  },
  refinery_lp_small: {
    icon: "⚗️",
    title: "Refinery LP (Small)",
    tagline: "Production Planning",
  },
  refinery_lp_medium: {
    icon: "🔧",
    title: "Refinery LP (Medium)",
    tagline: "Multi-unit Planning",
  },
  power_dispatch: {
    icon: "⚡",
    title: "Power Economic Dispatch",
    tagline: "Unit Commitment & Dispatch",
  },
  supply_chain: {
    icon: "🚢",
    title: "Petroleum Supply Chain",
    tagline: "Multi-echelon Transport",
  },
};

async function loadDemos() {
  const grid = document.getElementById("demos-grid");
  try {
    const data = await apiGet("/api/demos");
    grid.innerHTML = "";

    for (const [key, demo] of Object.entries(data)) {
      const meta = DEMO_META[key] || { icon: "📊", title: key, tagline: "" };
      const card = document.createElement("div");
      card.className = "demo-card";
      card.innerHTML = `
        <div class="demo-icon">${meta.icon}</div>
        <div class="demo-type-badge badge-${demo.problem_type}">${demo.problem_type}</div>
        <div class="demo-name">${meta.title}</div>
        <div class="demo-desc">${demo.description.slice(0, 120)}…</div>
        <div class="demo-stats">
          <div class="demo-stat"><strong>${demo.n_vars}</strong> vars</div>
          <div class="demo-stat"><strong>${demo.n_integer}</strong> integer</div>
          <div class="demo-stat"><strong>${demo.n_ineq}</strong> ineq</div>
          <div class="demo-stat"><strong>${demo.n_eq}</strong> eq</div>
        </div>
        <button class="demo-run-btn" id="demo-btn-${key}" onclick="runDemo('${key}')">
          <span>▶</span> Solve with INDRA
        </button>
      `;
      grid.appendChild(card);
    }
  } catch (e) {
    grid.innerHTML = `<div class="demo-loading">⚠️ Cannot connect to INDRA server. Start the server first.</div>`;
  }
}

async function runDemo(key) {
  const btn = document.getElementById(`demo-btn-${key}`);
  if (btn) { btn.disabled = true; btn.textContent = "Solving…"; }
  showLoading(`Solving: ${DEMO_META[key]?.title || key}`, "INDRA engine running…");

  try {
    const result = await apiPost(`/api/demos/${key}`, { algorithm: "auto" });
    hideLoading();
    showDemoResult(key, result);
  } catch (e) {
    hideLoading();
    alert("Error: " + e.message);
  } finally {
    if (btn) { btn.disabled = false; btn.innerHTML = "<span>▶</span> Solve with INDRA"; }
  }
}

function showDemoResult(key, result) {
  const panel = document.getElementById("demo-result-panel");
  const inner = document.getElementById("demo-result-inner");
  const meta = DEMO_META[key] || { icon: "📊", title: key };
  const isOpt = result.status === "OPTIMAL";

  const sampleVars = (result.solution_preview || []).slice(0, 5)
    .map((v, i) => `x${i}: <strong>${v.toFixed(4)}</strong>`)
    .join("  &nbsp;|&nbsp;  ");

  const psStats = result.presolve_stats || {};

  inner.innerHTML = `
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:20px;">
      <span style="font-size:2rem;">${meta.icon}</span>
      <div>
        <div style="font-size:1.2rem;font-weight:700;">${meta.title}</div>
        <div style="font-size:0.82rem;color:var(--text-muted);">${result.description || ""}</div>
      </div>
      <div style="margin-left:auto;">
        <span class="result-badge ${isOpt ? 'optimal' : 'infeasible'}">${result.status}</span>
      </div>
    </div>

    <div class="demo-result-grid">
      <div class="demo-result-metric">
        <div class="metric-icon">🎯</div>
        <div class="metric-val">${result.objective_value != null ? result.objective_value.toFixed(4) : "—"}</div>
        <div class="metric-lbl">Objective</div>
      </div>
      <div class="demo-result-metric">
        <div class="metric-icon">⏱️</div>
        <div class="metric-val">${fmtTime(result.solve_time || 0)}</div>
        <div class="metric-lbl">Solve Time</div>
      </div>
      <div class="demo-result-metric">
        <div class="metric-icon">🔄</div>
        <div class="metric-val">${result.iterations ?? "—"}</div>
        <div class="metric-lbl">Iterations</div>
      </div>
      <div class="demo-result-metric">
        <div class="metric-icon">🌳</div>
        <div class="metric-val">${result.nodes_explored || "—"}</div>
        <div class="metric-lbl">B&B Nodes</div>
      </div>
      <div class="demo-result-metric">
        <div class="metric-icon">📐</div>
        <div class="metric-val">${result.optimality_gap != null ? (result.optimality_gap * 100).toFixed(3) + "%" : "—"}</div>
        <div class="metric-lbl">Gap</div>
      </div>
    </div>

    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px;">
      <div>
        <div class="section-label">Algorithm Used</div>
        <div style="font-family:var(--mono);font-size:0.85rem;color:var(--saffron-light);">${result.algorithm || "—"}</div>
        <div class="section-label mt-4">Problem Size</div>
        <div style="font-size:0.85rem;color:var(--text-sec);">
          ${result.n_vars} vars → ${result.n_vars_after_presolve} after presolve
          <span style="color:var(--green);"> (↓${result.n_vars - result.n_vars_after_presolve} removed)</span>
        </div>
        <div class="section-label mt-4">Residuals</div>
        <div style="font-family:var(--mono);font-size:0.8rem;color:var(--text-sec);">
          Primal: ${(result.primal_residual || 0).toExponential(2)}<br/>
          Dual: ${(result.dual_residual || 0).toExponential(2)}
        </div>
      </div>
      <div>
        <div class="section-label">Presolve Stats</div>
        <div style="font-family:var(--mono);font-size:0.8rem;color:var(--text-sec);">
          Vars removed: ${psStats.vars_removed || 0}<br/>
          Cons removed: ${psStats.cons_removed || 0}<br/>
          Bounds tightened: ${psStats.bounds_tightened || 0}<br/>
          Rounds: ${psStats.rounds || 0}
        </div>
        ${sampleVars ? `
          <div class="section-label mt-4">Solution Preview</div>
          <div style="font-family:var(--mono);font-size:0.78rem;color:var(--text-sec);">${sampleVars}</div>
        ` : ""}
      </div>
    </div>

    <div class="section-label mt-4">Solver Log</div>
    <div class="log-box" style="max-height:200px;">
      ${(result.logs || []).map(l => {
        const line = typeof l === "object" ? l.msg : l;
        let cls = "log-line-info";
        if (line.includes("✓")) cls = "log-line-ok";
        if (line.includes("[WARN]")) cls = "log-line-warn";
        if (line.includes("[ERROR]")) cls = "log-line-err";
        return `<div class="${cls}">${escHtml(line)}</div>`;
      }).join("") || "<div class='log-placeholder'>No logs available</div>"}
    </div>
  `;

  panel.style.display = "block";
  panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function closeDemoResult() {
  document.getElementById("demo-result-panel").style.display = "none";
}

// ============================================================
// BENCHMARKS
// ============================================================

async function runBenchmark() {
  const btn = document.getElementById("run-bench-btn");
  btn.disabled = true;
  btn.innerHTML = `<div class="spinner" style="width:20px;height:20px;margin:0;border-width:2px;"></div> Running…`;
  showLoading("Running Benchmarks", "Solving all 6 demo problems with INDRA…");

  try {
    const data = await apiPost("/api/benchmark", {});
    hideLoading();
    renderBenchmarkTable(data.benchmark_results);
    renderBenchmarkChart(data.benchmark_results);
  } catch (e) {
    hideLoading();
    alert("Benchmark error: " + e.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<span class="btn-icon">🏁</span> Run Live Benchmark`;
  }
}

function renderBenchmarkTable(results) {
  const tbody = document.getElementById("bench-tbody");
  tbody.innerHTML = "";

  for (const [name, res] of Object.entries(results)) {
    const meta = DEMO_META[name] || { title: name };
    const tr = document.createElement("tr");
    if (res.skipped) {
      tr.innerHTML = `<td>${meta.title || name}</td><td colspan="8" style="color:var(--text-muted);">Skipped: ${res.reason}</td>`;
    } else if (res.error) {
      tr.innerHTML = `<td>${meta.title || name}</td><td colspan="8" style="color:#EF4444;">${res.error}</td>`;
    } else {
      const statusCls = res.status === "OPTIMAL" ? "status-optimal" : res.status === "INFEASIBLE" ? "status-infeasible" : "status-unknown";
      tr.innerHTML = `
        <td>${meta.title || name}</td>
        <td><span class="demo-type-badge badge-${res.problem_type}" style="font-size:0.7rem;">${res.problem_type}</span></td>
        <td>${res.n_vars}</td>
        <td>${res.n_integer || 0}</td>
        <td style="font-family:var(--mono);font-size:0.8rem;">${res.algorithm}</td>
        <td class="${statusCls}">${res.status}</td>
        <td style="font-family:var(--mono);">${res.objective != null ? res.objective.toFixed(4) : "—"}</td>
        <td style="font-family:var(--mono);">${res.time?.toFixed(4) || "—"}</td>
        <td>${res.iterations || "—"}</td>
      `;
    }
    tbody.appendChild(tr);
  }
}

function renderBenchmarkChart(results) {
  const wrap = document.querySelector(".bench-chart-wrap");
  const canvas = document.getElementById("bench-chart");
  if (!canvas) return;
  wrap.style.display = "block";

  const ctx = canvas.getContext("2d");
  const W = canvas.offsetWidth || 800;
  const H = 250;
  canvas.width = W;
  canvas.height = H;

  const validResults = Object.entries(results)
    .filter(([, r]) => !r.skipped && !r.error && r.time != null)
    .slice(0, 8);

  if (!validResults.length) return;

  const labels = validResults.map(([k]) => DEMO_META[k]?.title || k);
  const times = validResults.map(([, r]) => r.time);
  const maxT = Math.max(...times, 0.001);

  const pad = { t: 20, r: 20, b: 60, l: 60 };
  const barW = (W - pad.l - pad.r) / times.length * 0.6;
  const barGap = (W - pad.l - pad.r) / times.length * 0.4;

  ctx.clearRect(0, 0, W, H);

  // Background
  ctx.fillStyle = "rgba(255,255,255,0.02)";
  ctx.roundRect(0, 0, W, H, 12);
  ctx.fill();

  // Grid lines
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + (H - pad.t - pad.b) * (1 - i / 4);
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(W - pad.r, y);
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.stroke();
    const val = (maxT * i / 4).toFixed(3);
    ctx.fillStyle = "rgba(255,255,255,0.35)";
    ctx.font = "10px 'JetBrains Mono'";
    ctx.textAlign = "right";
    ctx.fillText(val + "s", pad.l - 6, y + 4);
  }

  times.forEach((t, i) => {
    const x = pad.l + i * (barW + barGap) + barGap / 2;
    const barH = ((H - pad.t - pad.b) * t) / maxT;
    const y = H - pad.b - barH;

    const grd = ctx.createLinearGradient(0, y, 0, H - pad.b);
    grd.addColorStop(0, "#FF6B35");
    grd.addColorStop(1, "#7C3AED");

    ctx.beginPath();
    ctx.roundRect(x, y, barW, barH, [4, 4, 0, 0]);
    ctx.fillStyle = grd;
    ctx.fill();

    // Time label
    ctx.fillStyle = "#FFD700";
    ctx.font = "bold 9px 'JetBrains Mono'";
    ctx.textAlign = "center";
    ctx.fillText(t < 0.001 ? "<1ms" : t.toFixed(3) + "s", x + barW / 2, y - 6);

    // X-axis label
    ctx.fillStyle = "rgba(255,255,255,0.5)";
    ctx.font = "9px 'Outfit'";
    ctx.save();
    ctx.translate(x + barW / 2, H - pad.b + 10);
    ctx.rotate(-Math.PI / 6);
    ctx.fillText(labels[i].slice(0, 18), 0, 0);
    ctx.restore();
  });

  ctx.fillStyle = "rgba(255,255,255,0.6)";
  ctx.font = "bold 11px 'Outfit'";
  ctx.textAlign = "center";
  ctx.fillText("INDRA Solve Time by Problem (seconds)", W / 2, 14);
}

// ============================================================
// NAVBAR SCROLL HIGHLIGHTING
// ============================================================

function initScrollSpy() {
  const sections = ["hero", "algorithms", "playground", "demos", "benchmark"];
  const links = sections.map(id => ({
    id,
    el: document.getElementById(id),
    link: document.getElementById("nav-" + {
      hero:"home", algorithms:"algo", playground:"playground", demos:"demos", benchmark:"bench"
    }[id]),
  }));

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        const item = links.find(l => l.el === e.target);
        if (item?.link) {
          links.forEach(l => l.link?.classList.remove("active"));
          item.link.classList.add("active");
        }
      }
    });
  }, { threshold: 0.3 });

  links.forEach(({ el }) => el && observer.observe(el));
}

// ============================================================
// INIT
// ============================================================

window.addEventListener("DOMContentLoaded", () => {
  checkServerStatus();
  initHeroCanvas();
  loadDemos();
  initScrollSpy();
  onTypeChange(); // set initial UI state

  // Retry server status every 10s
  setInterval(checkServerStatus, 10000);
});
