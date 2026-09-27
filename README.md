# INDRA — Indigenous Optimization Solver

**INDRA** (Indigenous Numerical Decision-making & Resource Allocator) is a from-scratch, pure-Python mathematical optimization engine built for **Smart India Hackathon (SIH) 2026**, Problem Statement **SIH26119** by **MRPL (Mangalore Refinery and Petrochemicals Limited)**.

> 🏆 A sovereign, dependency-light solver for Linear Programming (LP), Mixed-Integer LP (MILP), Quadratic Programming (QP), and real-world refinery production optimization — with a built-in interactive web dashboard.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Revised Simplex** | Phase I / Phase II LP solver with anti-cycling |
| **Interior Point (IPM)** | Primal-Dual log-barrier with Mehrotra predictor-corrector |
| **Branch & Bound** | Full MILP solver with presolve, cutting planes & best-first search |
| **QP Solver** | Active-set method for convex quadratic programs |
| **PDHG** | First-order primal-dual hybrid gradient solver |
| **Presolve** | Variable fixing, bound tightening, redundant constraint removal |
| **Refinery Demos** | Small / medium / large MRPL-style production LP demos |
| **REST API** | Flask-based JSON API for solve requests |
| **Web Dashboard** | Interactive real-time visualization of solver progress & results |

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/indigenous-solver.git
cd indigenous-solver
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run everything (tests + web server)

```bash
python run.py
```

This will:
1. Run quick solver sanity tests (LP, MILP, Refinery demo)
2. Launch the web dashboard at **http://localhost:5050**
3. Automatically open your browser

### Other launch modes

```bash
# Run tests only (no server)
python run.py --test-only

# Start server only (skip tests)
python run.py --server-only
```

---

## 📁 Project Structure

```
IndigenousSolver/
├── run.py                  # Main entry point (tests + server)
├── requirements.txt        # Python dependencies
│
├── solver/                 # Core optimization engine
│   ├── problem.py          # OptimizationProblem data model
│   ├── simplex.py          # Revised Simplex solver
│   ├── interior_point.py   # Interior Point Method (IPM)
│   ├── branch_and_bound.py # Branch & Bound MILP solver
│   ├── qp_solver.py        # Quadratic Programming solver
│   ├── pdhg.py             # PDHG first-order solver
│   └── presolve.py         # Presolve reductions
│
├── api/
│   └── server.py           # Flask REST API
│
├── web/
│   ├── index.html          # Web dashboard UI
│   ├── style.css           # Styles
│   └── app.js              # Frontend logic
│
├── demos/
│   └── refinery_demos.py   # MRPL refinery production demos
│
├── cli/                    # Command-line interface
├── benchmarks/             # Benchmark scripts
└── tests/                  # Test suite
```

---

## 🛢️ Refinery Use Case (MRPL)

The solver is purpose-built for MRPL's refinery production planning problem:

- **Crude allocation** across multiple processing units
- **Product yield** optimization (LPG, Naphtha, Diesel, HFO, etc.)
- **Capacity and demand constraints**
- **Operating cost minimization** subject to blend specs

Demo sizes: `small`, `medium`, `large`

```python
from demos.refinery_demos import refinery_production_lp
demo = refinery_production_lp("medium")
```

---

## 🌐 Web API

The REST API is served at `http://localhost:5050/api/`.

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Health check |
| `/api/solve` | POST | Submit an optimization problem |
| `/api/demos` | GET | List built-in demo problems |
| `/api/demos/<name>` | GET | Load a specific demo |

---

## 🧠 Algorithms

### Revised Simplex
Full two-phase implementation with LU factorization, Bland's rule anti-cycling, and degeneracy handling.

### Interior Point Method (IPM)
Primal-dual log-barrier with Mehrotra predictor-corrector step and adaptive barrier parameter μ.

### Branch & Bound (MILP)
Best-first node selection, LP relaxation at each node, variable branching heuristics, cutting plane support.

---

## 📋 Requirements

- Python 3.9+
- `numpy >= 1.24.0`
- `scipy >= 1.10.0`
- `flask >= 2.3.0`
- `flask-cors >= 4.0.0`

---

## 🏫 About

| Field | Info |
|---|---|
| **Event** | Smart India Hackathon (SIH) 2026 |
| **Problem Statement** | SIH26119 |
| **Organization** | MRPL — Mangalore Refinery and Petrochemicals Limited |
| **Category** | Software |

---

## 📄 License

This project is developed for academic and hackathon purposes. All rights reserved © 2026.
