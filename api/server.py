"""
INDRA Solver REST API
=====================
Flask-based REST API exposing solver capabilities.

Endpoints:
  POST /api/solve          — Submit optimization problem, get solution
  POST /api/solve/stream   — Server-sent events for real-time solve progress
  GET  /api/demos          — List available demo problems
  POST /api/demos/{name}   — Solve a named demo problem
  GET  /api/status         — Solver status and GPU info
  GET  /api/benchmarks     — Benchmark results
"""

import os
import sys
import json
import time
import threading
import traceback
import numpy as np
from typing import Dict, Any

from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver.problem import OptimizationProblem, SolveStatus
from solver.simplex import RevisedSimplexSolver
from solver.interior_point import InteriorPointSolver
from solver.branch_and_bound import BranchAndBoundSolver
from solver.pdhg import PDHGSolver, GPU_AVAILABLE
from solver.qp_solver import QPSolver
from solver.presolve import Presolve
from demos.refinery_demos import get_all_demos

app = Flask(__name__, static_folder="../web", static_url_path="")
CORS(app)

# ---- Shared solve log for SSE ----
_solve_logs: Dict[str, list] = {}
_solve_results: Dict[str, Any] = {}

SOLVER_REGISTRY = {
    "simplex": RevisedSimplexSolver,
    "ipm": InteriorPointSolver,
    "pdhg": PDHGSolver,
    "bnb": BranchAndBoundSolver,
    "qp_admm": QPSolver,
}


def build_problem_from_json(data: Dict) -> OptimizationProblem:
    """Parse problem from JSON API request."""
    c = np.array(data["c"], dtype=float)
    n = len(c)

    A_ub = np.array(data["A_ub"], dtype=float) if data.get("A_ub") else None
    b_ub = np.array(data["b_ub"], dtype=float) if data.get("b_ub") else None
    A_eq = np.array(data["A_eq"], dtype=float) if data.get("A_eq") else None
    b_eq = np.array(data["b_eq"], dtype=float) if data.get("b_eq") else None
    Q = np.array(data["Q"], dtype=float) if data.get("Q") else None
    lb = np.array(data.get("lb", [0.0] * n), dtype=float)
    ub_raw = data.get("ub")
    ub = np.array(ub_raw, dtype=float) if ub_raw else np.full(n, np.inf)
    integer_vars = data.get("integer_vars")
    name = data.get("name", "api_problem")
    maximize = data.get("maximize", False)

    return OptimizationProblem(
        c=c, Q=Q,
        A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        lb=lb, ub=ub,
        integer_vars=integer_vars,
        name=name,
        maximize=maximize,
    )


def select_solver(problem: OptimizationProblem, algo: str, params: Dict, log_cb=None):
    """Instantiate the right solver."""
    ptype = problem.problem_type.value

    if algo == "auto":
        if problem.Q is not None:
            algo = "qp_admm"
        elif problem.integer_vars:
            algo = "bnb"
        elif problem.n_vars > 2000 or problem.n_ineq + problem.n_eq > 2000:
            algo = "pdhg"
        else:
            algo = "ipm"

    common = {"verbose": False, "log_callback": log_cb}

    if algo == "simplex":
        return RevisedSimplexSolver(
            max_iterations=params.get("max_iter", 50000),
            tolerance=params.get("tol", 1e-8),
            pivot_rule=params.get("pivot_rule", "dantzig"),
            **common,
        ), algo
    elif algo == "ipm":
        return InteriorPointSolver(
            max_iterations=params.get("max_iter", 200),
            tolerance=params.get("tol", 1e-8),
            **common,
        ), algo
    elif algo == "pdhg":
        return PDHGSolver(
            max_iterations=params.get("max_iter", 100000),
            tolerance=params.get("tol", 1e-6),
            use_gpu=params.get("use_gpu", True),
            log_interval=params.get("log_interval", 1000),
            **common,
        ), algo
    elif algo == "bnb":
        return BranchAndBoundSolver(
            max_nodes=params.get("max_nodes", 100000),
            max_time=params.get("max_time", 120.0),
            tolerance=params.get("tol", 1e-6),
            gomory_cuts=params.get("gomory_cuts", True),
            **common,
        ), algo
    elif algo == "qp_admm":
        return QPSolver(
            method=params.get("qp_method", "admm"),
            max_iterations=params.get("max_iter", 10000),
            tolerance=params.get("tol", 1e-6),
            **common,
        ), algo
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


# ============================================================
# Routes
# ============================================================

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/status")
def status():
    return jsonify({
        "solver": "INDRA",
        "version": "0.1.0-prototype",
        "full_name": "Indigenous Numerical Decision-making and Resource Allocator",
        "gpu_available": GPU_AVAILABLE,
        "algorithms": list(SOLVER_REGISTRY.keys()),
        "problem_types": ["LP", "MILP", "QP", "MIQP"],
        "timestamp": time.time(),
    })


@app.route("/api/demos")
def list_demos():
    demos = get_all_demos()
    result = {}
    for key, demo in demos.items():
        p = demo["problem"]
        result[key] = {
            "name": key,
            "description": demo.get("description", ""),
            "problem_type": p.problem_type.value,
            "n_vars": p.n_vars,
            "n_ineq": p.n_ineq,
            "n_eq": p.n_eq,
            "n_integer": len(p.integer_vars) if p.integer_vars else 0,
        }
    return jsonify(result)


@app.route("/api/demos/<demo_name>", methods=["POST"])
def solve_demo(demo_name):
    demos = get_all_demos()
    if demo_name not in demos:
        return jsonify({"error": f"Demo '{demo_name}' not found"}), 404

    demo = demos[demo_name]
    problem = demo["problem"]
    params = request.json or {}
    algo = params.get("algorithm", "auto")

    logs = []
    log_cb = lambda msg: logs.append({"time": time.time(), "msg": msg})

    # Presolve
    ps = Presolve(verbose=False)
    problem_ps, ps_stats = ps.presolve(problem)

    solver, algo_used = select_solver(problem_ps, algo, params, log_cb=log_cb)

    t0 = time.time()
    result = solver.solve(problem_ps)
    elapsed = time.time() - t0

    if result.is_optimal() and result.x is not None:
        x_full = ps.postsolve(result.x)
    else:
        x_full = None

    return jsonify({
        "demo": demo_name,
        "description": demo.get("description", ""),
        "algorithm": algo_used,
        "status": result.status.value,
        "objective_value": result.objective_value,
        "solve_time": round(elapsed, 4),
        "iterations": result.iterations,
        "nodes_explored": result.nodes_explored,
        "primal_residual": result.primal_residual,
        "dual_residual": result.dual_residual,
        "optimality_gap": result.optimality_gap,
        "presolve_stats": ps_stats,
        "n_vars": problem.n_vars,
        "n_vars_after_presolve": problem_ps.n_vars,
        "problem_type": problem.problem_type.value,
        "solution_preview": x_full[:10].tolist() if x_full is not None else None,
        "logs": logs[-50:],
    })


@app.route("/api/solve", methods=["POST"])
def solve():
    data = request.json
    if not data or "c" not in data:
        return jsonify({"error": "Missing required field 'c' (objective vector)"}), 400

    try:
        problem = build_problem_from_json(data)
    except Exception as e:
        return jsonify({"error": f"Problem parsing error: {str(e)}"}), 400

    params = data.get("params", {})
    algo = data.get("algorithm", "auto")

    logs = []
    log_cb = lambda msg: logs.append({"time": time.time(), "msg": msg})

    try:
        # Apply presolve
        ps = Presolve(verbose=False)
        problem_ps, ps_stats = ps.presolve(problem)

        solver, algo_used = select_solver(problem_ps, algo, params, log_cb=log_cb)
        t0 = time.time()
        result = solver.solve(problem_ps)
        elapsed = time.time() - t0

        if result.is_optimal() and result.x is not None:
            x_full = ps.postsolve(result.x)
        else:
            x_full = None

        return jsonify({
            "status": result.status.value,
            "algorithm": algo_used,
            "objective_value": result.objective_value,
            "x": x_full.tolist() if x_full is not None else None,
            "solve_time": round(elapsed, 6),
            "iterations": result.iterations,
            "nodes_explored": result.nodes_explored,
            "primal_residual": result.primal_residual,
            "dual_residual": result.dual_residual,
            "optimality_gap": result.optimality_gap,
            "presolve_stats": ps_stats,
            "problem_type": problem.problem_type.value,
            "logs": [l["msg"] for l in logs[-100:]],
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc(),
        }), 500


@app.route("/api/benchmark", methods=["POST"])
def run_benchmark():
    """Run a quick benchmark across all demos."""
    params = request.json or {}
    demos = get_all_demos()
    results = {}

    for name, demo in demos.items():
        problem = demo["problem"]
        algo = "auto"

        # Skip large problems in quick benchmark
        if problem.n_vars > 500 and not params.get("full", False):
            results[name] = {"skipped": True, "reason": "Too large for quick benchmark"}
            continue

        logs = []
        try:
            ps = Presolve(verbose=False)
            problem_ps, ps_stats = ps.presolve(problem)
            solver, algo_used = select_solver(problem_ps, algo, params, log_cb=lambda m: logs.append(m))
            t0 = time.time()
            result = solver.solve(problem_ps)
            elapsed = time.time() - t0

            results[name] = {
                "status": result.status.value,
                "objective": result.objective_value,
                "time": round(elapsed, 4),
                "algorithm": algo_used,
                "iterations": result.iterations,
                "nodes": result.nodes_explored,
                "problem_type": problem.problem_type.value,
                "n_vars": problem.n_vars,
                "n_integer": len(problem.integer_vars) if problem.integer_vars else 0,
            }
        except Exception as e:
            results[name] = {"error": str(e)}

    return jsonify({
        "benchmark_results": results,
        "timestamp": time.time(),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"\n{'='*60}")
    print("  INDRA Optimization Solver — API Server")
    print(f"  http://localhost:{port}")
    print(f"  GPU: {'Available ✓' if GPU_AVAILABLE else 'Not available (CPU mode)'}")
    print(f"{'='*60}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
