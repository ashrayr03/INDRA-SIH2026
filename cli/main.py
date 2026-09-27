"""
INDRA Solver CLI
================
Command-line interface for the INDRA optimization solver.

Usage:
  python -m cli.main solve --help
  python -m cli.main demo crude_blending
  python -m cli.main benchmark
  python -m cli.main server
"""

import sys
import os
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from solver.problem import OptimizationProblem, SolveStatus
from solver.simplex import RevisedSimplexSolver
from solver.interior_point import InteriorPointSolver
from solver.branch_and_bound import BranchAndBoundSolver
from solver.pdhg import PDHGSolver, GPU_AVAILABLE
from solver.qp_solver import QPSolver
from solver.presolve import Presolve
from demos.refinery_demos import get_all_demos


BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║    ██╗███╗   ██╗██████╗ ██████╗  █████╗                     ║
║    ██║████╗  ██║██╔══██╗██╔══██╗██╔══██╗                    ║
║    ██║██╔██╗ ██║██║  ██║██████╔╝███████║                    ║
║    ██║██║╚██╗██║██║  ██║██╔══██╗██╔══██║                    ║
║    ██║██║ ╚████║██████╔╝██║  ██║██║  ██║                    ║
║    ╚═╝╚═╝  ╚═══╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝                   ║
║                                                              ║
║  Indigenous Numerical Decision-making & Resource Allocator   ║
║  Version 0.1.0-prototype | SIH 2026 | MRPL                  ║
║  GPU: {gpu}                                           ║
╚══════════════════════════════════════════════════════════════╝
"""


def print_banner():
    gpu_str = "ENABLED (CuPy)" if GPU_AVAILABLE else "Not available (CPU mode)"
    print(BANNER.format(gpu=gpu_str.ljust(30)))


def print_result(result, problem_name=""):
    print(f"\n{'='*60}")
    print(f"  INDRA Solution Report — {problem_name}")
    print(f"{'='*60}")
    print(f"  Status       : {result.status.value}")
    print(f"  Objective    : {result.objective_value}")
    print(f"  Solve Time   : {result.solve_time:.4f} s")
    print(f"  Iterations   : {result.iterations}")
    print(f"  Algorithm    : {result.algorithm}")
    if result.nodes_explored > 0:
        print(f"  B&B Nodes    : {result.nodes_explored}")
    if result.primal_residual:
        print(f"  Primal Res.  : {result.primal_residual:.4e}")
    if result.dual_residual:
        print(f"  Dual Res.    : {result.dual_residual:.4e}")
    if result.optimality_gap:
        print(f"  Gap          : {result.optimality_gap:.4f}%")
    if result.x is not None:
        x_preview = result.x[:8]
        print(f"  Solution x*  : [{', '.join(f'{v:.4f}' for v in x_preview)}"
              f"{'…]' if len(result.x) > 8 else ']'}")
    print(f"{'='*60}\n")


def run_demo(demo_name: str, algorithm: str = "auto", verbose: bool = True):
    """Solve a named demo problem."""
    demos = get_all_demos()
    if demo_name not in demos:
        print(f"[ERROR] Demo '{demo_name}' not found.")
        print(f"Available: {', '.join(demos.keys())}")
        return

    demo = demos[demo_name]
    problem = demo["problem"]

    print(f"\n[INDRA] Demo: {demo_name}")
    print(f"  {demo.get('description', '')}")
    print(f"  Type: {problem.problem_type.value}")
    print(f"  Variables: {problem.n_vars} | Integers: {len(problem.integer_vars or [])}")
    print(f"  Constraints: {problem.n_ineq} ineq + {problem.n_eq} eq")

    # Presolve
    ps = Presolve(verbose=verbose)
    problem_ps, ps_stats = ps.presolve(problem)
    print(f"  Presolve: -{ps_stats['vars_removed']} vars, -{ps_stats['cons_removed']} cons")

    # Select algorithm
    if algorithm == "auto":
        if problem.Q is not None:
            algorithm = "qp_admm"
        elif problem.integer_vars:
            algorithm = "bnb"
        elif problem.n_vars > 2000:
            algorithm = "pdhg"
        else:
            algorithm = "ipm"

    print(f"  Algorithm: {algorithm.upper()}")
    print()

    solvers = {
        "simplex": lambda: RevisedSimplexSolver(verbose=verbose),
        "ipm": lambda: InteriorPointSolver(verbose=verbose),
        "pdhg": lambda: PDHGSolver(verbose=verbose),
        "bnb": lambda: BranchAndBoundSolver(verbose=verbose),
        "qp_admm": lambda: QPSolver(verbose=verbose),
    }

    if algorithm not in solvers:
        print(f"[ERROR] Unknown algorithm: {algorithm}")
        return

    solver = solvers[algorithm]()
    result = solver.solve(problem_ps)

    if result.is_optimal() and result.x is not None:
        x_full = ps.postsolve(result.x)
        result.x = x_full

    print_result(result, demo_name)


def run_benchmark(verbose: bool = False):
    """Quick benchmark of all demo problems."""
    print("\n[INDRA] Running benchmark across all demo problems…\n")
    demos = get_all_demos()

    results = []
    header = f"{'Problem':<25} {'Type':<6} {'Vars':>6} {'Int':>5} {'Algo':<10} {'Status':<12} {'Obj':>14} {'Time':>8} {'Iters':>7}"
    print(header)
    print("-" * len(header))

    for name, demo in demos.items():
        problem = demo["problem"]

        if problem.integer_vars:
            algo = "bnb"
        elif problem.Q is not None:
            algo = "qp_admm"
        else:
            algo = "ipm"

        solvers = {
            "ipm": InteriorPointSolver(verbose=False),
            "bnb": BranchAndBoundSolver(verbose=False, max_nodes=5000, max_time=30),
            "qp_admm": QPSolver(verbose=False),
        }

        ps = Presolve(verbose=False)
        problem_ps, _ = ps.presolve(problem)

        solver = solvers[algo]
        t0 = time.time()
        try:
            result = solver.solve(problem_ps)
            elapsed = time.time() - t0
            row = (
                f"{name[:24]:<25} {problem.problem_type.value:<6} "
                f"{problem.n_vars:>6} {len(problem.integer_vars or []):>5} "
                f"{algo:<10} {result.status.value:<12} "
                f"{result.objective_value if result.objective_value is not None else '—':>14.4f} "
                f"{elapsed:>7.3f}s {result.iterations:>7}"
            )
        except Exception as e:
            elapsed = time.time() - t0
            row = f"{name[:24]:<25} {'':6} {'':>6} {'':>5} {algo:<10} {'ERROR':<12} {str(e)[:30]}"

        print(row)
        results.append({"name": name, "algo": algo, "time": elapsed})

    print(f"\n[DONE] Benchmark complete. Total problems: {len(demos)}")


def cmd_solve(args):
    """Solve a problem from a JSON file."""
    if not args.file:
        print("[ERROR] Provide --file path to JSON problem file")
        return

    with open(args.file) as f:
        data = json.load(f)

    problem = OptimizationProblem(
        c=np.array(data["c"]),
        A_ub=np.array(data["A_ub"]) if data.get("A_ub") else None,
        b_ub=np.array(data["b_ub"]) if data.get("b_ub") else None,
        A_eq=np.array(data["A_eq"]) if data.get("A_eq") else None,
        b_eq=np.array(data["b_eq"]) if data.get("b_eq") else None,
        lb=np.array(data.get("lb", [0]*len(data["c"]))),
        ub=np.array(data["ub"]) if data.get("ub") else None,
        integer_vars=data.get("integer_vars"),
        name=data.get("name", args.file),
    )

    ps = Presolve()
    problem_ps, _ = ps.presolve(problem)

    algo = args.algorithm or "auto"
    if algo == "auto":
        algo = "bnb" if problem.integer_vars else "ipm"

    solvers = {
        "simplex": RevisedSimplexSolver(verbose=True),
        "ipm": InteriorPointSolver(verbose=True),
        "pdhg": PDHGSolver(verbose=True),
        "bnb": BranchAndBoundSolver(verbose=True),
        "qp_admm": QPSolver(verbose=True),
    }

    result = solvers[algo].solve(problem_ps)
    print_result(result, problem.name)

    if args.output and result.x is not None:
        out = result.to_dict()
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[INDRA] Solution written to {args.output}")


def main():
    print_banner()

    parser = argparse.ArgumentParser(
        description="INDRA — Indigenous Optimization Solver CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # demo subcommand
    p_demo = subparsers.add_parser("demo", help="Solve a named refinery demo problem")
    p_demo.add_argument("name", nargs="?", default=None,
                         help="Demo name (crude_blending, cdu_scheduling, etc.)")
    p_demo.add_argument("--algorithm", "-a", default="auto",
                         choices=["auto", "simplex", "ipm", "pdhg", "bnb", "qp_admm"])
    p_demo.add_argument("--list", action="store_true", help="List available demos")

    # benchmark subcommand
    p_bench = subparsers.add_parser("benchmark", help="Run benchmark on all demos")
    p_bench.add_argument("--verbose", action="store_true")

    # solve subcommand
    p_solve = subparsers.add_parser("solve", help="Solve LP/MILP from JSON file")
    p_solve.add_argument("--file", "-f", required=True, help="Path to JSON problem file")
    p_solve.add_argument("--algorithm", "-a", default="auto")
    p_solve.add_argument("--output", "-o", help="Output JSON file for solution")

    # server subcommand
    p_server = subparsers.add_parser("server", help="Start the INDRA API server + web dashboard")
    p_server.add_argument("--port", type=int, default=5050)

    args = parser.parse_args()

    if args.command == "demo":
        if args.list or args.name is None:
            demos = get_all_demos()
            print("[INDRA] Available demo problems:\n")
            for name, demo in demos.items():
                p = demo["problem"]
                print(f"  {name:<25} {p.problem_type.value:<6} {p.n_vars:>5} vars  {len(p.integer_vars or []):>3} int")
            print()
        else:
            run_demo(args.name, args.algorithm)

    elif args.command == "benchmark":
        run_benchmark(args.verbose)

    elif args.command == "solve":
        cmd_solve(args)

    elif args.command == "server":
        print(f"[INDRA] Starting server on http://localhost:{args.port}")
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        os.environ["PORT"] = str(args.port)
        from api.server import app
        app.run(host="0.0.0.0", port=args.port, debug=False)

    else:
        parser.print_help()
        print("\nQuick start:")
        print("  python -m cli.main demo --list")
        print("  python -m cli.main demo crude_blending")
        print("  python -m cli.main benchmark")
        print("  python -m cli.main server")


if __name__ == "__main__":
    main()
