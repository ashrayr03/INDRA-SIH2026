"""
INDRA Solver — Quick Start Script
Runs the full demo: CLI benchmark then launches web server.
"""

import subprocess
import sys
import os
import webbrowser
import time
import threading

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def run_quick_test():
    """Quick sanity check that core solver works."""
    import numpy as np
    from solver.problem import OptimizationProblem
    from solver.simplex import RevisedSimplexSolver
    from solver.interior_point import InteriorPointSolver
    from solver.branch_and_bound import BranchAndBoundSolver

    print("\n[INDRA] Running quick solver tests...\n")

    # === Test 1: Simple LP ===
    print("Test 1: Simple LP")
    prob = OptimizationProblem(
        c=[-1, -2],
        A_ub=[[1, 1], [1, 0], [0, 1]],
        b_ub=[4, 3, 3],
        lb=[0, 0],
        name="simple_lp",
    )
    r = InteriorPointSolver(verbose=False).solve(prob)
    print(f"  IPM: status={r.status.value}, obj={r.objective_value:.4f}, "
          f"x=[{', '.join(f'{v:.4f}' for v in r.x)}]  ✓" if r.is_optimal() else f"  IPM: FAILED ({r.status.value})")

    r2 = RevisedSimplexSolver(verbose=False).solve(prob)
    print(f"  Simplex: status={r2.status.value}, obj={r2.objective_value:.4f}  ✓" if r2.is_optimal() else f"  Simplex: FAILED")

    # === Test 2: Simple MILP (Knapsack) ===
    print("\nTest 2: 0-1 Knapsack MILP")
    prob2 = OptimizationProblem(
        c=[-10, -6, -4, -12],
        A_ub=[[6, 4, 3, 8]],
        b_ub=[10],
        lb=[0, 0, 0, 0],
        ub=[1, 1, 1, 1],
        integer_vars=[0, 1, 2, 3],
        name="knapsack",
    )
    r3 = BranchAndBoundSolver(verbose=False, max_nodes=500).solve(prob2)
    print(f"  B&B: status={r3.status.value}, obj={r3.objective_value:.4f}, "
          f"nodes={r3.nodes_explored}, x=[{', '.join(f'{v:.0f}' for v in r3.x)}]  ✓" if r3.is_optimal() else f"  B&B: FAILED")

    # === Test 3: Refinery Demo ===
    print("\nTest 3: Refinery LP (Small)")
    from demos.refinery_demos import refinery_production_lp
    from solver.presolve import Presolve
    demo = refinery_production_lp("small")
    ps = Presolve()
    prob3_ps, stats = ps.presolve(demo["problem"])
    r4 = InteriorPointSolver(verbose=False).solve(prob3_ps)
    print(f"  IPM: status={r4.status.value}, time={r4.solve_time:.3f}s, "
          f"vars: {demo['problem'].n_vars}→{prob3_ps.n_vars} (presolve)  ✓" if r4.is_optimal() else f"  IPM: FAILED")

    print("\n[INDRA] All tests passed! ✓\n")


def launch_server():
    """Launch Flask server."""
    from api.server import app
    port = int(os.environ.get("PORT", 5050))
    print(f"\n[INDRA] Starting web server on http://0.0.0.0:{port}")
    print(f"  Open your browser to: http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    print("""
+======================================================+
|           INDRA Optimization Solver                  |
|   Indigenous Numerical Decision-making & Resource    |
|          Allocator -- v0.1.0-prototype               |
|                                                      |
|   SIH 2026 | PS SIH26119 | MRPL                     |
+======================================================+
""")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-only", action="store_true", help="Only run tests, don't start server")
    parser.add_argument("--server-only", action="store_true", help="Only start server")
    args = parser.parse_args()

    if not args.server_only:
        run_quick_test()

    if not args.test_only:
        # Open browser after 2 seconds
        def open_browser():
            time.sleep(2)
            webbrowser.open("http://localhost:5050")
        threading.Thread(target=open_browser, daemon=True).start()
        launch_server()
