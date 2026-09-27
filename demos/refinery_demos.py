"""
Refinery Optimization Demo Problems
====================================
Industry-specific MILP/LP models for Indian refinery operations:

1. Crude Oil Blending (MILP) — MRPL's core use case
   Select crude oils from different sources, blend to meet
   product specifications (density, sulfur, viscosity, etc.)
   while minimizing procurement cost.

2. Crude Distillation Unit (CDU) Scheduling (MILP)
   Schedule which crude to process in which tank over a planning horizon.

3. Product Blending (QP)
   Blend refinery streams to meet fuel product specifications,
   with nonlinear blending for octane number (linearized).

4. Refinery Production Planning (LP)
   Monthly LP model: process unit capacities, product demands, economics.

5. Power System Economic Dispatch (MILP)
   Unit commitment and economic dispatch for power grid.

6. Transportation/Logistics (MILP)
   Multi-commodity flow for petroleum product pipeline routing.
"""

import numpy as np
from typing import Dict, Any
from solver.problem import OptimizationProblem


# ============================================================
# 1. CRUDE OIL BLENDING MILP
# ============================================================

def crude_blending_milp(n_crudes: int = 8, n_products: int = 4) -> Dict[str, Any]:
    """
    Crude Oil Blending for MRPL.
    
    Variables:
      x[i,j] = volume of crude i used to produce product j (continuous, kBbl)
      y[i]   = 1 if crude i is selected for purchase (binary)
    
    Minimize: total procurement cost
    Subject to:
      - Product demand satisfaction
      - Crude availability limits
      - Quality specs (sulfur %, API gravity, viscosity)
      - If y[i]=0 then x[i,j]=0 for all j  (logical linking)
      - Minimum batch size if crude selected
    """
    rng = np.random.default_rng(42)

    # Crude properties (realistic Indian/Middle-East crudes)
    crude_names = [
        "Arab Heavy", "Arab Light", "Basrah Light", "Iranian Heavy",
        "Bonny Light", "Saharan Blend", "Oman Export", "Kuwait Export"
    ][:n_crudes]

    product_names = ["Naphtha", "Kerosene", "Gas Oil", "Fuel Oil"][:n_products]

    # Cost per kBbl (USD/Bbl approximate)
    cost = np.array([72, 78, 70, 68, 85, 82, 74, 71], dtype=float)[:n_crudes]

    # Crude properties
    sulfur = np.array([2.85, 1.77, 2.1, 2.4, 0.14, 0.09, 1.0, 2.5])[:n_crudes]  # wt%
    api_gravity = np.array([27.4, 33.4, 33.7, 30.4, 33.9, 44.2, 34.7, 31.4])[:n_crudes]

    # Availability (kBbl/month)
    availability = np.array([500, 400, 350, 300, 200, 150, 400, 350], dtype=float)[:n_crudes]

    # Yield matrix: crude i -> product j fraction (simplified linear yield)
    yield_matrix = np.array([
        [0.22, 0.12, 0.28, 0.32],  # Arab Heavy
        [0.26, 0.16, 0.32, 0.20],  # Arab Light
        [0.24, 0.15, 0.30, 0.25],  # Basrah Light
        [0.20, 0.12, 0.28, 0.34],  # Iranian Heavy
        [0.28, 0.18, 0.34, 0.15],  # Bonny Light
        [0.30, 0.20, 0.35, 0.10],  # Saharan
        [0.25, 0.16, 0.31, 0.22],  # Oman
        [0.21, 0.13, 0.29, 0.31],  # Kuwait
    ], dtype=float)[:n_crudes, :n_products]

    # Product demand (kBbl/month)
    demand = np.array([300, 200, 400, 250], dtype=float)[:n_products]

    # Max sulfur in blended feed (wt%)
    max_sulfur_blend = 2.0

    # ---- Variable encoding ----
    # x[i,j] for i=0..n_crudes-1, j=0..n_products-1: volume allocation
    # y[i]   for i=0..n_crudes-1: binary selection
    n_x = n_crudes * n_products
    n_y = n_crudes
    n_total = n_x + n_y

    def x_idx(i, j): return i * n_products + j
    def y_idx(i): return n_x + i

    # ---- Objective: minimize procurement cost ----
    c = np.zeros(n_total)
    for i in range(n_crudes):
        for j in range(n_products):
            c[x_idx(i, j)] = cost[i]  # cost per kBbl of crude allocated

    # ---- Inequality constraints ----
    ineq_rows = []
    ineq_rhs = []

    # 1. Crude availability: sum_j x[i,j] <= availability[i]
    for i in range(n_crudes):
        row = np.zeros(n_total)
        for j in range(n_products):
            row[x_idx(i, j)] = 1.0
        ineq_rows.append(row)
        ineq_rhs.append(float(availability[i]))

    # 2. Linking: x[i,j] <= M * y[i]  (if crude not selected, no allocation)
    M_big = 600.0
    for i in range(n_crudes):
        for j in range(n_products):
            row = np.zeros(n_total)
            row[x_idx(i, j)] = 1.0
            row[y_idx(i)] = -M_big
            ineq_rows.append(row)
            ineq_rhs.append(0.0)

    # 3. Sulfur blend constraint: sum_i (sulfur[i] * sum_j x[i,j]) <= max_sulfur * total_vol
    # Linearized: sum_i sulfur[i] * x[i,j] <= max_sulfur * total_vol_j  for each product
    for j in range(n_products):
        row = np.zeros(n_total)
        for i in range(n_crudes):
            row[x_idx(i, j)] = sulfur[i] - max_sulfur_blend
        ineq_rows.append(row)
        ineq_rhs.append(0.0)

    # 4. Maximum number of crude selections (logistics constraint): sum y <= 6
    row = np.zeros(n_total)
    for i in range(n_crudes):
        row[y_idx(i)] = 1.0
    ineq_rows.append(row)
    ineq_rhs.append(min(6, n_crudes))

    # ---- Equality constraints: product demand satisfaction ----
    eq_rows = []
    eq_rhs = []
    for j in range(n_products):
        row = np.zeros(n_total)
        for i in range(n_crudes):
            row[x_idx(i, j)] = yield_matrix[i, j]
        eq_rows.append(row)
        eq_rhs.append(float(demand[j]))

    A_ub = np.array(ineq_rows)
    b_ub = np.array(ineq_rhs)
    A_eq = np.array(eq_rows)
    b_eq = np.array(eq_rhs)

    lb = np.zeros(n_total)
    ub = np.concatenate([
        np.full(n_x, 600.0),  # max crude allocation per product
        np.ones(n_y),          # binary y in {0,1}
    ])

    integer_vars = list(range(n_x, n_total))

    problem = OptimizationProblem(
        c=c,
        A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        lb=lb, ub=ub,
        integer_vars=integer_vars,
        name="MRPL_CrudeBlending_MILP",
        var_names=[f"x_{crude_names[i]}_{product_names[j]}"
                   for i in range(n_crudes) for j in range(n_products)] +
                  [f"y_{crude_names[i]}" for i in range(n_crudes)],
    )

    return {
        "problem": problem,
        "crude_names": crude_names,
        "product_names": product_names,
        "cost": cost,
        "demand": demand,
        "availability": availability,
        "yield_matrix": yield_matrix,
        "description": (
            f"Crude Oil Blending MILP: Select from {n_crudes} crude sources to meet "
            f"{n_products} product demands while minimizing procurement cost. "
            f"Includes binary crude selection, availability, sulfur quality, and "
            f"logical linking constraints. {n_total} vars ({n_y} binary), "
            f"{A_ub.shape[0]} ineq, {A_eq.shape[0]} eq."
        ),
    }


# ============================================================
# 2. CDU TANK SCHEDULING MILP
# ============================================================

def cdu_scheduling_milp(n_tanks: int = 5, n_periods: int = 4) -> Dict[str, Any]:
    """
    Crude Distillation Unit (CDU) Tank Scheduling.
    
    Schedule which crude tank feeds the CDU in each time period.
    Variables:
      x[t,p] = 1 if tank t feeds CDU in period p (binary)
      v[t,p] = volume drawn from tank t in period p (continuous)
    """
    rng = np.random.default_rng(7)

    tank_capacity = rng.uniform(100, 500, n_tanks)   # kBbl
    tank_initial = rng.uniform(50, tank_capacity)     # initial volume
    cdu_capacity = 120.0  # kBbl/period throughput

    n_x = n_tanks * n_periods
    n_v = n_tanks * n_periods
    n_total = n_x + n_v

    def x_idx(t, p): return t * n_periods + p
    def v_idx(t, p): return n_x + t * n_periods + p

    # Minimize: sum of switching costs (when different tank selected)
    c = np.zeros(n_total)
    switch_cost = 5.0  # kUSD per switch
    for t in range(n_tanks):
        for p in range(1, n_periods):
            # Approximation: x[t,p] != x[t,p-1] costs switching
            # (simplified: penalize selection for secondary periods)
            c[x_idx(t, p)] += switch_cost * 0.1

    ineq_rows = []
    ineq_rhs = []
    eq_rows = []
    eq_rhs = []

    # Exactly one tank feeds CDU per period
    for p in range(n_periods):
        row = np.zeros(n_total)
        for t in range(n_tanks):
            row[x_idx(t, p)] = 1.0
        eq_rows.append(row)
        eq_rhs.append(1.0)

    # CDU throughput: sum_t v[t,p] = cdu_capacity per period
    for p in range(n_periods):
        row = np.zeros(n_total)
        for t in range(n_tanks):
            row[v_idx(t, p)] = 1.0
        eq_rows.append(row)
        eq_rhs.append(cdu_capacity)

    # Linking: v[t,p] <= tank_capacity[t] * x[t,p]
    for t in range(n_tanks):
        for p in range(n_periods):
            row = np.zeros(n_total)
            row[v_idx(t, p)] = 1.0
            row[x_idx(t, p)] = -float(tank_capacity[t])
            ineq_rows.append(row)
            ineq_rhs.append(0.0)

    # Volume balance: tank volume >= 0 in each period
    for t in range(n_tanks):
        cumulative_draw = np.zeros(n_total)
        for p in range(n_periods):
            cumulative_draw[v_idx(t, p)] = 1.0
            row = cumulative_draw.copy()
            ineq_rows.append(row)
            ineq_rhs.append(float(tank_initial[t]))

    A_ub = np.array(ineq_rows) if ineq_rows else None
    b_ub = np.array(ineq_rhs) if ineq_rhs else None
    A_eq = np.array(eq_rows)
    b_eq = np.array(eq_rhs)

    lb = np.zeros(n_total)
    ub = np.concatenate([np.ones(n_x), np.full(n_v, cdu_capacity)])
    integer_vars = list(range(n_x))

    problem = OptimizationProblem(
        c=c,
        A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        lb=lb, ub=ub,
        integer_vars=integer_vars,
        name="MRPL_CDU_Scheduling",
    )

    return {
        "problem": problem,
        "description": (
            f"CDU Tank Scheduling: {n_tanks} tanks × {n_periods} periods. "
            f"Minimize switching costs while meeting CDU throughput. "
            f"{n_total} vars ({n_x} binary)."
        ),
    }


# ============================================================
# 3. REFINERY LP — PRODUCTION PLANNING
# ============================================================

def refinery_production_lp(scale: str = "medium") -> Dict[str, Any]:
    """
    Monthly Refinery LP Production Planning.
    
    Minimizes operating cost while meeting product demands
    and respecting unit capacities. Classic refinery LP.
    
    Process units: CDU, FCC, Reformer, HDS, Coker, Alky
    Products: LPG, Gasoline, Jet, Diesel, HSFO, LSFO
    """
    if scale == "small":
        n_units, n_crudes, n_products = 4, 3, 5
    elif scale == "medium":
        n_units, n_crudes, n_products = 6, 6, 8
    else:  # large
        n_units, n_crudes, n_products = 10, 12, 15

    rng = np.random.default_rng(123)

    unit_names = ["CDU", "VDU", "FCC", "Reformer", "HDS", "Coker",
                  "Alky", "Isomer", "Hydro", "Visbreaker"][:n_units]
    product_names = ["LPG", "Naphtha", "Gasoline", "Jet", "Diesel",
                     "HSFO", "LSFO", "Bitumen", "Coke", "H2",
                     "Propylene", "Benzene", "Toluene", "Xylene", "Lube"][:n_products]

    # Variables: x[u, c] = volume of crude c processed in unit u (kBbl/month)
    #            p[k] = production of product k (kBbl/month)
    n_x = n_units * n_crudes
    n_p = n_products
    n_total = n_x + n_p

    def x_idx(u, c): return u * n_crudes + c
    def p_idx(k): return n_x + k

    # Costs and revenues
    unit_cost = rng.uniform(0.5, 3.0, n_units)    # USD/Bbl operating cost
    product_price = rng.uniform(60, 120, n_products)  # USD/Bbl

    # Objective: minimize cost - revenue (maximize profit)
    c_obj = np.zeros(n_total)
    for u in range(n_units):
        for crude in range(n_crudes):
            c_obj[x_idx(u, crude)] = unit_cost[u]
    for k in range(n_products):
        c_obj[p_idx(k)] = -product_price[k]

    # Unit capacities (kBbl/month)
    capacity = rng.uniform(100, 800, n_units)

    ineq_rows = []
    ineq_rhs = []
    eq_rows = []
    eq_rhs = []

    # Unit capacity constraints
    for u in range(n_units):
        row = np.zeros(n_total)
        for crude in range(n_crudes):
            row[x_idx(u, crude)] = 1.0
        ineq_rows.append(row)
        ineq_rhs.append(float(capacity[u]))

    # Product demand (minimum)
    demand_min = rng.uniform(20, 100, n_products)
    demand_max = demand_min * rng.uniform(1.2, 3.0, n_products)

    for k in range(n_products):
        # p[k] >= demand_min[k]
        row = np.zeros(n_total)
        row[p_idx(k)] = -1.0
        ineq_rows.append(row)
        ineq_rhs.append(-float(demand_min[k]))

        # p[k] <= demand_max[k]
        row2 = np.zeros(n_total)
        row2[p_idx(k)] = 1.0
        ineq_rows.append(row2)
        ineq_rhs.append(float(demand_max[k]))

    # Mass balance: production = yield * input
    yield_factors = rng.uniform(0.05, 0.30, (n_products, n_units, n_crudes))
    for k in range(n_products):
        row = np.zeros(n_total)
        row[p_idx(k)] = 1.0
        for u in range(n_units):
            for crude in range(n_crudes):
                row[x_idx(u, crude)] -= yield_factors[k, u, crude]
        eq_rows.append(row)
        eq_rhs.append(0.0)

    lb = np.zeros(n_total)
    ub = np.concatenate([
        np.full(n_x, max(capacity)),
        np.full(n_p, max(demand_max)),
    ])

    problem = OptimizationProblem(
        c=c_obj,
        A_ub=np.array(ineq_rows),
        b_ub=np.array(ineq_rhs),
        A_eq=np.array(eq_rows) if eq_rows else None,
        b_eq=np.array(eq_rhs) if eq_rhs else None,
        lb=lb, ub=ub,
        name=f"MRPL_RefLP_{scale}",
        var_names=[f"x_{unit_names[u]}_{c}" for u in range(n_units) for c in range(n_crudes)] +
                  [f"p_{product_names[k]}" for k in range(n_products)],
    )

    return {
        "problem": problem,
        "description": (
            f"Refinery Production LP ({scale}): {n_units} process units, "
            f"{n_crudes} crude types, {n_products} products. "
            f"{n_total} vars, {len(ineq_rows)} ineq, {len(eq_rows)} eq."
        ),
    }


# ============================================================
# 4. POWER SYSTEM ECONOMIC DISPATCH (MILP)
# ============================================================

def power_dispatch_milp(n_generators: int = 10, n_periods: int = 6) -> Dict[str, Any]:
    """
    Unit Commitment & Economic Dispatch.
    
    Minimize generation cost over planning horizon.
    Variables: u[g,t] = on/off (binary), p[g,t] = power output (continuous)
    """
    rng = np.random.default_rng(99)

    # Generator parameters
    p_min = rng.uniform(10, 50, n_generators)   # MW minimum output
    p_max = rng.uniform(100, 500, n_generators) # MW maximum output
    cost_a = rng.uniform(1, 5, n_generators)    # $/MW (linear cost)
    cost_b = rng.uniform(100, 500, n_generators) # $/h startup cost
    min_up = rng.integers(1, 4, n_generators)   # min up time periods
    min_dn = rng.integers(1, 4, n_generators)   # min down time periods

    # Load demand per period (MW)
    total_cap = np.sum(p_max)
    demand_mw = rng.uniform(0.4 * total_cap, 0.8 * total_cap, n_periods)

    n_u = n_generators * n_periods  # binary on/off
    n_p = n_generators * n_periods  # continuous power
    n_total = n_u + n_p

    def u_idx(g, t): return g * n_periods + t
    def p_idx(g, t): return n_u + g * n_periods + t

    # Objective: minimize total generation + startup cost
    c_obj = np.zeros(n_total)
    for g in range(n_generators):
        for t in range(n_periods):
            c_obj[p_idx(g, t)] = cost_a[g]           # variable cost
            c_obj[u_idx(g, t)] = cost_b[g] / n_periods  # startup cost approx

    ineq_rows = []
    ineq_rhs = []
    eq_rows = []
    eq_rhs = []

    for t in range(n_periods):
        # Demand balance: sum_g p[g,t] = demand[t]
        row = np.zeros(n_total)
        for g in range(n_generators):
            row[p_idx(g, t)] = 1.0
        eq_rows.append(row)
        eq_rhs.append(float(demand_mw[t]))

        for g in range(n_generators):
            # p[g,t] >= p_min[g] * u[g,t]
            row = np.zeros(n_total)
            row[p_idx(g, t)] = -1.0
            row[u_idx(g, t)] = float(p_min[g])
            ineq_rows.append(row)
            ineq_rhs.append(0.0)

            # p[g,t] <= p_max[g] * u[g,t]
            row2 = np.zeros(n_total)
            row2[p_idx(g, t)] = 1.0
            row2[u_idx(g, t)] = -float(p_max[g])
            ineq_rows.append(row2)
            ineq_rhs.append(0.0)

    lb = np.zeros(n_total)
    ub = np.concatenate([np.ones(n_u), np.array([p_max[g] for g in range(n_generators) for _ in range(n_periods)])])

    problem = OptimizationProblem(
        c=c_obj,
        A_ub=np.array(ineq_rows),
        b_ub=np.array(ineq_rhs),
        A_eq=np.array(eq_rows),
        b_eq=np.array(eq_rhs),
        lb=lb, ub=ub,
        integer_vars=list(range(n_u)),
        name=f"PowerDispatch_{n_generators}gen_{n_periods}periods",
    )

    return {
        "problem": problem,
        "description": (
            f"Power Economic Dispatch: {n_generators} generators, {n_periods} periods. "
            f"Unit commitment + dispatch. {n_total} vars ({n_u} binary)."
        ),
    }


# ============================================================
# 5. SUPPLY CHAIN / TRANSPORTATION (MILP)
# ============================================================

def supply_chain_milp(n_sources: int = 4, n_depots: int = 6, n_customers: int = 10) -> Dict[str, Any]:
    """
    Petroleum Product Transportation Network.
    
    Multi-product, multi-echelon distribution network.
    Minimize transportation cost from refineries -> depots -> customers.
    """
    rng = np.random.default_rng(55)

    n_products_sc = 3  # Gasoline, Diesel, Kerosene
    supply = rng.uniform(500, 2000, (n_sources, n_products_sc))
    demand = rng.uniform(50, 300, (n_customers, n_products_sc))

    # Transportation costs (INR/kL/km * distance proxy)
    cost_sd = rng.uniform(5, 25, (n_sources, n_depots))
    cost_dc = rng.uniform(3, 15, (n_depots, n_customers))

    # Depot capacity
    depot_cap = rng.uniform(800, 2000, n_depots)

    # Variables: flow[s,d,k] source->depot, flow2[d,c,k] depot->customer
    n_f1 = n_sources * n_depots * n_products_sc
    n_f2 = n_depots * n_customers * n_products_sc
    n_total = n_f1 + n_f2

    def f1(s, d, k): return (s * n_depots + d) * n_products_sc + k
    def f2(d, c, k): return n_f1 + (d * n_customers + c) * n_products_sc + k

    c_obj = np.zeros(n_total)
    for s in range(n_sources):
        for d in range(n_depots):
            for k in range(n_products_sc):
                c_obj[f1(s, d, k)] = cost_sd[s, d]
    for d in range(n_depots):
        for c in range(n_customers):
            for k in range(n_products_sc):
                c_obj[f2(d, c, k)] = cost_dc[d, c]

    ineq_rows, ineq_rhs = [], []
    eq_rows, eq_rhs = [], []

    # Supply limits
    for s in range(n_sources):
        for k in range(n_products_sc):
            row = np.zeros(n_total)
            for d in range(n_depots):
                row[f1(s, d, k)] = 1.0
            ineq_rows.append(row)
            ineq_rhs.append(float(supply[s, k]))

    # Demand satisfaction
    for c in range(n_customers):
        for k in range(n_products_sc):
            row = np.zeros(n_total)
            for d in range(n_depots):
                row[f2(d, c, k)] = 1.0
            ineq_rows.append(row)
            ineq_rhs.append(float(-demand[c, k]))  # >= demand means -<= -demand
            # Fix: >= is -flow <= -demand
            ineq_rows[-1] = -ineq_rows[-1]

    # Depot flow balance (in = out per product)
    for d in range(n_depots):
        for k in range(n_products_sc):
            row = np.zeros(n_total)
            for s in range(n_sources):
                row[f1(s, d, k)] = 1.0
            for c in range(n_customers):
                row[f2(d, c, k)] = -1.0
            eq_rows.append(row)
            eq_rhs.append(0.0)

    # Depot capacity
    for d in range(n_depots):
        row = np.zeros(n_total)
        for s in range(n_sources):
            for k in range(n_products_sc):
                row[f1(s, d, k)] = 1.0
        ineq_rows.append(row)
        ineq_rhs.append(float(depot_cap[d]))

    lb = np.zeros(n_total)
    ub = np.full(n_total, max(supply.max(), demand.max()) * 10)

    problem = OptimizationProblem(
        c=c_obj,
        A_ub=np.array(ineq_rows) if ineq_rows else None,
        b_ub=np.array(ineq_rhs) if ineq_rhs else None,
        A_eq=np.array(eq_rows) if eq_rows else None,
        b_eq=np.array(eq_rhs) if eq_rhs else None,
        lb=lb, ub=ub,
        name=f"PetroChem_SupplyChain_{n_sources}x{n_depots}x{n_customers}",
    )

    return {
        "problem": problem,
        "description": (
            f"Petroleum Supply Chain: {n_sources} refineries → {n_depots} depots → "
            f"{n_customers} customers, {n_products_sc} products. "
            f"{n_total} vars (LP), {len(ineq_rows)} ineq, {len(eq_rows)} eq."
        ),
    }


def get_all_demos() -> Dict[str, Dict]:
    """Return all demo problems."""
    return {
        "crude_blending": crude_blending_milp(n_crudes=6, n_products=4),
        "cdu_scheduling": cdu_scheduling_milp(n_tanks=5, n_periods=4),
        "refinery_lp_small": refinery_production_lp("small"),
        "refinery_lp_medium": refinery_production_lp("medium"),
        "power_dispatch": power_dispatch_milp(n_generators=8, n_periods=6),
        "supply_chain": supply_chain_milp(n_sources=4, n_depots=5, n_customers=8),
    }
