"""Willow topology utilities, low-error subgraph selection, and readout checks."""

import time
from typing import Iterable, Optional
import networkx as nx
import cirq


def normalize_coupler_pair(
    q1: cirq.GridQubit, q2: cirq.GridQubit
) -> tuple[cirq.GridQubit, cirq.GridQubit]:
    """Return a canonical sorted pair of GridQubits."""
    return (q1, q2) if q1 <= q2 else (q2, q1)


def lookup_coupler_error(
    q1: cirq.GridQubit,
    q2: cirq.GridQubit,
    cz_pauli_error_rate: dict[tuple[cirq.GridQubit, cirq.GridQubit], tuple[float, float]],
) -> Optional[tuple[float, float]]:
    """Look up a coupler pair in cz_pauli_error_rate regardless of tuple ordering."""
    if (q1, q2) in cz_pauli_error_rate:
        return cz_pauli_error_rate[(q1, q2)]
    if (q2, q1) in cz_pauli_error_rate:
        return cz_pauli_error_rate[(q2, q1)]
    return None


def build_connectivity_graph(
    cz_pauli_error_rate: dict[tuple[cirq.GridQubit, cirq.GridQubit], tuple[float, float]],
    sq_pauli_error_rate: Optional[dict[cirq.GridQubit, tuple[float, float]]] = None,
    zero_errors: Optional[dict[cirq.GridQubit, tuple[float, float]]] = None,
    one_errors: Optional[dict[cirq.GridQubit, tuple[float, float]]] = None,
) -> nx.Graph:
    """Construct an undirected graph of working qubits and CZ couplers.

    Only qubits present in sq_pauli_error_rate (when provided) and couplers present
    in cz_pauli_error_rate are included.
    """
    g = nx.Graph()
    valid_qubits = set(sq_pauli_error_rate.keys()) if sq_pauli_error_rate is not None else None

    for (q1, q2), (cz_err, _) in cz_pauli_error_rate.items():
        if valid_qubits is not None and (q1 not in valid_qubits or q2 not in valid_qubits):
            continue
        sq1 = sq_pauli_error_rate[q1][0] if sq_pauli_error_rate and q1 in sq_pauli_error_rate else 0.0
        sq2 = sq_pauli_error_rate[q2][0] if sq_pauli_error_rate and q2 in sq_pauli_error_rate else 0.0
        ro1 = (
            0.5 * (zero_errors.get(q1, (0.0, 0.0))[0] + one_errors.get(q1, (0.0, 0.0))[0])
            if zero_errors and one_errors
            else 0.0
        )
        ro2 = (
            0.5 * (zero_errors.get(q2, (0.0, 0.0))[0] + one_errors.get(q2, (0.0, 0.0))[0])
            if zero_errors and one_errors
            else 0.0
        )
        g.add_node(q1, sq_error=sq1, readout_error=ro1)
        g.add_node(q2, sq_error=sq2, readout_error=ro2)
        weight = max(float(cz_err) + 2.0 * (sq1 + sq2) + 0.1 * (ro1 + ro2), 1e-6)
        g.add_edge(q1, q2, weight=weight, cz_error=float(cz_err))

    if valid_qubits is not None and len(g) == 0:
        for q in valid_qubits:
            sq_err = sq_pauli_error_rate[q][0]
            g.add_node(q, sq_error=sq_err, readout_error=0.0)

    return g


def find_best_linear_chain(
    graph: nx.Graph,
    length: int,
    time_budget_s: float = 1.5,
) -> list[cirq.GridQubit]:
    """Find a low-error simple path of specified length in the connectivity graph.

    Guarantees that every consecutive pair (path[i], path[i+1]) is an edge in graph.
    """
    if length <= 0:
        return []
    if length == 1:
        if len(graph) == 0:
            raise ValueError("Connectivity graph is empty")
        best_single = min(
            graph.nodes(),
            key=lambda n: graph.nodes[n].get("sq_error", 0.0) + graph.nodes[n].get("readout_error", 0.0),
        )
        return [best_single]
    if length > len(graph):
        raise ValueError(f"Requested path length {length} exceeds working graph size {len(graph)}")

    best_path: Optional[list[cirq.GridQubit]] = None
    best_weight = float("inf")
    deadline = time.perf_counter() + time_budget_s

    # Pre-sort neighbors of every node by edge weight
    sorted_neighbors: dict[cirq.GridQubit, list[tuple[cirq.GridQubit, float]]] = {}
    for u in graph.nodes():
        nbrs = [(v, graph[u][v].get("weight", 1.0)) for v in graph.neighbors(u)]
        nbrs.sort(key=lambda item: item[1])
        sorted_neighbors[u] = nbrs

    # Global minimum edge weight for admissible lower-bound pruning
    all_weights = [d.get("weight", 1.0) for _, _, d in graph.edges(data=True)]
    min_edge_w = min(all_weights) if all_weights else 0.0

    # Rank candidate starting nodes by mean weight of their best two incident edges
    candidate_starts = [n for n in graph.nodes() if len(sorted_neighbors[n]) > 0]
    candidate_starts.sort(
        key=lambda n: sum(w for _, w in sorted_neighbors[n][:2]) / min(len(sorted_neighbors[n]), 2)
    )

    for start_node in candidate_starts[:40]:
        if time.perf_counter() > deadline and best_path is not None:
            break
        visits = 0
        max_visits_per_start = 8000
        visited = {start_node}
        current_path = [start_node]

        def dfs(current_weight: float):
            nonlocal best_path, best_weight, visits
            visits += 1
            if visits > max_visits_per_start:
                return
            rem = length - len(current_path)
            if rem == 0:
                if current_weight < best_weight:
                    best_weight = current_weight
                    best_path = list(current_path)
                return

            if current_weight + rem * min_edge_w >= best_weight:
                return

            u = current_path[-1]
            for v, w in sorted_neighbors[u]:
                if v not in visited:
                    visited.add(v)
                    current_path.append(v)
                    dfs(current_weight + w)
                    current_path.pop()
                    visited.remove(v)
                    if visits > max_visits_per_start:
                        return

        dfs(0.0)

    if best_path is None:
        raise ValueError(f"Could not find a simple path of length {length} on the working couplers")

    return best_path


def find_best_connected_subgraph(
    graph: nx.Graph,
    num_qubits: int,
) -> list[cirq.GridQubit]:
    """Select a low-error connected subgraph of size num_qubits via Prim-like expansion."""
    try:
        return find_best_linear_chain(graph, num_qubits, time_budget_s=0.8)
    except ValueError:
        pass

    best_nodes: Optional[list[cirq.GridQubit]] = None
    best_cost = float("inf")

    for comp in nx.connected_components(graph):
        if len(comp) < num_qubits:
            continue
        for seed in list(comp)[:15]:
            selected = [seed]
            selected_set = {seed}
            total_cost = 0.0
            while len(selected) < num_qubits:
                best_edge = None
                best_w = float("inf")
                for u in selected:
                    for v in graph.neighbors(u):
                        if v not in selected_set:
                            w = graph[u][v].get("weight", 1.0)
                            if w < best_w:
                                best_w = w
                                best_edge = v
                if best_edge is None:
                    break
                selected.append(best_edge)
                selected_set.add(best_edge)
                total_cost += best_w
            if len(selected) == num_qubits and total_cost < best_cost:
                best_cost = total_cost
                best_nodes = selected

    if best_nodes is None:
        raise ValueError(f"No connected component of size >= {num_qubits} exists on device")
    return best_nodes


def check_readout_line_constraint(
    qubits_measured: Iterable[cirq.GridQubit],
    all_active_qubits: Iterable[cirq.GridQubit],
) -> bool:
    """Validate that mid-circuit measurements do not dephase active unmeasured qubits.

    On Willow, qubits sharing a diagonal line (row + col = const) share a readout line.
    Measuring some but not all qubits on that readout line dephases unmeasured qubits.
    """
    measured_set = set(qubits_measured)
    active_set = set(all_active_qubits)

    diagonal_groups: dict[int, list[cirq.GridQubit]] = {}
    for q in active_set:
        diag = q.row + q.col
        diagonal_groups.setdefault(diag, []).append(q)

    for group in diagonal_groups.values():
        group_measured = [q for q in group if q in measured_set]
        if 0 < len(group_measured) < len(group):
            return False
    return True
