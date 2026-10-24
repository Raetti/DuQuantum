"""Willow topology utilities, graph embedding, and readout line constraint checks."""

from typing import Iterable, Optional
import networkx as nx
import cirq


def build_connectivity_graph(
    cz_pauli_error_rate: dict[tuple[cirq.GridQubit, cirq.GridQubit], tuple[float, float]],
    sq_pauli_error_rate: Optional[dict[cirq.GridQubit, tuple[float, float]]] = None,
) -> nx.Graph:
    """Construct an undirected graph of available qubits and CZ couplers.

    Weights correspond to gate infidelity: -log(1 - error_rate).
    """
    g = nx.Graph()
    if sq_pauli_error_rate:
        for q, (err, _) in sq_pauli_error_rate.items():
            g.add_node(q, error=err)

    for (q1, q2), (cz_err, _) in cz_pauli_error_rate.items():
        # Infidelity weight approximates composition of errors
        weight = max(cz_err, 1e-6)
        g.add_edge(q1, q2, weight=weight)

    return g


def find_best_linear_chain(
    graph: nx.Graph,
    length: int,
) -> list[cirq.GridQubit]:
    """Find a low-error simple path of specified length in the connectivity graph.

    Uses DFS search starting from high-degree, low-error seed nodes.
    """
    if length > len(graph):
        raise ValueError(f"Requested path length {length} exceeds graph size {len(graph)}")

    best_path = None
    best_weight = float("inf")

    # Sort candidate starting nodes by incident edge error
    candidate_starts = sorted(
        graph.nodes(),
        key=lambda n: sum(graph[n][nbr].get("weight", 1.0) for nbr in graph[n]) / max(len(graph[n]), 1),
    )

    def dfs(current_path: list[cirq.GridQubit], current_weight: float):
        nonlocal best_path, best_weight
        if len(current_path) == length:
            if current_weight < best_weight:
                best_weight = current_weight
                best_path = list(current_path)
            return

        u = current_path[-1]
        for v in graph.neighbors(u):
            if v not in current_path:
                w = graph[u][v].get("weight", 1.0)
                if current_weight + w < best_weight:
                    current_path.append(v)
                    dfs(current_path, current_weight + w)
                    current_path.pop()

    for start_node in candidate_starts[:20]:
        dfs([start_node], 0.0)
        if best_path and best_weight < 0.05 * length:
            break

    if best_path is None:
        # Fall back to finding connected components and extracting a path
        for comp in nx.connected_components(graph):
            if len(comp) >= length:
                subg = graph.subgraph(comp)
                tree = nx.minimum_spanning_tree(subg)
                # Walk the tree
                walk = list(nx.dfs_preorder_nodes(tree))
                return walk[:length]
        raise ValueError(f"Could not find connected subgraph of size {length}")

    return best_path


def check_readout_line_constraint(qubits_measured: Iterable[cirq.GridQubit], all_active_qubits: Iterable[cirq.GridQubit]) -> bool:
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

    for diag, group in diagonal_groups.items():
        group_measured = [q for q in group if q in measured_set]
        if 0 < len(group_measured) < len(group):
            # Partial measurement on shared line induces dephasing
            return False
    return True
