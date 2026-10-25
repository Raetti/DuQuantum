"""Generate clean, publication-quality Willow105 grid heatmap visualization for presentation slides.

Incorporating user feedback:
1. End marker positioned above the qubit node; removed 'q...' from both markers (now clean 'Start' and 'End').
2. Qubit row/col coordinates made clear with readable grid axes.
3. Legend redesigned: item key placed vertically on the left, horizontal colorbar placed cleanly on the right.
4. Error color scale changed from purple to high-contrast red sequential palette (light pink -> deep crimson/burgundy).
   The selected chain is highlighted in electric cyan/blue for crisp contrast against the red couplers.
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import cirq
from cirq_google.engine import virtual_engine_factory as v

from src.willow_utils import build_connectivity_graph, find_best_linear_chain
from src.noise_models import (
    generate_two_qubit_pauli_error_rates,
    generate_qubit_error_rates,
)


def get_willow_pink_device():
    """Load official 105-qubit Willow Pink device specification."""
    return v.create_device_from_processor_id("willow_pink")


def plot_willow_heatmap(
    chain_length: int = 12,
    seed: int = 0,
    output_png: str = "assets/willow105_heatmap.png",
    output_svg: str = "assets/willow105_heatmap.svg",
):
    # Set up deterministic RNG
    rng = np.random.default_rng(seed)
    device = get_willow_pink_device()
    all_qubits = sorted(list(device.metadata.qubit_set))
    all_pairs = sorted(list(device.metadata.qubit_pairs))

    # Sample representative dropouts (~2-3%)
    dropout_mask = rng.random(len(all_qubits)) < 0.025
    dropouts = set(np.array(all_qubits)[dropout_mask])

    # Sample coupler error rates and dropouts
    cz_err_dict = generate_two_qubit_pauli_error_rates(
        median=2.5e-3,
        spread=0.5,
        dropout_rate=0.015,
        rng=rng,
        single_qubit_dropouts=dropouts,
    )
    sq_err_dict = generate_qubit_error_rates(
        median=4e-4,
        spread=0.5,
        dropouts=dropouts,
        uncertainty=2e-4,
        rng=rng,
    )

    # Build connectivity graph and select optimal chain
    graph = build_connectivity_graph(cz_err_dict, sq_err_dict)
    chain = find_best_linear_chain(graph, chain_length)
    chain_set = set(chain)
    chain_pairs = set()
    for i in range(len(chain) - 1):
        chain_pairs.add((chain[i], chain[i + 1]))
        chain_pairs.add((chain[i + 1], chain[i]))

    # Set up font styling
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    # 10.5 x 8.5 figure for comfortable grid + axis labels + bottom legend
    fig, ax = plt.subplots(figsize=(10.5, 8.5), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    # High-contrast sequential red colormap for CZ coupler error rates:
    # Light coral/rose -> vibrant red -> deep crimson -> dark burgundy
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "cz_crimson_scale",
        ["#FEE2E2", "#FCA5A5", "#EF4444", "#B91C1C", "#450A0A"],
        N=256,
    )
    err_values = [v[0] for v in cz_err_dict.values()]
    vmin = np.percentile(err_values, 5) if err_values else 1e-3
    vmax = np.percentile(err_values, 95) if err_values else 8e-3
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Color for selected chain: Electric Blue / Cyan for strong pop against red/crimson
    chain_color = "#0284C7"
    chain_node_edge = "#0369A1"

    # 1. Draw all couplers
    canonical_pairs = set()
    for q1, q2 in all_pairs:
        p = tuple(sorted([q1, q2]))
        canonical_pairs.add(p)

    for q1, q2 in canonical_pairs:
        cz_entry = cz_err_dict.get((q1, q2)) or cz_err_dict.get((q2, q1))
        
        # Coordinates: x = col, y = -row (row 0 at top)
        x_vals = [q1.col, q2.col]
        y_vals = [-q1.row, -q2.row]

        if cz_entry is None or q1 in dropouts or q2 in dropouts:
            # Defective / dropout coupler: faint dotted gray line
            ax.plot(
                x_vals,
                y_vals,
                color="#CBD5E1",
                linestyle=":",
                linewidth=1.4,
                zorder=1,
            )
        else:
            # Active working coupler
            err_val = cz_entry[0]
            coupler_color = cmap(norm(err_val))
            ax.plot(
                x_vals,
                y_vals,
                color=coupler_color,
                linestyle="-",
                linewidth=2.4,
                alpha=0.9,
                zorder=2,
            )

    # 2. Draw highlighted chain stroke on top
    for i in range(len(chain) - 1):
        q1 = chain[i]
        q2 = chain[i + 1]
        ax.plot(
            [q1.col, q2.col],
            [-q1.row, -q2.row],
            color=chain_color,
            linestyle="-",
            linewidth=4.5,
            solid_capstyle="round",
            zorder=3,
        )

    # 3. Draw qubits (nodes)
    node_radius = 0.28
    for q in all_qubits:
        x = q.col
        y = -q.row

        if q in chain_set:
            circle = plt.Circle(
                (x, y),
                node_radius,
                facecolor=chain_color,
                edgecolor=chain_node_edge,
                linewidth=1.8,
                zorder=5,
            )
            ax.add_patch(circle)
        elif q in dropouts:
            circle = plt.Circle(
                (x, y),
                node_radius * 0.9,
                facecolor="#CBD5E1",
                edgecolor="none",
                zorder=4,
            )
            ax.add_patch(circle)
        else:
            circle = plt.Circle(
                (x, y),
                node_radius,
                facecolor="#FFFFFF",
                edgecolor="#334155",
                linewidth=1.3,
                zorder=4,
            )
            ax.add_patch(circle)

    # 4. Annotate chain endpoints: BOTH cleanly above the qubits, with 'q...' removed
    start_q = chain[0]
    end_q = chain[-1]
    
    ax.text(
        start_q.col,
        -start_q.row + 0.46,
        "Start",
        color="#0369A1",
        fontsize=9.5,
        fontweight="bold",
        ha="center",
        va="bottom",
        zorder=6,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#E0F2FE", edgecolor="#38BDF8", linewidth=0.9),
    )
    ax.text(
        end_q.col,
        -end_q.row + 0.46,
        "End",
        color="#0369A1",
        fontsize=9.5,
        fontweight="bold",
        ha="center",
        va="bottom",
        zorder=6,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#E0F2FE", edgecolor="#38BDF8", linewidth=0.9),
    )

    # 5. Clear Grid Axes for Qubit Coordinates (rows & cols: just plain numbers)
    cols = sorted(list(set(q.col for q in all_qubits)))
    rows = sorted(list(set(q.row for q in all_qubits)))

    ax.set_xticks(cols)
    ax.set_xticklabels([f"{c}" for c in cols], fontsize=8.5, color="#475569")
    
    ax.set_yticks([-r for r in rows])
    ax.set_yticklabels([f"{r}" for r in rows], fontsize=8.5, color="#475569")
    
    # Optional subtle axis title labels
    ax.set_xlabel("Column", fontsize=8.5, color="#64748B", labelpad=6)
    ax.set_ylabel("Row", fontsize=8.5, color="#64748B", labelpad=6)

    ax.set_xlim(min(cols) - 0.9, max(cols) + 0.9)
    # Give space at bottom for the legend
    ax.set_ylim(-max(rows) - 2.8, -min(rows) + 1.2)
    ax.set_aspect("equal")

    # Clean spine aesthetics
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#CBD5E1")
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(colors="#64748B", length=3, direction="out")

    # 6. Redesigned Bottom Layout: Vertical Item Key (Left) + Infidelity Colorbar (Right)
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Working Qubit",
            markerfacecolor="#FFFFFF",
            markeredgecolor="#334155",
            markeredgewidth=1.3,
            markersize=9,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Dropout Qubit (Defective)",
            markerfacecolor="#CBD5E1",
            markeredgecolor="none",
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            color="#CBD5E1",
            linestyle=":",
            linewidth=2.2,
            label="Dead Coupler",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=chain_color,
            linewidth=3.2,
            label=f"Selected Chain (N={chain_length})",
            markerfacecolor=chain_color,
            markeredgecolor=chain_node_edge,
            markersize=8,
        ),
    ]

    # Vertical key on the bottom-left (1 column, stacked vertically)
    legend = ax.legend(
        handles=legend_elements,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.01),
        ncol=1,  # Vertical stack as requested
        frameon=False,
        fontsize=8.5,
        handletextpad=0.7,
        labelspacing=0.5,
    )
    plt.setp(legend.get_texts(), color="#1E293B")

    # Horizontal Colorbar for CZ Pauli Infidelity on the bottom-right
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # Position: [left, bottom, width, height] in normalized figure coordinates
    cbar_ax = fig.add_axes([0.48, 0.055, 0.44, 0.024])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("CZ Pauli Infidelity ($p_{\\mathrm{CZ}}$)", fontsize=8.5, color="#1E293B", labelpad=5, fontweight="medium")
    cbar.ax.tick_params(labelsize=8, colors="#334155", length=3)
    cbar.outline.set_edgecolor("#CBD5E1")
    cbar.outline.set_linewidth(0.8)

    cbar.ax.xaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda val, pos: f"{val*100:.2f}%")
    )

    # Save outputs
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    os.makedirs(os.path.dirname(output_svg), exist_ok=True)

    fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
    fig.savefig(output_svg, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)

    print(f"Heatmap successfully updated:")
    print(f" - Raster (PNG): {output_png}")
    print(f" - Vector (SVG): {output_svg}")


if __name__ == "__main__":
    plot_willow_heatmap(chain_length=12, seed=0)
