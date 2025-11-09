# prompt_graph_iac_enhanced.py
# Enhanced Tkinter demo: Prompt Graphs on an IaC toy graph
# - Improved UI with tabbed interface
# - Real-time graph updates with smooth visualization
# - Better ranking display with progress bars
# - Interactive node inspection and prompt editing
# - Fixed cosine similarity for varying dimensions

import tkinter as tk
from tkinter import ttk, messagebox
import networkx as nx
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import json

# ===========================
# Utilities & Math
# ===========================
RNG = np.random.default_rng(7)


def unit(x):
    """Normalize vector to unit length."""
    n = np.linalg.norm(x) + 1e-9
    return x / n


def cosine(a, b):
    """Cosine similarity with dimension tolerance."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    la, lb = a.shape[0], b.shape[0]
    if la != lb:
        if la < lb:
            a = np.pad(a, (0, lb - la))
        else:
            b = np.pad(b, (0, la - lb))
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


def pretty_rank(scores):
    """Sort scores descending."""
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


# ===========================
# IaC Domain Data
# ===========================
RESOURCES = [
    ("vpc", {"kind": "network", "region": "us-east-1"}),
    ("subnet", {"kind": "network", "region": "us-east-1"}),
    ("igw", {"kind": "network", "region": "us-east-1"}),
    ("sg_web", {"kind": "security", "purpose": "web"}),
    ("ec2_web", {"kind": "compute", "purpose": "web"}),
    ("rds_db", {"kind": "database", "purpose": "data"}),
    ("s3_logs", {"kind": "storage", "purpose": "logging"}),
]

EDGES = [
    ("vpc", "subnet"),
    ("vpc", "igw"),
    ("subnet", "ec2_web"),
    ("ec2_web", "sg_web"),
    ("ec2_web", "s3_logs"),
    ("ec2_web", "rds_db"),
]

KIND2VEC = {
    "network": np.array([1, 0, 0, 0], dtype=float),
    "security": np.array([0, 1, 0, 0], dtype=float),
    "compute": np.array([0, 0, 1, 0], dtype=float),
    "storage": np.array([0, 0, 0, 1], dtype=float),
    "database": unit(np.array([0.5, 0.2, 0.2, 0.6], dtype=float)),
}

PROMPTS = {
    "Pt1": {
        "desc": "Security posture / hardening",
        "color": "#d8b5ff",
        "init": lambda dim: unit(np.array([0.2, 0.9, 0.1, 0.1]) + RNG.normal(0, 0.03, dim))
    },
    "Pt2": {
        "desc": "Cost optimization / storage & data",
        "color": "#c9b5ff",
        "init": lambda dim: unit(np.array([0.1, 0.1, 0.2, 0.9]) + RNG.normal(0, 0.03, dim))
    },
    "Pt_q": {
        "desc": "Query: What helps a public webapp?",
        "color": "#bf7fff",
        "init": lambda dim: unit(np.array([0.6, 0.2, 0.8, 0.1]) + RNG.normal(0, 0.03, dim))
    },
}

KIND_COLORS = {
    "network": "#66c2a5",
    "security": "#fc8d62",
    "compute": "#8da0cb",
    "storage": "#e78ac3",
    "database": "#a6d854",
    "prompt": "#bf7fff",
}


# ===========================
# Graph Building
# ===========================
def build_base_graph(dim=4):
    """Build base IaC graph with node features."""
    G = nx.Graph()
    G.graph["base_dim"] = dim
    for n, attrs in RESOURCES:
        base = KIND2VEC[attrs["kind"]]
        noise = RNG.normal(0, 0.05, size=dim)
        G.add_node(n,
                   kind=attrs["kind"],
                   feat=unit(base + noise),
                   is_prompt=False,
                   metadata=attrs)
    G.add_edges_from(EDGES)
    return G


def init_prompt_vectors(dim=4):
    """Initialize prompt node vectors."""
    return {
        name: PROMPTS[name]["init"](dim)
        for name in PROMPTS
    }


# ===========================
# Insertion Patterns
# ===========================
def apply_cross_links(G, prompts, targets_by_prompt):
    """Add prompt nodes + edges to selected targets."""
    H = G.copy()
    for p_name, vec in prompts.items():
        H.add_node(p_name, kind="prompt", feat=vec.copy(), is_prompt=True)
        for tgt in targets_by_prompt.get(p_name, []):
            if H.has_node(tgt):
                H.add_edge(p_name, tgt)
    return H


def apply_feature_adding(G, prompts, targets_by_prompt):
    """Add prompt vector to node features."""
    H = G.copy()
    for p_name, vec in prompts.items():
        H.add_node(p_name, kind="prompt", feat=vec.copy(), is_prompt=True)
        for tgt in targets_by_prompt.get(p_name, []):
            if H.has_node(tgt):
                H.nodes[tgt]["feat"] = unit(H.nodes[tgt]["feat"] + vec)
    return H


def apply_concatenation(G, prompts, targets_by_prompt):
    """Concatenate prompt vector to node feature."""
    H = G.copy()
    for p_name, vec in prompts.items():
        H.add_node(p_name, kind="prompt", feat=vec.copy(), is_prompt=True)
        for tgt in targets_by_prompt.get(p_name, []):
            if H.has_node(tgt):
                H.nodes[tgt]["feat"] = unit(np.concatenate([H.nodes[tgt]["feat"], vec]))
    return H


def apply_multiplication(G, prompts, targets_by_prompt):
    """Elementwise gating with sigmoid."""
    H = G.copy()
    for p_name, vec in prompts.items():
        H.add_node(p_name, kind="prompt", feat=vec.copy(), is_prompt=True)
        gate = 1 / (1 + np.exp(-vec))
        for tgt in targets_by_prompt.get(p_name, []):
            if H.has_node(tgt):
                a = H.nodes[tgt]["feat"]
                g = gate[: len(a)]
                H.nodes[tgt]["feat"] = unit(a * g)
    return H


# ===========================
# Scoring & Analysis
# ===========================
def relevance_scores(G, query_name="Pt_q"):
    """Compute cosine similarity of all nodes to query."""
    if not G.has_node(query_name):
        return {}
    q = G.nodes[query_name]["feat"]
    scores = {}
    for n, d in G.nodes(data=True):
        if d.get("is_prompt"):
            continue
        fv = d["feat"]
        scores[n] = cosine(q, fv)
    return scores


def graph_stats(G):
    """Compute graph statistics."""
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "resources": sum(1 for _, d in G.nodes(data=True) if not d.get("is_prompt")),
        "prompts": sum(1 for _, d in G.nodes(data=True) if d.get("is_prompt")),
        "avg_degree": sum(dict(G.degree()).values()) / max(G.number_of_nodes(), 1),
    }


# ===========================
# Enhanced Tkinter App
# ===========================
class EnhancedPromptGraphApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🌐 Prompt Graph IaC Demo (Enhanced)")
        self.geometry("1400x800")
        self.configure(bg="#f5f5f5")

        # State
        self.base_dim = 4
        self.G0 = build_base_graph(self.base_dim)
        self.prompts = init_prompt_vectors(self.base_dim)
        self.pattern = tk.StringVar(value="Cross Links")
        self.selected_node = tk.StringVar(value="")
        self.auto_refresh = tk.BooleanVar(value=True)

        self._build_ui()
        self.redraw()

    def _build_ui(self):
        """Build main UI with tabbed interface."""
        # Header
        header = ttk.Frame(self)
        header.pack(fill="x", padx=8, pady=8)
        ttk.Label(header, text="Prompt Graph - Infrastructure as Code",
                  font=("Segoe UI", 14, "bold")).pack(side="left")

        # Main container with notebook
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=4, pady=4)

        # Tab 1: Graph Visualization
        tab_graph = ttk.Frame(nb)
        nb.add(tab_graph, text="📊 Graph Visualization")
        self._build_graph_tab(tab_graph)

        # Tab 2: Controls & Patterns
        tab_controls = ttk.Frame(nb)
        nb.add(tab_controls, text="⚙️ Patterns & Targets")
        self._build_controls_tab(tab_controls)

        # Tab 3: Analysis & Ranking
        tab_analysis = ttk.Frame(nb)
        nb.add(tab_analysis, text="📈 Relevance Analysis")
        self._build_analysis_tab(tab_analysis)

        # Tab 4: Node Inspector
        tab_inspector = ttk.Frame(nb)
        nb.add(tab_inspector, text="🔍 Node Inspector")
        self._build_inspector_tab(tab_inspector)

    def _build_graph_tab(self, parent):
        """Build graph visualization tab."""
        frame = ttk.Frame(parent, padding=4)
        frame.pack(fill="both", expand=True)

        self.fig = Figure(figsize=(9, 7), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Toolbar
        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(fill="x", padx=4, pady=4)
        ttk.Button(toolbar_frame, text="Refresh Graph", command=self.redraw).pack(side="left", padx=2)
        ttk.Button(toolbar_frame, text="Fit Layout", command=self.redraw).pack(side="left", padx=2)
        ttk.Checkbutton(toolbar_frame, text="Auto-refresh", variable=self.auto_refresh).pack(side="left", padx=2)

    def _build_controls_tab(self, parent):
        """Build controls and patterns tab."""
        left = ttk.Frame(parent, padding=8)
        left.pack(side="left", fill="both", expand=False)

        right = ttk.Frame(parent, padding=8)
        right.pack(side="right", fill="both", expand=True)

        # Patterns selection
        ttk.Label(left, text="Insertion Pattern", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 6))
        for name in ["Cross Links", "Feature Adding", "Concatenation", "Multiplication"]:
            ttk.Radiobutton(left, text=name, value=name, variable=self.pattern,
                            command=self._on_pattern_change).pack(anchor="w", pady=2)

        ttk.Separator(left).pack(fill="x", pady=8)

        # Prompt targets with cleaner layout
        ttk.Label(left, text="Prompt Targeting", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(4, 6))

        self.targets = {}
        for p in ["Pt1", "Pt2", "Pt_q"]:
            lf = ttk.LabelFrame(left, text=p, padding=6)
            lf.pack(fill="x", pady=4)
            self.targets[p] = {}

            for n, _ in RESOURCES:
                v = tk.BooleanVar(value=(p != "Pt_q"))
                if p == "Pt_q" and n in {"ec2_web", "sg_web", "igw"}:
                    v.set(True)
                ttk.Checkbutton(lf, text=n, variable=v, command=self._on_targets_change).pack(anchor="w")
                self.targets[p][n] = v

        # Right side: Graph statistics
        ttk.Label(right, text="Graph Statistics", font=("Segoe UI", 11, "bold")).pack(anchor="nw")
        self.stats_text = tk.Text(right, height=12, width=35, bg="white", font=("Courier", 9))
        self.stats_text.pack(fill="both", expand=False, pady=6)

        # Action buttons
        ttk.Separator(left).pack(fill="x", pady=8)
        ttk.Button(left, text="🔄 Randomize Prompts", command=self._randomize_prompts).pack(fill="x", pady=2)
        ttk.Button(left, text="🔧 Reset All", command=self._reset).pack(fill="x", pady=2)

    def _build_analysis_tab(self, parent):
        """Build analysis and ranking tab."""
        frame = ttk.Frame(parent, padding=8)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Resource Relevance to Pt_q (Query Prompt)",
                  font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 8))

        # Ranking with progress bars
        scroll_frame = ttk.Frame(frame)
        scroll_frame.pack(fill="both", expand=True)

        canvas = tk.Canvas(scroll_frame, bg="white", highlightthickness=0)
        scrollbar = ttk.Scrollbar(scroll_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, padding=4)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.ranking_frame = scrollable_frame

    def _build_inspector_tab(self, parent):
        """Build node inspector tab."""
        left = ttk.Frame(parent, padding=8)
        left.pack(side="left", fill="y")

        right = ttk.Frame(parent, padding=8)
        right.pack(side="right", fill="both", expand=True)

        # Node selector
        ttk.Label(left, text="Select Node", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 4))
        self.node_combo = ttk.Combobox(left, textvariable=self.selected_node, state="readonly", width=20)
        self.node_combo.pack(fill="x", pady=(0, 6))
        self.node_combo.bind("<<ComboboxSelected>>", lambda e: self._update_inspector())

        ttk.Button(left, text="Inspect", command=self._update_inspector).pack(fill="x", pady=4)

        # Inspector details
        ttk.Label(right, text="Node Details", font=("Segoe UI", 11, "bold")).pack(anchor="nw")
        self.inspector_text = tk.Text(right, height=20, width=50, bg="white", font=("Courier", 9))
        self.inspector_text.pack(fill="both", expand=True, pady=6)

    def _on_pattern_change(self):
        if self.auto_refresh.get():
            self.redraw()

    def _on_targets_change(self):
        if self.auto_refresh.get():
            self.redraw()

    def _reset(self):
        self.G0 = build_base_graph(self.base_dim)
        self.prompts = init_prompt_vectors(self.base_dim)
        self.redraw()

    def _randomize_prompts(self):
        self.prompts = init_prompt_vectors(self.base_dim)
        self.redraw()

    def _update_inspector(self):
        """Update node inspector details."""
        H = self.build_prompted()
        node = self.selected_node.get()

        self.inspector_text.delete("1.0", tk.END)

        if not node or node not in H:
            self.inspector_text.insert(tk.END, "No node selected.")
            return

        d = H.nodes[node]
        text = f"Node: {node}\n"
        text += f"{'=' * 40}\n\n"
        text += f"Kind: {d.get('kind', 'N/A')}\n"
        text += f"Is Prompt: {d.get('is_prompt', False)}\n"
        text += f"Feature Dim: {len(d.get('feat', []))}\n"
        text += f"Feature: {np.array2string(d.get('feat', []), precision=3, max_line_width=40)}\n"

        if not d.get('is_prompt'):
            scores = relevance_scores(H, "Pt_q")
            if node in scores:
                text += f"\nRelevance to Pt_q: {scores[node]:.4f}\n"

        neighbors = list(H.neighbors(node))
        text += f"\nNeighbors ({len(neighbors)}): {', '.join(neighbors)}\n"

        text += f"\nDegree: {H.degree(node)}\n"

        self.inspector_text.insert(tk.END, text)

    def collect_targets(self):
        out = {}
        for p, dd in self.targets.items():
            chosen = [n for n, v in dd.items() if v.get()]
            out[p] = chosen
        return out

    def build_prompted(self):
        """Build prompted graph based on selected pattern."""
        targets = self.collect_targets()
        pat = self.pattern.get()

        if pat == "Cross Links":
            H = apply_cross_links(self.G0, self.prompts, targets)
        elif pat == "Feature Adding":
            H = apply_feature_adding(self.G0, self.prompts, targets)
        elif pat == "Concatenation":
            H = apply_concatenation(self.G0, self.prompts, targets)
        elif pat == "Multiplication":
            H = apply_multiplication(self.G0, self.prompts, targets)
        else:
            H = self.G0.copy()

        # Ensure Pt_q exists
        if not H.has_node("Pt_q"):
            H.add_node("Pt_q", kind="prompt", feat=self.prompts["Pt_q"], is_prompt=True)

        return H

    def redraw(self):
        """Redraw visualization and update all tabs."""
        H = self.build_prompted()

        # Draw graph
        self.ax.clear()
        pos = nx.spring_layout(H, seed=42, k=1.2, iterations=50)

        res_nodes = [n for n, d in H.nodes(data=True) if not d.get("is_prompt")]
        prm_nodes = [n for n, d in H.nodes(data=True) if d.get("is_prompt")]

        # Draw nodes with color coding
        node_colors_res = [KIND_COLORS.get(H.nodes[n].get("kind", "network"), "#999999") for n in res_nodes]
        nx.draw_networkx_nodes(H, pos, nodelist=res_nodes, node_color=node_colors_res,
                               node_size=800, ax=self.ax, edgecolors="black", linewidths=1.5)
        nx.draw_networkx_nodes(H, pos, nodelist=prm_nodes, node_color=KIND_COLORS["prompt"],
                               node_shape="s", node_size=900, ax=self.ax, edgecolors="black", linewidths=2)

        nx.draw_networkx_labels(H, pos, font_size=8, font_weight="bold", ax=self.ax)
        nx.draw_networkx_edges(H, pos, width=1.5, alpha=0.6, ax=self.ax, edge_color="#555555")

        self.ax.set_title(f"Pattern: {self.pattern.get()}", fontsize=12, fontweight="bold", pad=10)
        self.ax.axis("off")
        self.fig.tight_layout()
        self.canvas.draw()

        # Update stats
        stats = graph_stats(H)
        stats_text = f"""Graph Stats:
{'=' * 30}
Nodes (total):     {stats['nodes']}
Resources:         {stats['resources']}
Prompts:           {stats['prompts']}
Edges:             {stats['edges']}
Avg Degree:        {stats['avg_degree']:.2f}
Pattern:           {self.pattern.get()}
"""
        self.stats_text.delete("1.0", tk.END)
        self.stats_text.insert(tk.END, stats_text)

        # Update ranking with progress bars
        scores = relevance_scores(H, "Pt_q")
        ranks = pretty_rank(scores)

        for widget in self.ranking_frame.winfo_children():
            widget.destroy()

        max_score = max(scores.values()) if scores else 1.0

        for node, score in ranks:
            row = ttk.Frame(self.ranking_frame)
            row.pack(fill="x", pady=4)

            ttk.Label(row, text=f"{node:12s}", width=12).pack(side="left", padx=4)

            bar_frame = ttk.Frame(row, height=20)
            bar_frame.pack(side="left", fill="x", expand=True, padx=4)

            canvas = tk.Canvas(bar_frame, height=20, bg="white", highlightthickness=0)
            canvas.pack(fill="x", expand=True)

            bar_width = (score / max_score) * 200
            canvas.create_rectangle(0, 0, bar_width, 20, fill="#2ecc71", outline="#27ae60")

            ttk.Label(row, text=f"{score:.4f}", width=8).pack(side="right", padx=4)

        # Update node combo
        all_nodes = sorted(H.nodes())
        self.node_combo['values'] = all_nodes
        if self.selected_node.get() not in all_nodes:
            self.selected_node.set(all_nodes[0] if all_nodes else "")


# ===========================
# Main
# ===========================
if __name__ == "__main__":
    try:
        app = EnhancedPromptGraphApp()
        app.mainloop()
    except Exception as e:
        import traceback

        messagebox.showerror("Error", f"{e}\n\n{traceback.format_exc()}")
