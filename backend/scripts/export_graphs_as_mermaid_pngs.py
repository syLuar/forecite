import os
import sys
import importlib
from pathlib import Path

# Set up paths
BASE_DIR = Path(__file__).resolve().parent.parent
GRAPHS_DIR = BASE_DIR / "app" / "graphs"
VIS_DIR = GRAPHS_DIR / "vis"
VIS_DIR.mkdir(parents=True, exist_ok=True)

# List of (module, workflow variable, output filename)
WORKFLOWS = [
    # v2
    ("app.graphs.v2.research_graph", "research_graph", "research_graph_v2.png"),
    ("app.graphs.v2.drafting_graph", "drafting_graph", "drafting_graph_v2.png"),
    ("app.graphs.v2.counterargument_graph", "counterargument_graph", "counterargument_graph_v2.png"),
    ("app.graphs.v3.research_graph", "research_graph", "research_graph_v3.png"),
    ("app.graphs.v3.drafting_graph", "drafting_graph", "drafting_graph_v3.png"),
    ("app.graphs.v3.counterargument_graph", "counterargument_graph", "counterargument_graph_v3.png"),
]

def render_mermaid_to_png(graph, output_path):
    # draw_mermaid_png accepts a graph or a mermaid string
    png_bytes = graph.get_graph().draw_mermaid_png()
    with open(output_path, "wb") as f:
        f.write(png_bytes)

def main():
    sys.path.insert(0, str(BASE_DIR))
    for module_name, var_name, out_file in WORKFLOWS:
        try:
            mod = importlib.import_module(module_name)
            graph = getattr(mod, var_name)
            out_path = VIS_DIR / out_file
            render_mermaid_to_png(graph, out_path)
            print(f"Exported {module_name}.{var_name} to {out_path}")
        except Exception as e:
            print(f"Failed to export {module_name}.{var_name}: {e}")

if __name__ == "__main__":
    main()
