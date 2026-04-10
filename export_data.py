import json
import pandas as pd
import networkx as nx
from networkx.readwrite import json_graph
import community.community_louvain as louvain

# 1. Load the dataframe (contains names, URLs, x/y coordinates)
df = pd.read_csv('saints_viz_data.csv')

# 2. Load the Graph topology from your JSON file
with open('saints_graph.json', 'r') as f:
    graph_data = json.load(f)

# Reconstruct the NetworkX graph object from the JSON data
G = json_graph.node_link_graph(graph_data)

# 1. Calculate Communities (Clusters)
partition = louvain.best_partition(G) # Assigns a cluster ID (0, 1, 2...) to every node

nodes = []
for i in range(len(df)):
    row = df.iloc[i]
    nodes.append({
        "id": i,
        "name": str(row['name']),
        "url": str(row['url']),
        "group": partition[i], # This is the color category
        "x": float(row['x']) * 10,
        "y": float(row['y']) * 10
    })

# 4. Prepare the final JSON structure for D3.js
final_data = {
    "nodes": nodes,
    "links": [{"source": u, "target": v} for u, v in G.edges()]
}

# 5. Export for the Web UI
with open('saints_ui_data.json', 'w') as f:
    json.dump(final_data, f, indent=4)

print("UI Data ready: 'saints_ui_data.json' created with coordinates and links.")