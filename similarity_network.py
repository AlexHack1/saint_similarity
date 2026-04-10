import json
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load your embeddings
embeddings = np.load('saint_embeddings.npy')

# 2. Calculate Similarity
sim_matrix = cosine_similarity(embeddings)

# 3. Create the NetworkX object
G = nx.Graph()
K = 3 # Connect each saint to their 3 most similar neighbors

for i in range(len(sim_matrix)):
    G.add_node(i)
    # Get indices of top K (excluding the saint themselves)
    top_k_indices = np.argsort(sim_matrix[i])[-K-1:-1]
    for idx in top_k_indices:
        G.add_edge(i, int(idx), weight=float(sim_matrix[i][idx]))

# 4. Save to Disk (This creates the file your export script is looking for)
from networkx.readwrite import json_graph
graph_json = json_graph.node_link_data(G)

with open('saints_graph.json', 'w') as f:
    json.dump(graph_json, f, indent=4)

print("Successfully generated 'saints_graph.json'")