import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import re
from InstructorEmbedding import INSTRUCTOR
import torch
from sklearn.metrics.pairwise import cosine_similarity
import json

def surgical_scrub(text, saint_name):
    # 1. Handle the full name (e.g., "John of the Cross")
    # 2. Handle the name without "Saint" or "St." (e.g., "John")
    # 3. Handle possessives (e.g., "John's")
    
    clean_name = re.sub(r'^(Saint|St\.)\s+', '', saint_name, flags=re.IGNORECASE).strip()
    
    # We create a list of aliases to find
    aliases = [
        re.escape(saint_name),                # Full "Saint John of the Cross"
        re.escape(clean_name),               # "John of the Cross"
        rf"{re.escape(clean_name.split()[0])}" # Just "John" (be careful with short names)
    ]
    
    # Sort by length so we replace the longest match first (preventing partial matches)
    aliases.sort(key=len, reverse=True)
    pattern = r'\b(' + '|'.join(aliases) + r')\b'
    
    # Replace with a neutral placeholder
    return re.sub(pattern, "[THE_SAINT]", text, flags=re.IGNORECASE)

# Apply this only to the specific row's biography
df = pd.read_csv('saints_data_full.csv')
df = df.dropna(subset=['biography']).reset_index(drop=True)
df['biography'] = df.apply(lambda x: surgical_scrub(x['biography'], x['name']), axis=1)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model and move it to the GPU
#model = INSTRUCTOR('hkunlp/instructor-large', device=device)
# The rest of your code stays the same!
#model._text_length = lambda text: len(text)

# 2. Generate Embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

instruction = "Represent the hagiography for clustering by historical era, location, and theological contribution, ignoring common names:"
#instruction = "Represent the hagiography for clustering: focus equally on historical era, geographic location, and the specific manner of the saint's life and death (vocation, miracles, and martyrdom)."

print("Generating embeddings...")
embeddings = model.encode(df['biography'].tolist(), show_progress_bar=True)
#embeddings = model.encode([[instruction, text] for text in df['biography'].tolist()])

# 3. Save raw embeddings (384-dimensional)
# Use this for calculating exact similarity scores later
np.save('saint_embeddings.npy', embeddings)

# 2. Calculate the similarity matrix (Saints x Saints)
# This results in a 1631x1631 matrix of scores between 0 and 1
sim_matrix = cosine_similarity(embeddings)

# 3. For each saint, find the top 10 indices
top_10_list = []
for i in range(len(sim_matrix)):
    # Sort indices by similarity score (descending)
    # [1:11] because index 0 is always the saint itself (similarity = 1.0)
    similar_indices = np.argsort(sim_matrix[i])[::-1][1:11]
    top_10_list.append(similar_indices.tolist())

top_10_data = []
for i, indices in enumerate(top_10_list):
    # Grab the scores for the specific neighbors of saint i
    # We round to 4 decimals to keep the JSON file size smaller
    scores = [round(float(sim_matrix[i][idx]), 4) for idx in indices]
    
    # Bundle them together: [{'idx': 244, 'score': 0.98}, ...]
    combined = [{"idx": int(idx), "score": s} for idx, s in zip(indices, scores)]
    top_10_data.append(combined)

df['similar'] = top_10_data


# 4. Run t-SNE for Visualization
# perplexity: roughly the number of neighbors each point considers (try 5-30)
print("Running t-SNE dimension reduction...")
tsne = TSNE(
    n_components=2, 
    perplexity=30, 
    random_state=42, 
    init='pca', 
    learning_rate='auto'
)
tsne_results = tsne.fit_transform(embeddings)

# 5. Save the 2D coordinates back into a CSV for easy plotting
df['x'] = tsne_results[:, 0]
df['y'] = tsne_results[:, 1]

# 1. Convert DataFrame to a list of dictionaries (one per saint)
data_records = df.to_dict(orient='records')

# 2. Save directly to JSON
with open('saints_embedding_data.json', 'w', encoding='utf-8') as f:
    # Use indent=2 to make it readable for debugging, 
    # or remove it for a smaller file size in production
    json.dump(data_records, f, ensure_ascii=False, indent=2)

print("Checkpoint saved! You now have 'saint_embeddings.npy' and 'saints_embedding_data.json'")
