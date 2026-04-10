import pandas as pd
import plotly.express as px

# 1. Load the coordinates we saved earlier
df = pd.read_csv('saints_viz_data.csv')

# 2. Create an interactive scatter plot
fig = px.scatter(
    df, 
    x='x', 
    y='y', 
    hover_name='name',
    title='The Digital Hagiography: Saint Similarity Map',
    labels={'x': 't-SNE Dimension 1', 'y': 't-SNE Dimension 2'},
    template='plotly_dark', # Dark mode looks great for graphs
    color_discrete_sequence=['#ffca28'] # Gold dots for saints
)

# 3. Update the layout for a "clean" look
fig.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=1, color='White')))
fig.update_layout(
    font_family="serif",
    title_font_size=24,
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
)

fig.show()