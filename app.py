import streamlit as st
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import networkx as nx
import pickle
import plotly.graph_objects as go
import pandas as pd
import re
import pdfplumber
from transformers import pipeline
import spacy

# Load data
@st.cache_data
def load_data():
    with open('processed_publications.json', 'r') as f:
        pubs = json.load(f)
    return pubs

@st.cache_resource
def load_models():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index('embeddings.index')
    with open('knowledge_graph.gpickle', 'rb') as f:
        G = pickle.load(f)
    return embedder, index, G

# Load additional models for PDF processing
@st.cache_resource
def load_processing_models():
    nlp = spacy.load('en_core_web_sm')
    summarizer = pipeline('summarization', model='facebook/bart-large-cnn')
    return nlp, summarizer

def summarize_text(text, summarizer):
    if len(text.split()) < 50:
        return text
    try:
        summary = summarizer(text[:1024], max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    except:
        summary = text[:500] + '...'
    return summary

def extract_keywords(text, nlp):
    doc = nlp(text)
    entities = [ent.text.lower() for ent in doc.ents if ent.label_ in ['ORG', 'GPE', 'PERSON', 'MISC', 'WORK_OF_ART']]
    nouns = [chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) <= 3]
    keywords = list(set(entities + nouns))
    return keywords[:20]

pubs = load_data()
embedder, index, G = load_models()
nlp, summarizer = load_processing_models()

st.title('Space Biology Knowledge Engine')

# PDF Upload Section
st.header('📄 Upload and Analyze PDF')
uploaded_file = st.file_uploader("Upload a NASA bioscience PDF", type="pdf")

if uploaded_file is not None:
    with st.spinner('Extracting text from PDF...'):
        with pdfplumber.open(uploaded_file) as pdf:
            text = ''
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + '\n'
    
    st.success('Text extracted successfully!')
    
    with st.spinner('Processing text...'):
        summary = summarize_text(text, summarizer)
        keywords = extract_keywords(text, nlp)
        embedding = embedder.encode(text)
    
    st.subheader('📋 Summary')
    st.write(summary)
    
    st.subheader('🔑 Keywords')
    st.write(', '.join(keywords))
    
    st.subheader('📊 Full Text Excerpt')
    st.write(text[:1000] + '...' if len(text) > 1000 else text)
    
    # Optional: Add to dataset
    if st.button('Add to Knowledge Base'):
        new_pub = {
            'title': uploaded_file.name,
            'full_text': text,
            'summary': summary,
            'keywords': keywords,
            'embedding': embedding.tolist()
        }
        pubs.append(new_pub)
        # Update FAISS index
        index.add(np.array([embedding]))
        # Update knowledge graph (simple add nodes)
        for kw in keywords:
            if kw not in G:
                G.add_node(kw, count=1)
            else:
                G.nodes[kw]['count'] += 1
        # Save updated data
        with open('processed_publications.json', 'w') as f:
            json.dump(pubs, f, indent=4)
        faiss.write_index(index, 'embeddings.index')
        with open('knowledge_graph.gpickle', 'wb') as f:
            pickle.dump(G, f)
        st.success('Added to knowledge base!')
        st.rerun()

# Sidebar for filters
st.sidebar.header('Filters')
topics = ['radiation', 'muscle atrophy', 'bone density', 'plant biology', 'microgravity']
selected_topic = st.sidebar.selectbox('Select Topic', ['All'] + topics)

# Search bar
query = st.text_input('🔍 Semantic Search (e.g., "plant growth in microgravity")')

# Process search
if query:
    q_emb = embedder.encode(query)
    D, I = index.search(np.array([q_emb]), k=10)
    results = [pubs[i] for i in I[0] if D[0][list(I[0]).index(i)] < 1.0]  # Threshold
else:
    results = pubs

# Filter by topic
if selected_topic != 'All':
    results = [p for p in results if selected_topic in ' '.join(p.get('keywords', [])).lower()]

# Display results
st.header(f'📑 Publications ({len(results)} found)')
for pub in results[:10]:  # Limit display
    with st.expander(pub['title']):
        st.write('**Summary:**', pub.get('summary', 'N/A'))
        st.write('**Keywords:**', ', '.join(pub.get('keywords', [])))
        st.write('**Full Text Excerpt:**', pub['full_text'][:500] + '...')

# Visualizations
st.header('📊 Visualizations')

# Topic distribution
topic_counts = {}
for t in topics:
    count = sum(1 for p in pubs if t in ' '.join(p.get('keywords', [])).lower())
    topic_counts[t] = count

st.subheader('Topic Distribution')
st.bar_chart(topic_counts)

# Trends over time (assuming year in title)
years = []
for pub in pubs:
    match = re.search(r'\b(19|20)\d{2}\b', pub['title'])
    if match:
        years.append(int(match.group()))
    else:
        years.append(2020)  # Default

df = pd.DataFrame({'year': years, 'topic': [next((t for t in topics if t in ' '.join(p.get('keywords', [])).lower()), 'other') for p in pubs]})
trend = df.groupby(['year', 'topic']).size().unstack().fillna(0)
st.subheader('Trends Over Time')
st.line_chart(trend)

# Knowledge Graph
st.subheader('🌌 Knowledge Graph (Co-occurrence Network)')
pos = nx.spring_layout(G)
edge_x, edge_y = [], []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

node_x = [pos[node][0] for node in G.nodes()]
node_y = [pos[node][1] for node in G.nodes()]
node_text = [f'{node} ({G.nodes[node].get("count", 0)})' for node in G.nodes()]

fig = go.Figure()
fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color='gray', width=1)))
fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_text, marker=dict(size=10, color='blue')))
st.plotly_chart(fig)

# Knowledge Gaps
st.header('🚀 Knowledge Gaps')
gaps = {t: count for t, count in topic_counts.items() if count < 10}  # Arbitrary threshold
if gaps:
    st.write('Topics with fewer than 10 papers:')
    for t, c in gaps.items():
        st.write(f'- {t}: {c} papers')
else:
    st.write('No significant gaps detected.')

# Mission Insights
st.header('🛰️ Mission Insights')
insights = {
    'Mars': 'Focus on radiation shielding and life support.',
    'Moon': 'Lunar dust and low gravity effects.',
    'ISS': 'Microgravity studies on human physiology.'
}
selected_mission = st.selectbox('Select Mission', list(insights.keys()))
st.write(insights[selected_mission])