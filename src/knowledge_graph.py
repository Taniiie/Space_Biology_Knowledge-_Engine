import json
import networkx as nx
import pickle
from collections import defaultdict

def build_knowledge_graph(input_file='processed_publications.json', output_file='knowledge_graph.gpickle'):
    """
    Build a knowledge graph from processed publications.
    Nodes: keywords, edges: co-occurrence in papers.
    """
    with open(input_file, 'r') as f:
        pubs = json.load(f)

    G = nx.Graph()
    keyword_papers = defaultdict(list)

    for pub in pubs:
        keywords = pub['keywords']
        for kw in keywords:
            keyword_papers[kw].append(pub['title'])

    # Add nodes
    for kw, papers in keyword_papers.items():
        G.add_node(kw, papers=papers, count=len(papers))

    # Add edges based on co-occurrence
    for pub in pubs:
        keywords = pub['keywords']
        for i in range(len(keywords)):
            for j in range(i+1, len(keywords)):
                kw1, kw2 = keywords[i], keywords[j]
                if G.has_edge(kw1, kw2):
                    G[kw1][kw2]['weight'] += 1
                else:
                    G.add_edge(kw1, kw2, weight=1)

    with open(output_file, 'wb') as f:
        pickle.dump(G, f)
    return G

def get_mission_relevance(keywords):
    """
    Simple heuristic to assign mission relevance.
    """
    relevance_map = {
        'mars': 'Mars exploration',
        'moon': 'Lunar missions',
        'microgravity': 'Space station research',
        'radiation': 'Deep space radiation',
        'plant': 'Life support systems',
        'astronaut': 'Human health in space'
    }
    relevance = []
    for kw in keywords:
        for key, rel in relevance_map.items():
            if key in kw.lower():
                relevance.append(rel)
    return list(set(relevance))

if __name__ == '__main__':
    build_knowledge_graph()