import json
from transformers import pipeline
import spacy
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load models
nlp = spacy.load('en_core_web_sm')
summarizer = pipeline('summarization', model='facebook/bart-large-cnn')
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def summarize_text(text):
    """
    Summarize the given text using BART model.
    """
    if len(text.split()) < 50:
        return text  # Too short to summarize
    try:
        summary = summarizer(text[:1024], max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    except:
        summary = text[:500] + '...'  # Fallback
    return summary

def extract_keywords(text):
    """
    Extract keywords using spaCy entities and noun chunks.
    """
    doc = nlp(text)
    entities = [ent.text.lower() for ent in doc.ents if ent.label_ in ['ORG', 'GPE', 'PERSON', 'MISC', 'WORK_OF_ART']]
    nouns = [chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) <= 3]
    keywords = list(set(entities + nouns))
    return keywords[:20]  # Limit to top 20

def process_publications(input_file='publications.json', output_file='processed_publications.json'):
    """
    Process publications: summarize, extract keywords, compute embeddings.
    """
    with open(input_file, 'r') as f:
        pubs = json.load(f)

    processed = []
    embeddings = []

    for pub in pubs:
        text = pub['full_text']
        summary = summarize_text(text)
        keywords = extract_keywords(text)
        emb = embedder.encode(text)
        embeddings.append(emb)

        pub['summary'] = summary
        pub['keywords'] = keywords
        pub['embedding'] = emb.tolist()
        processed.append(pub)

    # Build FAISS index for semantic search
    embeddings = np.array(embeddings)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, 'embeddings.index')

    with open(output_file, 'w') as f:
        json.dump(processed, f, indent=4)

    return processed

if __name__ == '__main__':
    process_publications()