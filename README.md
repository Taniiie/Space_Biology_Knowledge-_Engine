# Space Biology Knowledge Engine

A comprehensive tool for analyzing NASA bioscience publications using NLP and knowledge graphs.

## Features

- **Data Collection**: Extract text from PDF publications, focusing on key sections (Intro, Results, Conclusion).
- **Preprocessing & Summarization**: Summarize publications, extract keywords and entities, compute sentence embeddings.
- **Knowledge Graph**: Build a graph linking experiments, organisms, findings, and mission relevance.
- **Dashboard**: Interactive web app with semantic search, topic filters, summaries, and visualizations.
- **Extra Features**: Knowledge gaps analysis, trends over time, mission-specific insights.

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Place NASA bioscience PDFs in the `data/` directory.

3. Run the pipeline:
   ```
   python run.py
   ```

4. Start the dashboard:
   ```
   streamlit run app.py
   ```

## Usage

- **Search**: Enter queries like "plant growth in microgravity" for semantic search.
- **Filter**: Select topics such as radiation, muscle atrophy, etc.
- **Visualize**: View topic distributions, trends, and knowledge graphs.
- **Insights**: Explore mission-specific knowledge and gaps.

## Technologies

- PDF Extraction: pdfplumber
- NLP: transformers, spaCy, sentence-transformers
- Embeddings & Search: FAISS
- Graph: NetworkX
- Dashboard: Streamlit, Plotly