#!/usr/bin/env python3
"""
Run the Space Biology Knowledge Engine pipeline.
"""

import os
import sys

from src.data_collection import process_pdfs
from src.preprocessing import process_publications
from src.knowledge_graph import build_knowledge_graph

def main():
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created {data_dir}. Please place PDF files there.")
        return

    print("Step 1: Extracting text from PDFs...")
    process_pdfs(data_dir)

    print("Step 2: Preprocessing and summarizing...")
    process_publications()

    print("Step 3: Building knowledge graph...")
    build_knowledge_graph()

    print("Pipeline complete. Run 'streamlit run app.py' to start the dashboard.")

if __name__ == '__main__':
    main()