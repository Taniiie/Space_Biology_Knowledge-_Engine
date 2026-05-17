import os
import pdfplumber
import json

def extract_text_from_pdf(pdf_path):
    """
    Extract full text from a PDF file.
    """
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + '\n'
    return text

def extract_sections(text):
    """
    Extract key sections: Intro, Results, Conclusion.
    This is a simple heuristic-based extraction.
    """
    sections = {'intro': '', 'results': '', 'conclusion': ''}
    lines = text.split('\n')
    current_section = None
    for line in lines:
        line_lower = line.lower().strip()
        if 'introduction' in line_lower or 'intro' in line_lower:
            current_section = 'intro'
        elif 'results' in line_lower or 'findings' in line_lower:
            current_section = 'results'
        elif 'conclusion' in line_lower or 'discussion' in line_lower:
            current_section = 'conclusion'
        elif current_section and line.strip():
            sections[current_section] += line + '\n'
    return sections

def process_pdfs(data_dir, output_file='publications.json'):
    """
    Process all PDFs in data_dir and save extracted data to JSON.
    """
    publications = []
    for file in os.listdir(data_dir):
        if file.endswith('.pdf'):
            path = os.path.join(data_dir, file)
            text = extract_text_from_pdf(path)
            sections = extract_sections(text)
            pub = {
                'title': file,
                'full_text': text,
                'sections': sections
            }
            publications.append(pub)
    with open(output_file, 'w') as f:
        json.dump(publications, f, indent=4)
    return publications

if __name__ == '__main__':
    process_pdfs('../data')