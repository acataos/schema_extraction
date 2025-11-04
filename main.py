import argparse
import json
import time
import fitz # Importa PyMuPDF
from extractor import Extractor

def get_fitz_dict_from_pdf(pdf_path: str) -> dict:
    """Carrega um PDF e extrai o 'dict' da primeira página."""
    with fitz.open(pdf_path) as doc:
        page = doc[0] # Pega a primeira página
        return page.get_text("dict")

def run_extraction(json_data):
    extractor = Extractor()
    
    results = []
    for item in json_data[:1]:
        pdf_file = "./data/files/" + item['pdf_path']
        fitz_dict = get_fitz_dict_from_pdf(pdf_file)
        
        results.append(extractor.extract(
            item["label"],
            item["extraction_schema"],
            fitz_dict
        ))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Extract fields from PDFs based on a JSON schema.")
    parser.add_argument("json_file", help="Path to the JSON file containing extraction instructions.")
    args = parser.parse_args()

    # read json file
    with open(args.json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    results = run_extraction(json_data)

