import argparse
import pathlib
import json
import time
import fitz # Importa PyMuPDF
from extractor import Extractor
import logging

def get_fitz_dict_from_pdf(pdf_path: str) -> dict:
    """Carrega um PDF e extrai o 'dict' da primeira página."""
    with fitz.open(pdf_path) as doc:
        page = doc[0] # Pega a primeira página
        return page.get_text("dict")

def run_extraction(json_data, parent_dir):
    extractor = Extractor()
    
    results = []
    for item in json_data:
        pdf_file = parent_dir / item['pdf_path']
        fitz_dict = get_fitz_dict_from_pdf(pdf_file)
        
        time_start = time.time()
        results.append(extractor.extract(
            item["label"],
            item["extraction_schema"],
            fitz_dict
        ))
        end_time = time.time()
        
        print(f"Extraction for {item['pdf_path']} took {end_time - time_start:.2f} seconds.")

    # Save results to output.json
    json_output_path = "./data/output/output.json"
    pathlib.Path("./data/output/").mkdir(parents=True, exist_ok=True)
    json.dump(results, open(json_output_path, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Extract fields from PDFs based on a JSON schema.")
    parser.add_argument("json_file", help="Path to the JSON file containing extraction instructions.")
    parser.add_argument(
        '-v', '--verbose',
        help="Be verbose",
        action="store_const", dest="loglevel", const=logging.INFO,
    )
    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)
    parent_dir = pathlib.Path(args.json_file).parent

    # read json file
    with open(args.json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    results = run_extraction(json_data, parent_dir)

