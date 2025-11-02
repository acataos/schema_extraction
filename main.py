import argparse
import pypdf
import json
from extractor import Extractor

def run_extraction(json_data):
    extractor = Extractor()
    
    results = []
    for item in json_data[:1]:
        pdf_file = "./data/files/" + item['pdf_path']
        reader = pypdf.PdfReader(pdf_file)
        # assume pdf only has one page
        pdf_text = reader.pages[0].extract_text()
        
        results.append(extractor.extract(
            item["label"],
            item["extraction_schema"],
            pdf_text
        ))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Extract fields from PDFs based on a JSON schema.")
    parser.add_argument("json_file", help="Path to the JSON file containing extraction instructions.")
    args = parser.parse_args()

    # read json file
    with open(args.json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    results = run_extraction(json_data)




