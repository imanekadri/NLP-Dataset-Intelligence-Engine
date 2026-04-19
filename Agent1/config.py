import os
from dataclasses import dataclass

@dataclass
class Agent1Config:
    """Configuration pour l'Agent 1 - Data Ingestion"""
    
     
    DATASETS_DIR = os.path.join(os.path.dirname(__file__), "../data")
    
    # output folders 
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
    EXTRACTED_DIR = os.path.join(OUTPUT_DIR, "extracted")
    
    TRACE_CSV = os.path.join(OUTPUT_DIR, "trace_index.csv")
    REPORT_JSON = os.path.join(OUTPUT_DIR, "report.json")
    DOCUMENTS_INFO_JSON = os.path.join(OUTPUT_DIR, "documents_info.json")

    
    SUPPORTED_FORMATS = ['.txt', '.csv', '.json', '.xml', '.pdf', '.docx', '.log', '.py', '.js']
    ENCODINGS_TO_TRY = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']

config = Agent1Config()