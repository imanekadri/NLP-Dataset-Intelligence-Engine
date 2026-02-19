"""Configuration pour l'Agent 1 - Data Ingestion"""

from dataclasses import dataclass
from typing import Dict, List
import os

@dataclass
class Agent1Config:
    """Configuration de l'agent de data ingestion"""
    
    # Chemins
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    DATASETS_DIR: str = os.path.join(BASE_DIR, "datasets", "data")
    
    # Formats supportés
    SUPPORTED_FORMATS: List[str] = None
    
    # Paramètres de traitement
    MAX_FILE_SIZE_MB: int = 100
    CHUNK_SIZE: int = 1000
    REMOVE_DUPLICATES: bool = True
    DETECT_LANGUAGE: bool = True
    
    # Encodages à tester
    ENCODINGS_TO_TRY: List[str] = None
    
    def __post_init__(self):
        self.SUPPORTED_FORMATS = [
            '.txt', '.csv', '.json', '.xml', 
            '.pdf', '.docx', '.log', '.xlsx'
        ]
        self.ENCODINGS_TO_TRY = [
            'utf-8', 'latin-1', 'iso-8859-1', 
            'cp1252', 'utf-16'
        ]

# Instance globale de configuration
config = Agent1Config()