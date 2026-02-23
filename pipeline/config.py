import os
from dataclasses import dataclass, field
from typing import List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class PipelineConfig:
    # --- Chemins ---
    DATASETS_DIR: str = os.path.join(PROJECT_ROOT, "data")
    OUTPUT_DIR: str = os.path.join(PROJECT_ROOT, "output")
    EXTRACTED_DIR: str = os.path.join(PROJECT_ROOT, "output", "extracted")
    TRACES_DIR: str = os.path.join(PROJECT_ROOT, "output", "traces")
    PROFILES_DIR: str = os.path.join(PROJECT_ROOT, "output", "profiles")
    EXPORTS_DIR: str = os.path.join(PROJECT_ROOT, "output", "exports")
    REPORTS_DIR: str = os.path.join(PROJECT_ROOT, "output", "reports")
    MODEL_DIR: str = os.path.join(PROJECT_ROOT, "models", "all-MiniLM-L6-v2")

    # --- Agent 1 : Ingestion ---
    SUPPORTED_FORMATS: List[str] = field(default_factory=lambda: [
        '.txt', '.csv', '.json', '.xml', '.pdf', '.docx', '.log', '.py', '.js'
    ])
    ENCODINGS_TO_TRY: List[str] = field(default_factory=lambda: [
        'utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16'
    ])

    # --- Agent 2 : Profiler ---
    NUM_CLUSTERS: int = 5
    RANDOM_STATE: int = 42
    NOISE_THRESHOLD: float = 0.30
    LANG_SAMPLE_SIZE: int = 200
    USE_TOPIC_MODELING: bool = True
    MIN_TOPIC_SIZE: int = 10
    EMBEDDING_BATCH_SIZE: int = 64
    # Sampling pour les opérations coûteuses (stats = corpus complet, pas de limite)
    MAX_DOCS_FOR_EMBEDDING: int = 10_000  # sample pour embeddings + clustering
    MAX_DOCS_FOR_TOPICS: int = 5_000     # sample pour BERTopic (sous-set des embeddings)

    # --- Agent 3 : NLP Understanding ---
    SPACY_MODEL: str = "en_core_web_sm"
    SENTIMENT_MODEL: str = "distilbert-base-uncased-finetuned-sst-2-english"
    MAX_TEXT_LENGTH_NLP: int = 512

    # --- Agent 6 : Splits ---
    TRAIN_RATIO: float = 0.8
    VAL_RATIO: float = 0.1
    TEST_RATIO: float = 0.1

    # --- Outils externes (Windows) ---
    TESSERACT_CMD: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    POPPLER_PATH: str = r"C:\Program Files\poppler-25.12.0\Library\bin"

    def ensure_dirs(self):
        for d in [self.OUTPUT_DIR, self.EXTRACTED_DIR, self.TRACES_DIR,
                   self.PROFILES_DIR, self.EXPORTS_DIR, self.REPORTS_DIR]:
            os.makedirs(d, exist_ok=True)
