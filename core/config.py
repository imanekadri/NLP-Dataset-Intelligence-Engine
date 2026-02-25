import os
from dataclasses import dataclass


@dataclass
class GlobalConfig:
    """Global configuration for all agents"""
    
    DATASETS_DIR = os.path.join(os.path.dirname(__file__), "../data")
   
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../output")
    
    ENCODINGS_TO_TRY = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
    SUPPORTED_FORMATS = ['.txt', '.csv', '.json', '.xml', '.pdf', '.docx', '.log', '.py', '.js']


#####################Agent1

@dataclass
class Agent1Config(GlobalConfig):
    """Configuration for Agent1: Text Ingestion"""
    AGENT_NAME = "agent1"
    AGENT_OUTPUT_DIR = os.path.join(GlobalConfig.OUTPUT_DIR, AGENT_NAME)

    EXTRACTED_DIR = os.path.join(AGENT_OUTPUT_DIR, "extracted")
    TRACE_CSV = os.path.join(AGENT_OUTPUT_DIR, "trace_index.csv")
    REPORT_JSON = os.path.join(AGENT_OUTPUT_DIR, "report.json")
    DOCUMENTS_INFO_JSON = os.path.join(AGENT_OUTPUT_DIR, "documents_info.json")


#########################Agent2

class Agent2Config:
    """Configuration for Agent2: Profiling & Clustering"""

    INPUT_FOLDER = Agent1Config.EXTRACTED_DIR
    OUTPUT_DIR = os.path.join(GlobalConfig.OUTPUT_DIR, "agent2")

    DATASET_PROFILE = os.path.join(OUTPUT_DIR, "dataset_profile.json")
    CLUSTER_REPORT = os.path.join(OUTPUT_DIR, "clusters.json")
    TOPIC_REPORT = os.path.join(OUTPUT_DIR, "topics.json")

    # Embeddings
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    DEVICE = "cpu"
    EMBEDDING_BATCH_SIZE = 32

    # Clustering
    NUM_CLUSTERS = 5
    RANDOM_STATE = 42

    # Analysis / Topic modeling
    USE_TOPIC_MODELING = True
    NOISE_THRESHOLD = 0.1
    LANG_SAMPLE_SIZE = 100
    MAX_DOCS_FOR_EMBEDDING = 2000
    MIN_TOPIC_SIZE = 10


global_config = GlobalConfig()
agent1_config = Agent1Config()
agent2_config = Agent2Config()