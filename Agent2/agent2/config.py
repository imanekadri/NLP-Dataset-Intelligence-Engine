import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Agent2Config:

    #  INPUT SETTINGS
    INPUT_FOLDER: str=os.path.join(os.path.dirname(os.path.dirname(__file__)),"agent1","output","extracted")

    SUPPORTED_EXTENSIONS: List[str] = field(default_factory=lambda: [".txt", ".csv", ".json", ".xml"])

    #  EMBEDDING SETTINGS
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    MAX_DOCS_FOR_EMBEDDING: int = 5000 # max documents to embed
    BATCH_SIZE: int = 32
   
    #  CLUSTERING SETTINGS
    NUM_CLUSTERS: int = 5
    RANDOM_STATE: int = 42
    USE_AUTO_CLUSTERING: bool = True
   
    #  LANGUAGE DETECTION
    LANG_SAMPLE_SIZE: int = 200   # sample size for language detection (text)
    MULTI_LANGUAGE_THRESHOLD: float = 0.20 # threshold to detect (multi-language dataset)
  
    #  NOISE & QUALITY ANALYSIS
    NOISE_THRESHOLD: float = 0.30 # (%%%% !!!! ###)  n3tbro noisy or spam
    MIN_TEXT_LENGTH: int = 50 #Minimum text length to analyze
    REMOVE_EMPTY_TEXTS: bool = True
    
    #  TOPIC MODELING
    USE_TOPIC_MODELING: bool = True #Enable topic modeling (BERTopic)
    MIN_TOPIC_SIZE: int = 10 # b 10 text bch nchklo topic
    TOP_N_WORDS: int = 10 # top N words for each topic

    #  OUTPUT SETTINGS
    OUTPUT_DIR: str = os.path.join( os.path.dirname(__file__), "output")
    DATASET_PROFILE: str = os.path.join(OUTPUT_DIR,"dataset_profile.json")
    CLUSTER_REPORT: str = os.path.join(OUTPUT_DIR,"clusters.json")
    TOPIC_REPORT: str = os.path.join(OUTPUT_DIR,"topics.json")

 
    #  PERFORMANCE
    ENABLE_PARALLEL: bool = True
    NUM_WORKERS: int = 4 #number of workers