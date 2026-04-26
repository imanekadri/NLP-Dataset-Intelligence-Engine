import os
from dataclasses import dataclass

#######################
# Global Configuration
#######################

@dataclass
class GlobalConfig:
    """Global configuration for all agents"""
    
    DATASETS_DIR = os.path.join(os.path.dirname(__file__), "../data")
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../output")
    
    ENCODINGS_TO_TRY = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
    SUPPORTED_FORMATS = [
        '.txt', '.csv', '.json', '.xml', '.pdf', '.docx',
        '.log', '.py', '.js', '.html', '.htm',
        '.png', '.jpg', '.jpeg', '.tiff', '.bmp'
    ]
    
    # Default embedding model for all agents
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


#######################
# Agent1 Config
#######################

@dataclass
class Agent1Config(GlobalConfig):
    """Configuration for Agent1: Text Ingestion"""
    
    AGENT_NAME = "agent1"
    AGENT_OUTPUT_DIR = os.path.join(GlobalConfig.OUTPUT_DIR, AGENT_NAME)
    
    EXTRACTED_DIR = os.path.join(AGENT_OUTPUT_DIR, "extracted")
    TRACE_CSV = os.path.join(AGENT_OUTPUT_DIR, "trace_index.csv")
    REPORT_JSON = os.path.join(AGENT_OUTPUT_DIR, "report.json")
    CLEANED_DIR = os.path.join(AGENT_OUTPUT_DIR, "cleaned")
    DOCUMENTS_INFO_JSON = os.path.join(AGENT_OUTPUT_DIR, "documents_info.json")


#######################
# Agent2 Config
#######################

@dataclass
class Agent2Config:
    """Configuration for Agent2: Profiling & Clustering"""
    
    INPUT_FOLDER = Agent1Config.EXTRACTED_DIR
    OUTPUT_DIR = os.path.join(GlobalConfig.OUTPUT_DIR, "agent2")
    CATEGORIES_DIR = os.path.join(OUTPUT_DIR, "categories")
    DATASET_PROFILE = os.path.join(OUTPUT_DIR, "dataset_profile.json")
    CLUSTER_REPORT = os.path.join(OUTPUT_DIR, "clusters.json")
    TOPIC_REPORT = os.path.join(OUTPUT_DIR, "topics.json")
    TRACE_CSV2 = os.path.join(OUTPUT_DIR, "trace_index.csv")

    # Embeddings
    MODEL_NAME = GlobalConfig.EMBEDDING_MODEL
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
    MIN_TOPIC_SIZE = 2


#######################
# Agent3 Config
#######################

@dataclass
class Agent3Config:
    OUTPUT_DIR = os.path.join(GlobalConfig.OUTPUT_DIR, "agent3")

    EMBEDDING_MODEL = GlobalConfig.EMBEDDING_MODEL

    SPACY_EN = "en_core_web_sm"
    SPACY_FR = "fr_core_news_sm"

    SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

    PII_SCORE_THRESHOLD = 0.7
    
    IMPORTANT_PII = [
        "PERSON",
        "EMAIL_ADDRESS",  # Mapped to EMAIL in the code logic
        "PHONE_NUMBER",
        "LOCATION"
    ]
    
    VALID_ENTITY_LABELS = ["ORG", "PERSON", "TECH"]

    TECH_KEYWORDS = [
        # Languages & Frameworks
        "Python", "SQL", "Java", "C++", "TypeScript", "JavaScript", "R", "Julia",
        "React", "Node.js", "Flask", "Django", "FastAPI", "Spring Boot",
        # ML / AI
        "Hadoop", "TensorFlow", "PyTorch", "BERT", "NLP", "GPT", "LLM",
        "Machine Learning", "Deep Learning", "Artificial Intelligence", "AI", "ML",
        "Scikit-Learn", "Keras", "OpenCV", "YOLO", "Transformers", "HuggingFace",
        # Data
        "Spark", "PySpark", "Pandas", "NumPy", "SciPy", "Matplotlib", "Seaborn",
        "Plotly", "Dask", "Apache Kafka", "Airflow",
        # DevOps & Cloud
        "Docker", "Kubernetes", "AWS", "Azure", "GCP", "Linux", "Unix", "CUDA",
        "GitHub", "Git", "GitLab", "CI/CD",
        # Tools & IDEs
        "Jupyter Notebook", "Jupyter", "Google Colab", "Anaconda", "Spyder",
        "PyCharm", "VS Code", "Visual Studio", "IntelliJ", "Anaconda Prompt",
        # Databases
        "PostgreSQL", "MongoDB", "Redis", "MySQL", "SQLite", "Elasticsearch",
        "Scrapy", "Power BI", "Tableau", "Excel",
        # Tech Companies
        "Adobe", "Microsoft", "Google", "Apple", "Meta", "Facebook", "Amazon", 
        "Netflix", "Oracle", "IBM", "Intel", "NVIDIA", "AMD", "Salesforce", 
        "OpenAI", "Anthropic", "DeepMind", "Twitter", "LinkedIn", "Instagram",
        # Specific named tools
        "Apache Spark", "Apache Hadoop", "Monte Carlo", "Guido van Rossum", "Andrew Ng"
    ]

    GENERIC_WORDS_BLACKLIST = [
        "Example", "Data", "Result", "Analysis", "Report", "Document", 
        "Information", "File", "System", "Process", "Training", "Development",
        "Standard", "Project", "Manager", "User", "Client", "Service", "Ability",
        "Task", "Object", "Simple", "Generic", "Testing", "Validate",
        "Chapter", "Introduction", "Overview", "Summary", "Section", "Table",
        "Figure", "Reference", "Conclusion", "Key Features", "Common",
        "Exploratory Data Analysis", "Data Science", "Business Intelligence",
        # Legal & Job roles (Fixing False Persons)
        "Foreclosure", "Lien", "Accountant", "Clerk", "Underwriter", "Vendor",
        "Performed", "Processed", "Basics", "Basics", "Interactive", "Recommended",
        "Advanced", "Basics", "Training", "Creating", "Writing", "Learning",
        "Intelligence", "Binary", "String", "Prompt", "Access", "Line", "Status",
        "Definition", "Workload", "Compliance", "Staff", "Assistant", "Lead",
        # Geographic (Fixing False Persons)
        "Alger", "Oran", "Annaba", "Algerie", "Algeria", "Neurology", "Cardiology",
        "Zéralda", "Affroun", "Reghaia", "Beida", "Naadja", "Baba Ali", "Hussein Dey",
        "Sidi", "Abdel", "Abde", "Tessala", "Merdja", "Gué", "Cne", "Méridja",
        # Military / HQ (Fixing False Persons)
        "MAJCOM", "Headquarters", "Command", "Sites", "Field", "Aacnnio", "Sparialaltalon"
    ]

    # Rule-based ORG filtering: reclassify if these look-like verbs appear
    ORG_VERB_SIGNATURES = [
        "Training", "Modeling", "Apply", "Validate", "Processing", "Testing",
        "Develop", "Learning", "Using", "Implementing"
    ]

    # Multilingual Sentiment Triggers: Analysis only runs if these appear
    SENTIMENT_TRIGGER_KEYWORDS = {
        "en": ["good", "bad", "excellent", "poor", "critique", "advantage", "disadvantage", "important", "issue", "problem", "love", "hate", "worth", "useful"],
        "fr": ["bon", "mauvais", "excellent", "avantage", "inconvénient", "important", "critique", "problème", "utile", "inutile", "dommage"],
        "ar": ["جيد", "سيء", "ممتاز", "رائع", "مهم", "مشكلة", "عيب", "ميزة", "مفيد", "فاشل", "خطأ"]
    }

    MIN_SUBDOMAIN_SIZE = 2
    MIN_DOCS_FOR_SUBDOMAINS = 10


#######################
# Instantiate configs
#######################

global_config = GlobalConfig()
agent1_config = Agent1Config()
agent2_config = Agent2Config()
agent3_config = Agent3Config()