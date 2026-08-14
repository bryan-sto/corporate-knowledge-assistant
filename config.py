import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# Cloudflare Workers AI configuration
CF_ACCOUNT_ID: str = os.getenv("CF_ACCOUNT_ID", "")
CF_API_TOKEN: str = os.getenv("CF_API_TOKEN", "")

# Groq API configuration (fallback provider)
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# Provider preference flag
NO_CLOUDFLARE: bool = os.getenv("NO_CLOUDFLARE", "false").lower() == "true"

# Embedding Model
EMBEDDING_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Retrieval & Guardrail Parameters
# Empirically calibrated using calibrate.py (In-scope min: 0.3393, Out-of-scope max: 0.2982)
RETRIEVAL_SIMILARITY_THRESHOLD: float = 0.32
TOP_K_CHUNKS: int = 3
LLM_TEMPERATURE: float = 0.1

# Data Storage Paths
BASE_DIR: Path = Path(__file__).parent
DATA_DIR: Path = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

FAISS_INDEX_PATH: Path = DATA_DIR / "index.faiss"
METADATA_PATH: Path = DATA_DIR / "metadata.json"

