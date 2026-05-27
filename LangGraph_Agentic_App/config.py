# config.py
from pathlib import Path
import os

try:
	import dotenv
except ImportError:
	dotenv = None

BASE_DIR = Path(__file__).parent
ENV_PATH = BASE_DIR.parent.parent.parent / ".env"
if dotenv is not None:
	dotenv.load_dotenv(dotenv_path=ENV_PATH)

# ── OpenAI/Anthropic/Ollama ────────────────────────────────────────────────────────────────────
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY", "")
os.environ["OLLAMA_API_KEY"]        = os.getenv("OLLAMA_API_KEY", "")  

# ── LangSmith tracing ─────────────────────────────────────────────────────────

os.environ["LANGCHAIN_TRACING_V2"]  = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_ENDPOINT"]    = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGCHAIN_API_KEY"]     = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"]     = os.getenv("LANGCHAIN_PROJECT", "3gpp-pipeline")

os.environ["LANGSMITH_TRACING"]     = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGSMITH_ENDPOINT"]    = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGSMITH_API_KEY"]     = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGSMITH_PROJECT"]     = os.getenv("LANGCHAIN_PROJECT", "3gpp-pipeline")
os.environ["OLLAMA_API_KEY"]        = os.getenv("OLLAMA_API_KEY", "")  


# ── Model / retrieval config ──────────────────────────────────────────────────
EMBED_MODEL                      = "voyageai/voyage-4-nano"
MODEL_CACHE_DIR                  = str(BASE_DIR.parent.parent / "models")
VECTORSTORE_DIR                  = str(BASE_DIR / "vectorstore")
CHUNKS_DIR                       = str(BASE_DIR / "chunks")
COLLECTION_NAME                  = "multiple_spec"
VECTORSTORE_508_DIR              = str(BASE_DIR / "vectorstore_508")
CHUNKS_508_DIR                   = str(BASE_DIR / "chunks_508")
COLLECTION_NAME_508              = "spec_508"
DATA_508_DIR                     = str(BASE_DIR / "data_508")
VISUALIZATIONS_DIR               = str(BASE_DIR / "Evaluation" / "visualizations")
EVALUATION_DIR                   = str(BASE_DIR / "Evaluation" / "outputs")
RERANKER_MODEL                   = "cross-encoder/ms-marco-MiniLM-L-12-v2"
# LLM_MODEL                        = "gpt-5"
LLM_MODEL                        = "gpt-5-mini"
# LLM_MODEL                        = "claude-sonnet-4-6"
# LLM_MODEL                        = "gpt-oss:20b-cloud"
# LLM_MODEL                        = "gemma4:31b-cloud"
# LLM_MODEL                        = "qwen3-next:80b-cloud"
QUERY_CONFIG_PATH                = str(BASE_DIR / "user_configurations" / "NR_handover_success_intra_nr_inter_freq.json")

CONTEXT_TEMPLATE_PATH            = BASE_DIR / "context_enhancement_user.json"

# --TOP K VALUES FOR RETRIEVAL NODES--

# QUERY_ENHANCE: Needs breadth to find the correct spec/clause.
# We increase TOP_SEARCH to give the reranker a pool to choose from.
QUERY_ENHANCE_VECTOR_WEIGHT      = 0.60  # Increased for semantic intent
QUERY_ENHANCE_MMR_WEIGHT         = 0.20  # Reduced to avoid too much "noise"
QUERY_ENHANCE_BM25_WEIGHT        = 0.20  # Balanced with Vector
QUERY_ENHANCE_TOP_SEARCH         = 15    # Give the reranker a pool to work with
QUERY_ENHANCE_TOP_RERANK         = 1     # Pass more context to the enhancement node

# ANSWER_EXTRACT: Needs high precision for the specific signaling/TP.
# We prioritize Vector here to avoid the "eCall" keyword-stuffing trap.
ANSWER_EXTRACT_VECTOR_WEIGHT     = 0.60  # Primary driver for logic-matching
ANSWER_EXTRACT_MMR_WEIGHT        = 0.20  # Low diversity, we want the most relevant only
ANSWER_EXTRACT_BM25_WEIGHT       = 0.20  # Still high for specific ID matching (e.g., NR-4)
ANSWER_EXTRACT_TOP_SEARCH        = 20    # Medium pool
ANSWER_EXTRACT_TOP_RERANK        = 3   # Ensure the LLM sees the top few candidates

# ── Agent configuration ───────────────────────────────────────────────────
AGENT_MAX_RETRIES                = 3     # Max retry attempts for agent self-validation
