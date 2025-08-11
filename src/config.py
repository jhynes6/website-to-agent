import os
import logging

# Load environment variables from .env file if it exists (for local development)
# In production (Docker), environment variables are set directly
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, rely on system environment variables
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('app.log') if os.access('.', os.W_OK) else logging.StreamHandler()  # File output if writable
    ]
)

# Get logger for this application
logger = logging.getLogger('website-to-agent')

# Debug: Log all environment variables starting with OPENAI
for key, value in os.environ.items():
    if key.startswith('OPENAI'):
        logger.info(f"Found env var: {key}={'***' if value else 'EMPTY'}")

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Default settings
DEFAULT_MAX_URLS = 10
DEFAULT_USE_FULL_TEXT = True
DEFAULT_MODEL = "gpt-4o"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 1024

# Crawling settings
DEFAULT_CRAWL_DEPTH = 2
DEFAULT_BATCH_SIZE = 3
DEFAULT_REQUEST_TIMEOUT = 30

# Ensure required API keys are available
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY is not set")
    logger.info("Available environment variables:")
    for key in sorted(os.environ.keys()):
        logger.info(f"  {key}={'***' if 'KEY' in key or 'PASSWORD' in key or 'TOKEN' in key else os.environ[key]}")
    raise ValueError("OPENAI_API_KEY is not set in environment variables or .env file")

# Log startup
logger.info("Configuration loaded successfully")
logger.info(f"Using model: {DEFAULT_MODEL}")
logger.info(f"Max URLs: {DEFAULT_MAX_URLS}")
logger.info(f"Crawl depth: {DEFAULT_CRAWL_DEPTH}")
logger.info("✅ Ready to use Crawl4AI for content extraction")
