import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('app.log')  # File output
    ]
)

# Get logger for this application
logger = logging.getLogger('website-to-agent')

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
    raise ValueError("OPENAI_API_KEY is not set in environment variables or .env file")

# Log startup
logger.info("Configuration loaded successfully")
logger.info(f"Using model: {DEFAULT_MODEL}")
logger.info(f"Max URLs: {DEFAULT_MAX_URLS}")
logger.info(f"Crawl depth: {DEFAULT_CRAWL_DEPTH}")
logger.info("✅ Ready to use Crawl4AI for content extraction")
