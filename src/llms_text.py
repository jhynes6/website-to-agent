import asyncio
import time
from datetime import datetime
from typing import Dict, List
import logging

from src.config import DEFAULT_MAX_URLS
from src.crawl4ai_client import Crawl4AIClient, OutputFormat

# Set up logging
logger = logging.getLogger('website-to-agent')

async def extract_website_content(url: str, max_urls: int = DEFAULT_MAX_URLS, show_full_text: bool = True) -> Dict:
    """
    Extract website content using the simple scraper.
    
    Args:
        url: Website URL to extract content from
        max_urls: Maximum number of pages to crawl (simplified - we'll just scrape the main page)
        show_full_text: Whether to include comprehensive text (maintained for compatibility)
        
    Returns:
        Dictionary containing extracted content and metadata
    """
    logger.info(f"📡 SIMPLE SCRAPER START: Initializing extraction for {url}")
    logger.info(f"⚙️ Parameters: max_urls={max_urls}, show_full_text={show_full_text}")
    logger.info(f"🔧 Using simple requests + BeautifulSoup scraper")
    
    start_time = time.time()
    
    try:
        # Use the simple scraper client
        client = Crawl4AIClient()
        
        # Extract content from the URL
        result = client.extract_website_content(url, max_urls, OutputFormat.MARKDOWN)
        
        # Handle extraction failures gracefully - DON'T CRASH THE APP!
        if not result['success']:
            error_msg = result.get('error', 'Unknown error')
            logger.warning(f"⚠️ Extraction failed for {url}: {error_msg}")
            
            # Create fallback content with error information
            fallback_content = f"""# Website Content: {url}
Status: Extraction Failed
Error: {error_msg}
Extracted on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

Unfortunately, we couldn't extract content from this website.
This could be due to:
- Website blocking automated requests
- Network connectivity issues  
- Server temporarily unavailable
- Website requiring JavaScript to load content

You can still try to describe what you know about this website, and I'll do my best to help create an agent based on your description."""
            
            # Return partial result so app can continue
            elapsed_time = time.time() - start_time
            return {
                'llmstxt': fallback_content,
                'llmsfulltxt': fallback_content,
                'processed_urls': [],
                'failed_urls': [url],
                'discovered_urls': [url],
                'extraction_timestamp': datetime.now().isoformat(),
                'total_processing_time': elapsed_time,
                'extraction_error': error_msg
            }
        
        # Format content for compatibility with existing code
        content = result['content']
        
        # Handle empty content gracefully
        if not content or len(content.strip()) < 10:
            logger.warning(f"⚠️ Very little content extracted from {url}")
            content = f"Limited content was extracted from this website. This might be a dynamic site that requires JavaScript to load its content."
        
        # Add metadata header
        header = f"# Website Content: {url}\n"
        header += f"Title: {result.get('title', 'N/A')}\n"
        header += f"Description: {result.get('description', 'N/A')}\n"
        header += f"Extracted on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += f"Total pages processed: 1\n\n---\n\n"
        formatted_content = header + content
        
        elapsed_time = time.time() - start_time
        
        logger.info(f"✅ SIMPLE SCRAPER COMPLETE: Processed 1 URL in {elapsed_time:.2f}s")
        logger.info(f"📄 Content extracted: {len(formatted_content)} chars total")
        
        return {
            'llmstxt': formatted_content,
            'llmsfulltxt': formatted_content,  # Same content for both
            'processed_urls': [url],
            'failed_urls': [],
            'discovered_urls': [url],
            'extraction_timestamp': datetime.now().isoformat(),
            'total_processing_time': elapsed_time
        }
        
    except Exception as e:
        logger.error(f"❌ SIMPLE SCRAPER CRITICAL ERROR: {str(e)}")
        
        # Even in critical failures, don't crash - provide fallback content
        elapsed_time = time.time() - start_time
        fallback_content = f"""# Website Content: {url}
Status: Critical Error
Error: {str(e)}
Extracted on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

A critical error occurred while trying to extract content from this website.
This might be due to network issues or website compatibility problems.

Despite this error, you can still proceed by describing what you know about 
this website, and I'll help create an agent based on your knowledge."""

        return {
            'llmstxt': fallback_content,
            'llmsfulltxt': fallback_content,
            'processed_urls': [],
            'failed_urls': [url],
            'discovered_urls': [url],
            'extraction_timestamp': datetime.now().isoformat(),
            'total_processing_time': elapsed_time,
            'extraction_error': str(e)
        }

# Convenience function for testing
async def test_extraction(url: str = "https://example.com"):
    """Test the extraction functionality."""
    try:
        result = await extract_website_content(url, max_urls=1)
        print(f"✅ Extraction successful!")
        print(f"📄 Content length: {len(result['llmstxt'])} characters")
        print(f"🔗 URLs processed: {len(result['processed_urls'])}")
        print(f"📊 URLs discovered: {len(result['discovered_urls'])}")
        print(f"⏱️ Processing time: {result['total_processing_time']:.2f}s")
        print(f"🔧 Using simple scraper (requests + BeautifulSoup)")
        print(f"📝 Content preview: {result['llmstxt'][:500]}...")
        return result
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    asyncio.run(test_extraction("https://example.com"))