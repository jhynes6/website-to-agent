import asyncio
import time
from datetime import datetime
from typing import Dict, List
import logging

from src.config import DEFAULT_MAX_URLS
from src.crawl4ai_client import (
    Crawl4AIClient, 
    CrawlConfig, 
    CrawlMode, 
    OutputFormat, 
    quick_crawl_async,
    OFFICIAL_SEEDING_AVAILABLE
)

# Set up logging
logger = logging.getLogger('website-to-agent')

async def extract_website_content(url: str, max_urls: int = DEFAULT_MAX_URLS, show_full_text: bool = True) -> Dict:
    """
    Extract website content using the improved Crawl4AI client.
    
    Args:
        url: Website URL to extract content from
        max_urls: Maximum number of pages to crawl (1-100)
        show_full_text: Whether to include comprehensive text (always True with Crawl4AI)
        
    Returns:
        Dictionary containing extracted content and metadata
    """
    logger.info(f"🕷️ CRAWL4AI START: Initializing extraction for {url}")
    logger.info(f"⚙️ Parameters: max_urls={max_urls}, show_full_text={show_full_text}")
    logger.info(f"🔧 Using official seeding: {OFFICIAL_SEEDING_AVAILABLE}")
    
    start_time = time.time()
    
    try:
        # Configure crawling with smart defaults
        config = CrawlConfig(
            max_pages=max_urls,
            max_depth=2,
            timeout=30,
            exclude_external_links=True,
            # Use seeding if available for better URL discovery
            seeding_source="sitemap+cc" if OFFICIAL_SEEDING_AVAILABLE else "sitemap",
            seeding_extract_head=True,
            seeding_concurrency=50,
            seeding_hits_per_sec=10
        )
        
        # Use the improved client
        async with Crawl4AIClient(config) as client:
            # Choose crawl mode based on page count and seeding availability
            if max_urls == 1:
                mode = CrawlMode.SINGLE_PAGE
            elif OFFICIAL_SEEDING_AVAILABLE:
                mode = CrawlMode.SEEDED
                logger.info("🌱 Using official seeded crawling for URL discovery")
            else:
                mode = CrawlMode.MULTI_PAGE
                logger.info("🌐 Using multi-page crawling (seeding not available)")
            
            # Perform the crawl
            results = await client.crawl_async(
                url=url,
                mode=mode,
                output_format=OutputFormat.MARKDOWN
            )
            
            # Handle both single result and list of results
            if not isinstance(results, list):
                results = [results]
            
            # Extract successful results
            successful_results = [r for r in results if r.success]
            failed_results = [r for r in results if not r.success]
            
            if not successful_results:
                logger.error("❌ No successful extractions")
                raise Exception("No content could be extracted from any URLs")
            
            # Combine content from all successful results
            combined_content = []
            processed_urls = []
            
            for result in successful_results:
                if result.extracted_content:
                    combined_content.append(f"# {result.url}\n\n{result.extracted_content}")
                    processed_urls.append(result.url)
            
            # Create final content
            llmstxt_content = "\n\n---\n\n".join(combined_content) if combined_content else ""
            
            # Add metadata header
            if llmstxt_content:
                header = f"# Website Content: {url}\n"
                header += f"Extracted on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                header += f"Total pages processed: {len(successful_results)}\n\n---\n\n"
                llmstxt_content = header + llmstxt_content
            
            elapsed_time = time.time() - start_time
            
            logger.info(f"✅ CRAWL4AI COMPLETE: Processed {len(successful_results)} URLs in {elapsed_time:.2f}s")
            logger.info(f"📄 Content extracted: {len(llmstxt_content)} chars total")
            logger.info(f"📊 Success rate: {len(successful_results)}/{len(results)} URLs")
            
            if failed_results:
                failed_urls = [r.url for r in failed_results]
                logger.warning(f"⚠️ Failed URLs: {failed_urls}")
            else:
                failed_urls = []
            
            return {
                'llmstxt': llmstxt_content,
                'llmsfulltxt': llmstxt_content,  # Same content for both
                'processed_urls': processed_urls,
                'failed_urls': failed_urls,
                'discovered_urls': [r.url for r in results],
                'extraction_timestamp': datetime.now().isoformat(),
                'total_processing_time': elapsed_time
            }
        
    except Exception as e:
        logger.error(f"❌ CRAWL4AI ERROR: {str(e)}")
        raise Exception(f"Content extraction failed: {str(e)}")

# Legacy helper functions removed - now using improved Crawl4AI client

# Convenience function for testing
async def test_extraction(url: str = "https://example.com"):
    """Test the extraction functionality."""
    try:
        result = await extract_website_content(url, max_urls=3)
        print(f"✅ Extraction successful!")
        print(f"📄 Content length: {len(result['llmstxt'])} characters")
        print(f"🔗 URLs processed: {len(result['processed_urls'])}")
        print(f"📊 URLs discovered: {len(result['discovered_urls'])}")
        print(f"⏱️ Processing time: {result['total_processing_time']:.2f}s")
        print(f"🔧 Official seeding: {OFFICIAL_SEEDING_AVAILABLE}")
        print(f"📝 Content preview: {result['llmstxt'][:500]}...")
        return result
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    asyncio.run(test_extraction("https://example.com"))