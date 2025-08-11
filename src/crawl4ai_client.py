"""
Simple Web Scraper Client
=========================

A reliable web scraper using requests + BeautifulSoup to replace the problematic Crawl4AI.
Designed to be fast, simple, and crash-resistant.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse, urljoin
import requests
from bs4 import BeautifulSoup
import re

# Set up logging
logger = logging.getLogger('website-to-agent')

class OutputFormat(Enum):
    """Output format options for crawled content"""
    HTML = "html"
    MARKDOWN = "markdown"
    TEXT = "text"
    JSON = "json"
    STRUCTURED = "structured"

@dataclass
class CrawlResult:
    """Result of a web crawl operation"""
    url: str
    html: str = ""
    markdown: str = ""
    extracted_content: str = ""
    cleaned_html: str = ""
    text: str = ""
    title: str = ""
    description: str = ""
    keywords: List[str] = None
    success: bool = True
    status_code: int = 200
    error: str = ""
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []

class SimpleScraper:
    """Simple, reliable web scraper using requests + BeautifulSoup"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        self.session.timeout = 30
        
    def scrape_url(self, url: str) -> CrawlResult:
        """Scrape a single URL"""
        try:
            logger.info(f"📡 SCRAPING: Starting scrape of {url}")
            
            # Make the request
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            logger.info(f"✅ HTTP {response.status_code}: Successfully fetched {url}")
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                script.decompose()
            
            # Extract title
            title_tag = soup.find('title')
            title = title_tag.get_text().strip() if title_tag else ""
            
            # Extract description
            desc_tag = soup.find('meta', attrs={'name': 'description'})
            description = desc_tag.get('content', '').strip() if desc_tag else ""
            
            # Extract keywords
            keywords_tag = soup.find('meta', attrs={'name': 'keywords'})
            keywords = keywords_tag.get('content', '').split(',') if keywords_tag else []
            keywords = [k.strip() for k in keywords if k.strip()]
            
            # Get main content
            # Try to find main content areas
            main_content = None
            for selector in ['main', 'article', '.content', '.main-content', '#content', '#main']:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            if not main_content:
                # If no main content area found, use body
                main_content = soup.find('body')
            
            if not main_content:
                # Last resort - use the whole soup
                main_content = soup
            
            # Extract clean text
            text_content = main_content.get_text(separator='\n', strip=True)
            
            # Clean up text - remove excessive whitespace
            text_content = re.sub(r'\n\s*\n', '\n\n', text_content)
            text_content = re.sub(r' +', ' ', text_content)
            text_content = text_content.strip()
            
            # Create markdown-like content
            markdown_content = self._html_to_markdown(main_content)
            
            logger.info(f"🎯 EXTRACTED: {len(text_content)} chars of text from {url}")
            
            return CrawlResult(
                url=url,
                html=str(main_content),
                markdown=markdown_content,
                extracted_content=text_content,
                cleaned_html=str(main_content),
                text=text_content,
                title=title,
                description=description,
                keywords=keywords,
                success=True,
                status_code=response.status_code
            )
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ REQUEST ERROR: Failed to scrape {url}: {str(e)}")
            return CrawlResult(
                url=url,
                success=False,
                status_code=getattr(e.response, 'status_code', 0) if hasattr(e, 'response') else 0,
                error=str(e)
            )
        except Exception as e:
            logger.error(f"❌ SCRAPE ERROR: Unexpected error scraping {url}: {str(e)}")
            return CrawlResult(
                url=url,
                success=False,
                error=str(e)
            )
    
    def _html_to_markdown(self, soup) -> str:
        """Convert HTML soup to simple markdown-like format"""
        try:
            # Extract headings
            content_parts = []
            
            for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'li']):
                text = element.get_text(strip=True)
                if not text:
                    continue
                
                if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    level = int(element.name[1])
                    content_parts.append(f"{'#' * level} {text}")
                elif element.name == 'li':
                    content_parts.append(f"- {text}")
                else:
                    content_parts.append(text)
            
            return '\n\n'.join(content_parts)
            
        except Exception as e:
            logger.error(f"❌ MARKDOWN CONVERSION ERROR: {str(e)}")
            return soup.get_text(separator='\n', strip=True)

class Crawl4AIClient:
    """
    Simple web scraper client that replaces the problematic Crawl4AI.
    Maintains the same interface for compatibility.
    """
    
    def __init__(self):
        self.scraper = SimpleScraper()
        logger.info("✅ SIMPLE SCRAPER: Initialized reliable requests + BeautifulSoup scraper")
    
    def extract_website_content(self, url: str, max_urls: int = 5, 
                              output_format: OutputFormat = OutputFormat.MARKDOWN) -> Dict[str, Any]:
        """
        Extract content from a website and optionally crawl additional pages
        
        Args:
            url: The website URL to scrape
            max_urls: Maximum number of pages to scrape from the domain
            output_format: Output format (maintained for compatibility)
            
        Returns:
            Dictionary with extracted content
        """
        try:
            logger.info(f"🎯 EXTRACT: Starting content extraction for {url} (max_urls: {max_urls})")
            
            scraped_urls = []
            all_content = []
            base_domain = urlparse(url).netloc
            urls_to_scrape = [url]
            scraped_set = set()
            
            logger.info(f"🌐 CRAWLING STRATEGY: Will scrape up to {max_urls} pages from domain: {base_domain}")
            logger.info(f"📋 INITIAL QUEUE: urls_to_scrape = {urls_to_scrape}")
            
            # Scrape pages up to max_urls limit
            while urls_to_scrape and len(scraped_urls) < max_urls:
                logger.info(f"🔄 LOOP ITERATION: Processing queue with {len(urls_to_scrape)} URLs remaining")
                current_url = urls_to_scrape.pop(0)
                logger.info(f"📤 DEQUEUED: Popped '{current_url}' from queue")
                logger.info(f"📋 QUEUE NOW: urls_to_scrape = {urls_to_scrape}")
                
                # Skip if already scraped
                if current_url in scraped_set:
                    logger.info(f"⏭️ SKIP: {current_url} already in scraped_set")
                    continue
                    
                logger.info(f"📡 SCRAPING ({len(scraped_urls)+1}/{max_urls}): {current_url}")
                result = self.scraper.scrape_url(current_url)
                scraped_set.add(current_url)
                
                if result.success:
                    scraped_urls.append(current_url)
                    content = self._format_content(result, output_format)
                    all_content.append(f"\n\n=== CONTENT FROM: {current_url} ===\n\n{content}")
                    logger.info(f"✅ SCRAPED ({len(scraped_urls)}/{max_urls}): {current_url} - {len(content)} chars")
                    
                    # Find more internal links if we need more pages
                    if len(scraped_urls) < max_urls:
                        try:
                            logger.info(f"🔍 LINK DISCOVERY: Analyzing {current_url} for internal links...")
                            
                            soup = BeautifulSoup(result.html, 'html.parser')
                            links = soup.find_all('a', href=True)
                            logger.info(f"📄 HTML ANALYSIS: Found {len(links)} total links on page")
                            
                            # Show current queue state before adding new links
                            logger.info(f"📋 QUEUE BEFORE: urls_to_scrape = {urls_to_scrape}")
                            
                            new_links_found = 0
                            rejected_links = {
                                'external_domain': 0,
                                'already_scraped': 0,
                                'already_queued': 0,
                                'invalid_file_type': 0
                            }
                            discovered_urls = []
                            
                            for link in links:
                                href = link['href']
                                # Convert relative URLs to absolute
                                full_url = urljoin(current_url, href)
                                parsed_link = urlparse(full_url)
                                
                                # Apply filters and log the decision
                                if parsed_link.netloc != base_domain:
                                    rejected_links['external_domain'] += 1
                                    logger.debug(f"🚫 REJECT (external): {full_url}")
                                elif full_url in scraped_set:
                                    rejected_links['already_scraped'] += 1
                                    logger.debug(f"🚫 REJECT (already scraped): {full_url}")
                                elif full_url in urls_to_scrape:
                                    rejected_links['already_queued'] += 1
                                    logger.debug(f"🚫 REJECT (already queued): {full_url}")
                                elif full_url.endswith(('.pdf', '.jpg', '.png', '.gif', '.css', '.js', '.zip')):
                                    rejected_links['invalid_file_type'] += 1
                                    logger.debug(f"🚫 REJECT (file type): {full_url}")
                                else:
                                    # This link passes all filters - add it to queue
                                    urls_to_scrape.append(full_url)
                                    discovered_urls.append(full_url)
                                    new_links_found += 1
                                    logger.info(f"✅ ACCEPT: {full_url}")
                            
                            # Log discovery summary
                            logger.info(f"🔗 DISCOVERY SUMMARY:")
                            logger.info(f"   • Total links found: {len(links)}")
                            logger.info(f"   • New URLs added to queue: {new_links_found}")
                            logger.info(f"   • Rejected - External domain: {rejected_links['external_domain']}")
                            logger.info(f"   • Rejected - Already scraped: {rejected_links['already_scraped']}")
                            logger.info(f"   • Rejected - Already queued: {rejected_links['already_queued']}")
                            logger.info(f"   • Rejected - Invalid file type: {rejected_links['invalid_file_type']}")
                            
                            if discovered_urls:
                                logger.info(f"🆕 NEW URLS DISCOVERED:")
                                for i, discovered_url in enumerate(discovered_urls, 1):
                                    logger.info(f"     {i}. {discovered_url}")
                            
                            # Show updated queue state
                            logger.info(f"📋 QUEUE AFTER: urls_to_scrape = {urls_to_scrape}")
                            logger.info(f"📊 QUEUE STATUS: {len(urls_to_scrape)} URLs remaining to scrape")
                                
                        except Exception as link_error:
                            logger.warning(f"⚠️ LINK EXTRACTION ERROR: {str(link_error)}")
                    
                else:
                    logger.warning(f"⚠️ SKIPPED: Failed to scrape {current_url} - {result.error}")
            
            # Combine all content
            combined_content = "\n".join(all_content)
            
            # Log final summary
            logger.info(f"🎯 CRAWL COMPLETE: Successfully scraped {len(scraped_urls)} pages:")
            for i, scraped_url in enumerate(scraped_urls, 1):
                logger.info(f"  {i}. {scraped_url}")
            
            if not scraped_urls:
                logger.error(f"❌ EXTRACTION FAILED: No pages could be scraped")
                return {
                    'success': False,
                    'error': 'No pages could be scraped successfully',
                    'content': '',
                    'urls_scraped': [],
                    'total_content_length': 0
                }
            
            logger.info(f"✅ EXTRACTION SUCCESS: Extracted {len(combined_content)} chars total from {len(scraped_urls)} pages")
            
            return {
                'success': True,
                'content': combined_content,
                'urls_scraped': scraped_urls,
                'total_content_length': len(combined_content),
                'pages_scraped': len(scraped_urls),
                'title': scraped_urls[0] if scraped_urls else url,  # Use first URL as title reference
                'description': f"Content from {len(scraped_urls)} pages on {base_domain}",
                'keywords': []  # Keywords not available for multi-page scrapes
            }
                    
        except Exception as e:
            logger.error(f"❌ EXTRACT ERROR: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'content': '',
                'urls_scraped': [],
                'total_content_length': 0
            }
    
    def _format_content(self, result: CrawlResult, output_format: OutputFormat) -> str:
        """Format crawled content based on requested format"""
        content_attributes = [
            'markdown',           # Most common
            'extracted_content',  # Alternative 
            'cleaned_html',       # Fallback
            'html',              # Raw HTML as last resort
            'text',              # Plain text
        ]
        
        if output_format == OutputFormat.MARKDOWN:
            for attr in content_attributes:
                content = getattr(result, attr, None)
                if content and content.strip():
                    return content.strip()
            return ''
        
        elif output_format == OutputFormat.HTML:
            return result.html or result.cleaned_html or ''
        
        elif output_format == OutputFormat.TEXT:
            return result.text or result.extracted_content or ''
        
        elif output_format == OutputFormat.JSON:
            return json.dumps({
                'title': result.title,
                'content': result.text or result.extracted_content,
                'description': result.description,
                'keywords': result.keywords
            }, indent=2)
        
        elif output_format == OutputFormat.STRUCTURED:
            return result.markdown or result.text or result.extracted_content or ''
        
        return result.text or result.extracted_content or ''

# For compatibility with existing code
async def extract_website_content(url: str, max_urls: int = 5, 
                                output_format: OutputFormat = OutputFormat.MARKDOWN) -> Dict[str, Any]:
    """Async wrapper for website content extraction"""
    client = Crawl4AIClient()
    return client.extract_website_content(url, max_urls, output_format)
