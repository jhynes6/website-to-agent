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
        Extract content from a website
        
        Args:
            url: The website URL to scrape
            max_urls: Maximum number of pages to scrape (simplified - we'll just scrape the main page)
            output_format: Output format (maintained for compatibility)
            
        Returns:
            Dictionary with extracted content
        """
        try:
            logger.info(f"🎯 EXTRACT: Starting content extraction for {url}")
            
            # Scrape the main URL
            result = self.scraper.scrape_url(url)
            
            if not result.success:
                logger.error(f"❌ EXTRACTION FAILED: {result.error}")
                return {
                    'success': False,
                    'error': result.error,
                    'content': '',
                    'urls_scraped': [],
                    'total_content_length': 0
                }
            
            # Format content based on output format
            content = self._format_content(result, output_format)
            
            logger.info(f"✅ EXTRACTION SUCCESS: Extracted {len(content)} chars from {url}")
            
            return {
                'success': True,
                'content': content,
                'urls_scraped': [url],
                'total_content_length': len(content),
                'title': result.title,
                'description': result.description,
                'keywords': result.keywords
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
