"""
Crawl4AI API Client

A comprehensive client for the Crawl4AI web crawler, providing both synchronous 
and asynchronous interfaces with advanced crawling capabilities.

Based on the official Crawl4AI repository: https://github.com/unclecode/crawl4ai
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass, asdict
from urllib.parse import urljoin, urlparse
import aiohttp
from bs4 import BeautifulSoup

# Ensure event loop compatibility
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

from crawl4ai import AsyncWebCrawler, WebCrawler
from crawl4ai.extraction_strategy import (
    ExtractionStrategy,
    LLMExtractionStrategy,
    CosineStrategy,
    JsonCssExtractionStrategy
)
from crawl4ai.chunking_strategy import (
    ChunkingStrategy,
    RegexChunking,
    NlpSentenceChunking,
    FixedLengthWordChunking
)

# Import official async configurations if available
try:
    from crawl4ai.async_configs import SeedingConfig, BrowserConfig, CrawlerRunConfig
    from crawl4ai.async_url_seeder import AsyncUrlSeeder
    OFFICIAL_SEEDING_AVAILABLE = True
except ImportError:
    OFFICIAL_SEEDING_AVAILABLE = False
    print("⚠️ Official async seeding not available. Using fallback implementation.")

# Set up logging
logger = logging.getLogger(__name__)


class CrawlMode(Enum):
    """Crawling modes supported by the client"""
    SINGLE_PAGE = "single_page"
    MULTI_PAGE = "multi_page" 
    SITEMAP = "sitemap"
    RECURSIVE = "recursive"
    GRAPH = "graph"
    SEEDED = "seeded"  # New: Uses official AsyncUrlSeeder


class OutputFormat(Enum):
    """Output formats for crawled content"""
    MARKDOWN = "markdown"
    HTML = "html"
    TEXT = "text"
    JSON = "json"
    STRUCTURED = "structured"


@dataclass
class CrawlConfig:
    """Configuration for crawling operations"""
    # Basic settings
    word_count_threshold: int = 10
    exclude_external_links: bool = True
    exclude_social_media_links: bool = True
    bypass_cache: bool = False
    delay_before_return_html: float = 2.0
    remove_overlay_elements: bool = True
    simulate_user: bool = True
    override_navigator: bool = True
    
    # Advanced settings
    user_agent: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    cookies: Optional[Dict[str, str]] = None
    proxy: Optional[str] = None
    timeout: int = 30
    
    # Extraction settings
    extraction_strategy: Optional[ExtractionStrategy] = None
    chunking_strategy: Optional[ChunkingStrategy] = None
    
    # Multi-page settings
    max_depth: int = 2
    max_pages: int = 10
    same_domain_only: bool = True
    
    # Content filtering
    css_selector: Optional[str] = None
    excluded_tags: List[str] = None
    only_text: bool = False
    
    # Official seeding settings (when available)
    seeding_source: str = "sitemap+cc"  # "sitemap", "cc", or "sitemap+cc"
    seeding_pattern: str = "*"  # URL pattern filter
    seeding_live_check: bool = False  # Verify URL liveness
    seeding_extract_head: bool = True  # Extract head metadata for relevance
    seeding_query: Optional[str] = None  # Query for BM25 relevance scoring
    seeding_score_threshold: Optional[float] = None  # Minimum relevance score
    seeding_concurrency: int = 100  # Concurrent requests for seeding
    seeding_hits_per_sec: int = 10  # Rate limiting
    seeding_force: bool = False  # Bypass cache
    
    def __post_init__(self):
        if self.excluded_tags is None:
            self.excluded_tags = ['script', 'style', 'nav', 'footer', 'header']
        
        if self.headers is None:
            self.headers = {
                'User-Agent': self.user_agent or 'Mozilla/5.0 (compatible; Crawl4AI/1.0; +https://crawl4ai.com)'
            }


@dataclass
class CrawlResult:
    """Result of a crawling operation"""
    url: str
    success: bool
    status_code: Optional[int] = None
    html: Optional[str] = None
    markdown: Optional[str] = None
    extracted_content: Optional[str] = None
    links: List[str] = None
    media: List[str] = None
    metadata: Dict[str, Any] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0
    timestamp: str = None
    relevance_score: Optional[float] = None  # For seeded URLs with query scoring
    
    def __post_init__(self):
        if self.links is None:
            self.links = []
        if self.media is None:
            self.media = []
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class Crawl4AIClient:
    """
    Comprehensive Crawl4AI client with both sync and async support
    """
    
    def __init__(self, config: Optional[CrawlConfig] = None):
        """
        Initialize the Crawl4AI client
        
        Args:
            config: CrawlConfig instance with crawling configuration
        """
        self.config = config or CrawlConfig()
        self._async_crawler: Optional[AsyncWebCrawler] = None
        self._sync_crawler: Optional[WebCrawler] = None
        self._url_seeder: Optional[AsyncUrlSeeder] = None
        self.session_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_processing_time': 0.0,
            'urls_seeded': 0,
            'session_start': datetime.now().isoformat()
        }
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start_async()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close_async()
        
    def __enter__(self):
        """Sync context manager entry"""
        self.start_sync()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Sync context manager exit"""
        self.close_sync()
    
    async def start_async(self):
        """Initialize async crawler and seeder"""
        if self._async_crawler is None:
            crawler_kwargs = {
                'always_bypass_cache': self.config.bypass_cache,
                'headers': self.config.headers,
                'verbose': True
            }
            
            if self.config.proxy:
                crawler_kwargs['proxy'] = self.config.proxy
                
            self._async_crawler = AsyncWebCrawler(**crawler_kwargs)
            
        # Initialize URL seeder if official implementation is available
        if OFFICIAL_SEEDING_AVAILABLE and self._url_seeder is None:
            self._url_seeder = AsyncUrlSeeder()
            
    def start_sync(self):
        """Initialize sync crawler"""
        if self._sync_crawler is None:
            crawler_kwargs = {
                'always_bypass_cache': self.config.bypass_cache,
                'headers': self.config.headers,
                'verbose': True
            }
            
            if self.config.proxy:
                crawler_kwargs['proxy'] = self.config.proxy
                
            self._sync_crawler = WebCrawler(**crawler_kwargs)
            
    async def close_async(self):
        """Close async crawler and seeder"""
        if self._async_crawler:
            # AsyncWebCrawler handles cleanup automatically when used as context manager
            self._async_crawler = None
            
        if self._url_seeder:
            # Close URL seeder if it has cleanup methods
            self._url_seeder = None
            
    def close_sync(self):
        """Close sync crawler"""
        if self._sync_crawler:
            # WebCrawler cleanup if needed
            self._sync_crawler = None
    
    async def crawl_async(
        self, 
        url: str, 
        mode: CrawlMode = CrawlMode.SINGLE_PAGE,
        output_format: OutputFormat = OutputFormat.MARKDOWN,
        custom_config: Optional[CrawlConfig] = None
    ) -> Union[CrawlResult, List[CrawlResult]]:
        """
        Asynchronously crawl a URL or multiple URLs
        
        Args:
            url: URL to crawl
            mode: Crawling mode (single page, multi-page, seeded, etc.)
            output_format: Desired output format
            custom_config: Override default config for this crawl
            
        Returns:
            CrawlResult or list of CrawlResults
        """
        config = custom_config or self.config
        start_time = time.time()
        
        try:
            await self.start_async()
            
            if mode == CrawlMode.SINGLE_PAGE:
                result = await self._crawl_single_page_async(url, config, output_format)
                self._update_stats(result.success, time.time() - start_time)
                return result
                
            elif mode == CrawlMode.SEEDED and OFFICIAL_SEEDING_AVAILABLE:
                results = await self._crawl_seeded_async(url, config, output_format)
                self._update_stats(len([r for r in results if r.success]), time.time() - start_time)
                return results
                
            elif mode == CrawlMode.MULTI_PAGE:
                results = await self._crawl_multi_page_async(url, config, output_format)
                self._update_stats(len([r for r in results if r.success]), time.time() - start_time)
                return results
                
            elif mode == CrawlMode.SITEMAP:
                results = await self._crawl_sitemap_async(url, config, output_format)
                self._update_stats(len([r for r in results if r.success]), time.time() - start_time)
                return results
                
            else:
                raise ValueError(f"Unsupported crawl mode: {mode}")
                
        except Exception as e:
            logger.error(f"Crawl failed for {url}: {str(e)}")
            self._update_stats(False, time.time() - start_time)
            return CrawlResult(
                url=url,
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time
            )
    
    def crawl_sync(
        self, 
        url: str, 
        mode: CrawlMode = CrawlMode.SINGLE_PAGE,
        output_format: OutputFormat = OutputFormat.MARKDOWN,
        custom_config: Optional[CrawlConfig] = None
    ) -> Union[CrawlResult, List[CrawlResult]]:
        """
        Synchronously crawl a URL or multiple URLs
        
        Args:
            url: URL to crawl
            mode: Crawling mode (single page, multi-page, etc.)
            output_format: Desired output format
            custom_config: Override default config for this crawl
            
        Returns:
            CrawlResult or list of CrawlResults
        """
        # Run async version in sync context
        return asyncio.run(self.crawl_async(url, mode, output_format, custom_config))
    
    async def _crawl_seeded_async(
        self,
        domain_or_url: str,
        config: CrawlConfig,
        output_format: OutputFormat
    ) -> List[CrawlResult]:
        """Crawl using official AsyncUrlSeeder for URL discovery"""
        if not OFFICIAL_SEEDING_AVAILABLE:
            logger.warning("Official seeding not available, falling back to multi-page crawling")
            return await self._crawl_multi_page_async(domain_or_url, config, output_format)
        
        start_time = time.time()
        
        # Extract domain from URL if needed
        if domain_or_url.startswith(('http://', 'https://')):
            domain = urlparse(domain_or_url).netloc
        else:
            domain = domain_or_url
            
        logger.info(f"🌱 SEEDED CRAWL: Starting URL discovery for domain {domain}")
        
        # Configure seeding
        seeding_config = SeedingConfig(
            source=config.seeding_source,
            pattern=config.seeding_pattern,
            live_check=config.seeding_live_check,
            extract_head=config.seeding_extract_head,
            max_urls=config.max_pages,
            concurrency=config.seeding_concurrency,
            hits_per_sec=config.seeding_hits_per_sec,
            force=config.seeding_force,
            query=config.seeding_query,
            score_threshold=config.seeding_score_threshold,
            verbose=True
        )
        
        try:
            # Discover URLs using official seeder
            discovered_data = await self._url_seeder.urls(domain, seeding_config)
            
            # Extract URLs from discovered data
            discovered_urls = []
            for item in discovered_data:
                if isinstance(item, dict) and 'url' in item:
                    discovered_urls.append(item['url'])
                elif isinstance(item, str):
                    discovered_urls.append(item)
            
            self.session_stats['urls_seeded'] = len(discovered_urls)
            logger.info(f"✅ SEEDED DISCOVERY: Found {len(discovered_urls)} URLs")
            
            # Crawl discovered URLs
            results = []
            semaphore = asyncio.Semaphore(5)  # Limit concurrent crawls
            
            async def crawl_with_semaphore(url_data):
                async with semaphore:
                    if isinstance(url_data, dict):
                        url = url_data['url']
                        relevance_score = url_data.get('relevance_score')
                    else:
                        url = url_data
                        relevance_score = None
                        
                    result = await self._crawl_single_page_async(url, config, output_format)
                    if relevance_score is not None:
                        result.relevance_score = relevance_score
                    return result
            
            # Create tasks for discovered URLs
            if discovered_data:
                tasks = [crawl_with_semaphore(url_data) for url_data in discovered_data[:config.max_pages]]
                
                # Execute with progress logging
                for i, coro in enumerate(asyncio.as_completed(tasks)):
                    result = await coro
                    results.append(result)
                    status = "✅" if result.success else "❌"
                    score_info = f" (score: {result.relevance_score:.3f})" if result.relevance_score else ""
                    logger.info(f"Seeded crawl {i+1}/{len(tasks)}: {status} {result.url}{score_info}")
            
            # Sort by relevance score if available
            if config.seeding_query and any(r.relevance_score is not None for r in results):
                results.sort(key=lambda x: x.relevance_score or 0, reverse=True)
                logger.info(f"📊 SEEDED RESULTS: Sorted {len(results)} results by relevance score")
            
            elapsed = time.time() - start_time
            logger.info(f"🎉 SEEDED CRAWL COMPLETE: {len([r for r in results if r.success])}/{len(results)} successful in {elapsed:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Seeded crawl failed for {domain}: {str(e)}")
            return [CrawlResult(
                url=domain_or_url,
                success=False,
                error_message=f"Seeded crawl failed: {str(e)}",
                processing_time=time.time() - start_time
            )]
    
    async def _crawl_single_page_async(
        self, 
        url: str, 
        config: CrawlConfig, 
        output_format: OutputFormat
    ) -> CrawlResult:
        """Crawl a single page asynchronously"""
        start_time = time.time()
        
        try:
            async with AsyncWebCrawler(
                always_bypass_cache=config.bypass_cache,
                headers=config.headers
            ) as crawler:
                
                # Prepare crawl parameters
                crawl_params = {
                    'url': url,
                    'word_count_threshold': config.word_count_threshold,
                    'exclude_external_links': config.exclude_external_links,
                    'exclude_social_media_links': config.exclude_social_media_links,
                    'bypass_cache': config.bypass_cache,
                    'delay_before_return_html': config.delay_before_return_html,
                    'remove_overlay_elements': config.remove_overlay_elements,
                    'simulate_user': config.simulate_user,
                    'override_navigator': config.override_navigator,
                }
                
                # Add extraction strategy if provided
                if config.extraction_strategy:
                    crawl_params['extraction_strategy'] = config.extraction_strategy
                    
                # Add chunking strategy if provided
                if config.chunking_strategy:
                    crawl_params['chunking_strategy'] = config.chunking_strategy
                
                # Add CSS selector if provided
                if config.css_selector:
                    crawl_params['css_selector'] = config.css_selector
                
                result = await crawler.arun(**crawl_params)
                
                if result.success:
                    # Extract content based on output format
                    content = self._format_content(result, output_format)
                    
                    return CrawlResult(
                        url=url,
                        success=True,
                        status_code=getattr(result, 'status_code', None),
                        html=result.html if output_format in [OutputFormat.HTML, OutputFormat.STRUCTURED] else None,
                        markdown=result.markdown if hasattr(result, 'markdown') else None,
                        extracted_content=content,
                        links=result.links.get('internal', []) + result.links.get('external', []) if hasattr(result, 'links') and result.links else [],
                        media=result.media.get('images', []) if hasattr(result, 'media') and result.media else [],
                        metadata=result.metadata if hasattr(result, 'metadata') else {},
                        processing_time=time.time() - start_time
                    )
                else:
                    return CrawlResult(
                        url=url,
                        success=False,
                        error_message=getattr(result, 'error_message', 'Unknown error'),
                        processing_time=time.time() - start_time
                    )
                    
        except Exception as e:
            logger.error(f"Error crawling {url}: {str(e)}")
            return CrawlResult(
                url=url,
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time
            )
    
    async def _crawl_multi_page_async(
        self, 
        start_url: str, 
        config: CrawlConfig, 
        output_format: OutputFormat
    ) -> List[CrawlResult]:
        """Crawl multiple pages starting from a URL (fallback method)"""
        discovered_urls = await self._discover_urls(start_url, config)
        
        results = []
        semaphore = asyncio.Semaphore(3)  # Limit concurrent requests
        
        async def crawl_with_semaphore(url):
            async with semaphore:
                return await self._crawl_single_page_async(url, config, output_format)
        
        # Create tasks for all URLs
        tasks = [crawl_with_semaphore(url) for url in discovered_urls[:config.max_pages]]
        
        # Execute with progress logging
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            result = await coro
            results.append(result)
            logger.info(f"Completed {i+1}/{len(tasks)}: {result.url} ({'✅' if result.success else '❌'})")
        
        return results
    
    async def _crawl_sitemap_async(
        self, 
        base_url: str, 
        config: CrawlConfig, 
        output_format: OutputFormat
    ) -> List[CrawlResult]:
        """Crawl URLs from sitemap"""
        sitemap_urls = await self._discover_sitemap_urls(base_url)
        
        results = []
        semaphore = asyncio.Semaphore(3)
        
        async def crawl_with_semaphore(url):
            async with semaphore:
                return await self._crawl_single_page_async(url, config, output_format)
        
        # Limit to max_pages
        urls_to_crawl = sitemap_urls[:config.max_pages]
        tasks = [crawl_with_semaphore(url) for url in urls_to_crawl]
        
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            result = await coro
            results.append(result)
            logger.info(f"Sitemap crawl {i+1}/{len(tasks)}: {result.url} ({'✅' if result.success else '❌'})")
        
        return results
    
    async def _discover_urls(self, start_url: str, config: CrawlConfig) -> List[str]:
        """Discover URLs from a starting page (fallback method)"""
        discovered = set([start_url])
        to_visit = [(start_url, 0)]  # (url, depth)
        base_domain = urlparse(start_url).netloc
        
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=config.timeout),
            headers=config.headers
        ) as session:
            
            while to_visit and len(discovered) < config.max_pages:
                current_url, depth = to_visit.pop(0)
                
                if depth >= config.max_depth:
                    continue
                
                try:
                    async with session.get(current_url) as response:
                        if response.status == 200 and 'text/html' in response.headers.get('content-type', ''):
                            html = await response.text()
                            soup = BeautifulSoup(html, 'html.parser')
                            
                            for link in soup.find_all('a', href=True):
                                href = link['href']
                                absolute_url = urljoin(current_url, href)
                                parsed = urlparse(absolute_url)
                                
                                # Filter URLs based on config
                                if (config.same_domain_only and parsed.netloc != base_domain):
                                    continue
                                    
                                if absolute_url not in discovered:
                                    discovered.add(absolute_url)
                                    if depth + 1 <= config.max_depth:
                                        to_visit.append((absolute_url, depth + 1))
                                
                except Exception as e:
                    logger.warning(f"Failed to discover URLs from {current_url}: {str(e)}")
                    continue
                
                # Small delay to be respectful
                await asyncio.sleep(0.5)
        
        return list(discovered)
    
    async def _discover_sitemap_urls(self, base_url: str) -> List[str]:
        """Discover URLs from sitemap.xml (fallback method)"""
        sitemap_urls = [
            urljoin(base_url, '/sitemap.xml'),
            urljoin(base_url, '/sitemap_index.xml'),
            urljoin(base_url, '/robots.txt')
        ]
        
        discovered_urls = []
        
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            headers=self.config.headers
        ) as session:
            
            for sitemap_url in sitemap_urls:
                try:
                    async with session.get(sitemap_url) as response:
                        if response.status == 200:
                            content = await response.text()
                            
                            if sitemap_url.endswith('robots.txt'):
                                # Extract sitemap URLs from robots.txt
                                for line in content.split('\n'):
                                    if line.lower().startswith('sitemap:'):
                                        sitemap_ref = line.split(':', 1)[1].strip()
                                        # Recursively fetch this sitemap
                                        sub_urls = await self._parse_sitemap(session, sitemap_ref)
                                        discovered_urls.extend(sub_urls)
                            else:
                                # Parse XML sitemap
                                urls = await self._parse_sitemap(session, sitemap_url, content)
                                discovered_urls.extend(urls)
                                
                except Exception as e:
                    logger.warning(f"Failed to fetch sitemap {sitemap_url}: {str(e)}")
                    continue
        
        return list(set(discovered_urls))  # Remove duplicates
    
    async def _parse_sitemap(self, session: aiohttp.ClientSession, sitemap_url: str, content: str = None) -> List[str]:
        """Parse sitemap XML and extract URLs"""
        if content is None:
            try:
                async with session.get(sitemap_url) as response:
                    if response.status != 200:
                        return []
                    content = await response.text()
            except:
                return []
        
        urls = []
        try:
            soup = BeautifulSoup(content, 'xml')
            
            # Handle sitemap index
            sitemap_tags = soup.find_all('sitemap')
            if sitemap_tags:
                for sitemap_tag in sitemap_tags:
                    loc = sitemap_tag.find('loc')
                    if loc:
                        sub_urls = await self._parse_sitemap(session, loc.text)
                        urls.extend(sub_urls)
            
            # Handle URL entries
            url_tags = soup.find_all('url')
            for url_tag in url_tags:
                loc = url_tag.find('loc')
                if loc:
                    urls.append(loc.text)
                    
        except Exception as e:
            logger.warning(f"Failed to parse sitemap {sitemap_url}: {str(e)}")
        
        return urls
    
    def _format_content(self, result, output_format: OutputFormat) -> str:
        """Format crawled content based on requested format with robust fallbacks"""
        # Try multiple attribute names in order of preference
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
            html_content = getattr(result, 'html', '') or getattr(result, 'cleaned_html', '') or ''
            return html_content
            
        elif output_format == OutputFormat.TEXT:
            # Extract text from any available content
            for attr in content_attributes:
                content = getattr(result, attr, None)
                if content and content.strip():
                    if attr == 'html' or 'html' in attr:
                        # If HTML, extract text
                        soup = BeautifulSoup(content, 'html.parser')
                        return soup.get_text(strip=True)
                    else:
                        return content.strip()
            return ''
            
        elif output_format == OutputFormat.JSON:
            # Build JSON with all available attributes
            json_data = {
                'url': getattr(result, 'url', ''),
                'title': getattr(result, 'title', ''),
                'success': getattr(result, 'success', False),
            }
            
            # Add first available content
            for attr in content_attributes:
                content = getattr(result, attr, None)
                if content and content.strip():
                    json_data['content'] = content.strip()
                    break
            
            # Add other attributes if available
            for attr_name in ['links', 'media', 'metadata']:
                attr_value = getattr(result, attr_name, {})
                if attr_value:
                    json_data[attr_name] = attr_value
                    
            return json.dumps(json_data, indent=2)
            
        elif output_format == OutputFormat.STRUCTURED:
            # Return structured data with all available content
            structured = {}
            for attr in content_attributes:
                content = getattr(result, attr, None)
                if content:
                    structured[attr] = content
            
            # Add metadata
            for attr_name in ['links', 'media', 'metadata', 'url', 'title', 'success']:
                attr_value = getattr(result, attr_name, None)
                if attr_value is not None:
                    structured[attr_name] = attr_value
                    
            return structured
            
        else:
            # Default: try to return any content we can find
            for attr in content_attributes:
                content = getattr(result, attr, None)
                if content and content.strip():
                    return content.strip()
            return ''
    
    def _update_stats(self, success: Union[bool, int], processing_time: float):
        """Update session statistics"""
        self.session_stats['total_requests'] += 1
        if isinstance(success, bool):
            if success:
                self.session_stats['successful_requests'] += 1
            else:
                self.session_stats['failed_requests'] += 1
        else:
            # success is count of successful requests
            self.session_stats['successful_requests'] += success
            self.session_stats['failed_requests'] += (1 - success) if success < 1 else 0
        
        self.session_stats['total_processing_time'] += processing_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        stats = self.session_stats.copy()
        stats['average_processing_time'] = (
            stats['total_processing_time'] / stats['total_requests'] 
            if stats['total_requests'] > 0 else 0
        )
        stats['success_rate'] = (
            stats['successful_requests'] / stats['total_requests'] 
            if stats['total_requests'] > 0 else 0
        )
        stats['official_seeding_available'] = OFFICIAL_SEEDING_AVAILABLE
        return stats
    
    def export_results(self, results: List[CrawlResult], filepath: str, format: str = 'json'):
        """Export crawl results to file"""
        data = [asdict(result) for result in results]
        
        filepath = Path(filepath)
        
        if format.lower() == 'json':
            with open(filepath.with_suffix('.json'), 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif format.lower() == 'csv':
            import pandas as pd
            df = pd.DataFrame(data)
            df.to_csv(filepath.with_suffix('.csv'), index=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Results exported to {filepath}")


# Convenience functions for quick usage
async def quick_crawl_async(
    url: str, 
    max_pages: int = 1,
    output_format: OutputFormat = OutputFormat.MARKDOWN,
    use_seeding: bool = None
) -> Union[CrawlResult, List[CrawlResult]]:
    """Quick async crawl function with optional seeding"""
    config = CrawlConfig(max_pages=max_pages)
    
    # Auto-detect seeding usage
    if use_seeding is None:
        use_seeding = OFFICIAL_SEEDING_AVAILABLE and max_pages > 1
    
    async with Crawl4AIClient(config) as client:
        if max_pages == 1:
            mode = CrawlMode.SINGLE_PAGE
        elif use_seeding and OFFICIAL_SEEDING_AVAILABLE:
            mode = CrawlMode.SEEDED
        else:
            mode = CrawlMode.MULTI_PAGE
            
        return await client.crawl_async(url, mode=mode, output_format=output_format)


def quick_crawl(
    url: str, 
    max_pages: int = 1,
    output_format: OutputFormat = OutputFormat.MARKDOWN,
    use_seeding: bool = None
) -> Union[CrawlResult, List[CrawlResult]]:
    """Quick sync crawl function with optional seeding"""
    return asyncio.run(quick_crawl_async(url, max_pages, output_format, use_seeding))


# Advanced seeding functions (when official implementation is available)
async def seeded_crawl_async(
    domain: str,
    query: Optional[str] = None,
    max_pages: int = 10,
    source: str = "sitemap+cc",
    score_threshold: Optional[float] = None,
    output_format: OutputFormat = OutputFormat.MARKDOWN
) -> List[CrawlResult]:
    """Advanced seeded crawling with query-based relevance scoring"""
    if not OFFICIAL_SEEDING_AVAILABLE:
        raise ValueError("Official seeding not available. Install crawl4ai with full dependencies.")
    
    config = CrawlConfig(
        max_pages=max_pages,
        seeding_source=source,
        seeding_extract_head=True,  # Required for relevance scoring
        seeding_query=query,
        seeding_score_threshold=score_threshold,
        seeding_concurrency=50,
        seeding_hits_per_sec=20
    )
    
    async with Crawl4AIClient(config) as client:
        return await client.crawl_async(domain, mode=CrawlMode.SEEDED, output_format=output_format)


def seeded_crawl(
    domain: str,
    query: Optional[str] = None,
    max_pages: int = 10,
    source: str = "sitemap+cc",
    score_threshold: Optional[float] = None,
    output_format: OutputFormat = OutputFormat.MARKDOWN
) -> List[CrawlResult]:
    """Sync version of advanced seeded crawling"""
    return asyncio.run(seeded_crawl_async(domain, query, max_pages, source, score_threshold, output_format))


# Example usage and testing
if __name__ == "__main__":
    async def test_client():
        """Test the Crawl4AI client with seeding"""
        print("🕷️ Testing Crawl4AI Client with Advanced Seeding")
        print(f"📊 Official seeding available: {OFFICIAL_SEEDING_AVAILABLE}")
        
        # Test single page crawl
        print("\n1. Testing single page crawl...")
        result = await quick_crawl_async("https://example.com")
        if isinstance(result, CrawlResult):
            print(f"✅ Success: {result.success}")
            print(f"📄 Content length: {len(result.extracted_content or '')}")
            print(f"⏱️ Processing time: {result.processing_time:.2f}s")
        
        # Test seeded crawl if available
        if OFFICIAL_SEEDING_AVAILABLE:
            print("\n2. Testing seeded crawl with query...")
            try:
                results = await seeded_crawl_async(
                    "example.com",
                    query="documentation tutorial guide",
                    max_pages=5,
                    score_threshold=0.1
                )
                
                if results:
                    successful = [r for r in results if r.success]
                    print(f"✅ Seeded crawl: {len(successful)}/{len(results)} successful")
                    
                    # Show relevance scores
                    for i, r in enumerate(successful[:3], 1):
                        score_info = f" (relevance: {r.relevance_score:.3f})" if r.relevance_score else ""
                        print(f"   {i}. {r.url}{score_info}")
                else:
                    print("⚠️ No results from seeded crawl")
            except Exception as e:
                print(f"❌ Seeded crawl failed: {str(e)}")
        else:
            print("\n2. Testing multi-page crawl (seeding not available)...")
            config = CrawlConfig(max_pages=3, max_depth=1)
            
            async with Crawl4AIClient(config) as client:
                results = await client.crawl_async(
                    "https://example.com", 
                    mode=CrawlMode.MULTI_PAGE
                )
                
                if isinstance(results, list):
                    successful = [r for r in results if r.success]
                    print(f"✅ Multi-page crawl: {len(successful)}/{len(results)} successful")
                    
                    # Show stats
                    stats = client.get_stats()
                    print(f"📊 Success rate: {stats['success_rate']:.1%}")
                    print(f"⏱️ Average processing time: {stats['average_processing_time']:.2f}s")
    
    # Run the test
    asyncio.run(test_client())
