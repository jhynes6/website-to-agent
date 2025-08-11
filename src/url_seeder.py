import asyncio
import aiohttp
import logging
from typing import List, Set, Dict, Optional
from urllib.parse import urljoin, urlparse, urlunparse
from bs4 import BeautifulSoup
import time
from datetime import datetime

# Set up logging
logger = logging.getLogger('website-to-agent')

class URLSeeder:
    """
    Asynchronous URL seeding class to discover webpages from a given website.
    """
    
    def __init__(self, max_pages: int = 10, max_depth: int = 2, timeout: int = 30):
        """
        Initialize the URL seeder.
        
        Args:
            max_pages: Maximum number of pages to discover
            max_depth: Maximum crawl depth from the starting URL
            timeout: Request timeout in seconds
        """
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.timeout = timeout
        self.discovered_urls: Set[str] = set()
        self.visited_urls: Set[str] = set()
        self.failed_urls: Set[str] = set()
        
    async def seed_urls(self, start_url: str) -> List[str]:
        """
        Asynchronously discover URLs starting from the given URL.
        
        Args:
            start_url: The starting URL to begin discovery
            
        Returns:
            List of discovered URLs
        """
        logger.info(f"🌱 URL SEEDING START: Beginning discovery from {start_url}")
        logger.info(f"⚙️ Parameters: max_pages={self.max_pages}, max_depth={self.max_depth}")
        
        start_time = time.time()
        base_domain = self._get_base_domain(start_url)
        
        # Initialize with the starting URL
        self.discovered_urls.add(start_url)
        urls_to_process = [(start_url, 0)]  # (url, depth)
        
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers={
                'User-Agent': 'Mozilla/5.0 (compatible; WebToAgent/1.0; +https://github.com/example/website-to-agent)'
            }
        ) as session:
            
            while urls_to_process and len(self.discovered_urls) < self.max_pages:
                # Process URLs in batches for better performance
                batch_size = min(5, len(urls_to_process))
                current_batch = urls_to_process[:batch_size]
                urls_to_process = urls_to_process[batch_size:]
                
                # Create tasks for concurrent processing
                tasks = []
                for url, depth in current_batch:
                    if url not in self.visited_urls and depth <= self.max_depth:
                        task = self._process_url(session, url, depth, base_domain)
                        tasks.append(task)
                
                if tasks:
                    # Process batch concurrently
                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Collect new URLs from successful results
                    for result in batch_results:
                        if isinstance(result, list):
                            for new_url, new_depth in result:
                                if (new_url not in self.discovered_urls and 
                                    len(self.discovered_urls) < self.max_pages):
                                    self.discovered_urls.add(new_url)
                                    urls_to_process.append((new_url, new_depth))
                
                # Add small delay to be respectful to the server
                await asyncio.sleep(0.5)
        
        elapsed_time = time.time() - start_time
        discovered_count = len(self.discovered_urls)
        failed_count = len(self.failed_urls)
        
        logger.info(f"✅ URL SEEDING COMPLETE: Discovered {discovered_count} URLs in {elapsed_time:.2f}s")
        logger.info(f"📊 Stats: {len(self.visited_urls)} visited, {failed_count} failed")
        
        return list(self.discovered_urls)[:self.max_pages]
    
    async def _process_url(self, session: aiohttp.ClientSession, url: str, depth: int, base_domain: str) -> List[tuple]:
        """
        Process a single URL to extract links.
        
        Args:
            session: aiohttp session
            url: URL to process
            depth: Current crawl depth
            base_domain: Base domain to stay within
            
        Returns:
            List of (url, depth) tuples for newly discovered URLs
        """
        if url in self.visited_urls:
            return []
            
        self.visited_urls.add(url)
        logger.debug(f"🔍 Processing URL: {url} (depth: {depth})")
        
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"⚠️ HTTP {response.status} for {url}")
                    self.failed_urls.add(url)
                    return []
                
                # Only process HTML content
                content_type = response.headers.get('content-type', '').lower()
                if 'text/html' not in content_type:
                    logger.debug(f"⏭️ Skipping non-HTML content: {url}")
                    return []
                
                html_content = await response.text()
                return self._extract_links(html_content, url, depth, base_domain)
                
        except asyncio.TimeoutError:
            logger.warning(f"⏰ Timeout processing {url}")
            self.failed_urls.add(url)
            return []
        except Exception as e:
            logger.warning(f"❌ Error processing {url}: {str(e)}")
            self.failed_urls.add(url)
            return []
    
    def _extract_links(self, html_content: str, current_url: str, depth: int, base_domain: str) -> List[tuple]:
        """
        Extract valid links from HTML content.
        
        Args:
            html_content: HTML content to parse
            current_url: Current URL being processed
            depth: Current crawl depth
            base_domain: Base domain to stay within
            
        Returns:
            List of (url, depth) tuples for newly discovered URLs
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            new_urls = []
            
            # Find all links
            for link in soup.find_all('a', href=True):
                href = link['href'].strip()
                if not href or href.startswith('#'):
                    continue
                
                # Convert relative URLs to absolute
                absolute_url = urljoin(current_url, href)
                
                # Validate and filter URLs
                if self._is_valid_url(absolute_url, base_domain):
                    new_urls.append((absolute_url, depth + 1))
            
            logger.debug(f"📎 Found {len(new_urls)} valid links on {current_url}")
            return new_urls
            
        except Exception as e:
            logger.warning(f"❌ Error extracting links from {current_url}: {str(e)}")
            return []
    
    def _is_valid_url(self, url: str, base_domain: str) -> bool:
        """
        Check if a URL is valid for crawling.
        
        Args:
            url: URL to validate
            base_domain: Base domain to stay within
            
        Returns:
            True if URL is valid for crawling
        """
        try:
            parsed = urlparse(url)
            
            # Must have http or https scheme
            if parsed.scheme not in ('http', 'https'):
                return False
            
            # Must be within the same domain
            if not parsed.netloc.endswith(base_domain):
                return False
            
            # Skip common non-content URLs
            path = parsed.path.lower()
            skip_extensions = ('.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
                             '.zip', '.rar', '.tar', '.gz', '.jpg', '.jpeg', '.png', '.gif',
                             '.mp4', '.avi', '.mov', '.mp3', '.wav', '.css', '.js')
            
            if any(path.endswith(ext) for ext in skip_extensions):
                return False
            
            # Skip common non-content paths
            skip_paths = ('/admin', '/login', '/logout', '/register', '/cart', '/checkout',
                         '/api/', '/wp-admin', '/wp-login', '/.well-known')
            
            if any(path.startswith(skip_path) for skip_path in skip_paths):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _get_base_domain(self, url: str) -> str:
        """
        Extract the base domain from a URL.
        
        Args:
            url: URL to extract domain from
            
        Returns:
            Base domain
        """
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except Exception:
            return ""
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get crawling statistics.
        
        Returns:
            Dictionary with crawling statistics
        """
        return {
            'discovered': len(self.discovered_urls),
            'visited': len(self.visited_urls),
            'failed': len(self.failed_urls)
        }

# Convenience function for easy usage
async def discover_urls(start_url: str, max_pages: int = 10, max_depth: int = 2) -> List[str]:
    """
    Convenience function to discover URLs from a starting URL.
    
    Args:
        start_url: Starting URL for discovery
        max_pages: Maximum number of pages to discover
        max_depth: Maximum crawl depth
        
    Returns:
        List of discovered URLs
    """
    seeder = URLSeeder(max_pages=max_pages, max_depth=max_depth)
    return await seeder.seed_urls(start_url)

if __name__ == "__main__":
    # Example usage
    async def main():
        urls = await discover_urls("https://example.com", max_pages=5)
        print(f"Discovered URLs: {urls}")
    
    asyncio.run(main())

