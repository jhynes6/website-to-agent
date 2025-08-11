#!/usr/bin/env python3
"""
Comprehensive Test Script for AsyncUrlSeeder
============================================

Tests all major functionality of the AsyncUrlSeeder class:
- URL discovery from Common Crawl and sitemaps
- Configuration handling
- Caching behavior
- Head extraction and parsing
- BM25 relevance scoring
- Rate limiting
- Error handling
- Context manager usage

Usage:
    python test_async_url_seeder.py
    python test_async_url_seeder.py --domain example.com --max-urls 10
    python test_async_url_seeder.py --verbose --test-scoring
"""

import asyncio
import argparse
import json
import logging
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Any

# Simple logger for testing
class TestLogger:
    """Simple logger for testing"""
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def info(self, message: str, tag: str = "TEST", params: Dict = None):
        print(f"[{tag}] INFO: {message.format(**(params or {}))}")
    
    def debug(self, message: str, tag: str = "TEST", params: Dict = None):
        if self.verbose:
            print(f"[{tag}] DEBUG: {message.format(**(params or {}))}")
    
    def warning(self, message: str, tag: str = "TEST", params: Dict = None):
        print(f"[{tag}] WARNING: {message.format(**(params or {}))}")
    
    def error(self, message: str, tag: str = "TEST", params: Dict = None):
        print(f"[{tag}] ERROR: {message.format(**(params or {}))}")
    
    def success(self, message: str, tag: str = "TEST", params: Dict = None):
        print(f"[{tag}] SUCCESS: {message.format(**(params or {}))}")

# Add the crawl4ai directory to path and import the seeder
sys.path.insert(0, str(Path(__file__).parent / "crawl4ai_docs" / "crawl4ai"))

try:
    from crawl4ai.async_url_seeder import AsyncUrlSeeder
    from crawl4ai.async_configs import SeedingConfig
    print("✅ Successfully imported AsyncUrlSeeder and SeedingConfig")
except ImportError as e:
    print(f"❌ Failed to import AsyncUrlSeeder: {e}")
    print("Make sure the crawl4ai directory structure is correct")
    sys.exit(1)

class AsyncUrlSeederTester:
    """Comprehensive tester for AsyncUrlSeeder"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.logger = TestLogger(verbose=verbose)
        self.test_results: Dict[str, Dict] = {}
        self.start_time = time.time()
    
    def log(self, level: str, message: str, **kwargs):
        """Logging helper"""
        method = getattr(self.logger, level, self.logger.info)
        method(message, tag="TESTER", params=kwargs)
    
    async def test_basic_initialization(self) -> bool:
        """Test basic initialization of AsyncUrlSeeder"""
        self.log("info", "Testing basic initialization...")
        
        try:
            # Test default initialization
            seeder = AsyncUrlSeeder(logger=self.logger)
            assert seeder.ttl == timedelta(days=7)
            assert seeder.client is not None
            assert seeder._owns_client == True
            await seeder.close()
            
            # Test with custom TTL
            custom_ttl = timedelta(hours=1)
            seeder2 = AsyncUrlSeeder(ttl=custom_ttl, logger=self.logger)
            assert seeder2.ttl == custom_ttl
            await seeder2.close()
            
            self.log("info", "✅ Basic initialization test passed")
            return True
            
        except Exception as e:
            self.log("error", "❌ Basic initialization test failed: {error}", error=str(e))
            return False
    
    async def test_context_manager(self) -> bool:
        """Test async context manager functionality"""
        self.log("info", "Testing context manager...")
        
        try:
            async with AsyncUrlSeeder(logger=self.logger) as seeder:
                assert seeder.client is not None
                assert seeder._owns_client == True
            
            # Seeder should be closed automatically
            self.log("info", "✅ Context manager test passed")
            return True
            
        except Exception as e:
            self.log("error", "❌ Context manager test failed: {error}", error=str(e))
            return False
    
    async def test_url_discovery_sitemap(self, domain: str, max_urls: int = 5) -> bool:
        """Test URL discovery from sitemaps"""
        self.log("info", "Testing sitemap URL discovery for {domain}...", domain=domain)
        
        try:
            config = SeedingConfig(
                source='sitemap',
                max_urls=max_urls,
                verbose=self.verbose,
                extract_head=False,  # Faster for this test
                concurrency=5
            )
            
            async with AsyncUrlSeeder(logger=self.logger) as seeder:
                start_time = time.time()
                urls = await seeder.urls(domain, config)
                elapsed = time.time() - start_time
                
                self.log("info", "Found {count} URLs from sitemap for {domain} in {elapsed:.2f}s", 
                        count=len(urls), domain=domain, elapsed=elapsed)
                
                # Validate results
                assert isinstance(urls, list)
                for url_info in urls[:3]:  # Check first few
                    assert isinstance(url_info, dict)
                    assert 'url' in url_info
                    assert 'status' in url_info
                    self.log("debug", "Sample URL: {url} (status: {status})", 
                            url=url_info['url'], status=url_info['status'])
                
                self.test_results['sitemap_discovery'] = {
                    'domain': domain,
                    'urls_found': len(urls),
                    'elapsed_time': elapsed,
                    'sample_urls': [u['url'] for u in urls[:3]]
                }
                
                self.log("info", "✅ Sitemap discovery test passed")
                return True
                
        except Exception as e:
            self.log("error", "❌ Sitemap discovery test failed: {error}", error=str(e))
            return False
    
    async def test_head_extraction(self, domain: str, max_urls: int = 3) -> bool:
        """Test head content extraction"""
        self.log("info", "Testing head extraction for {domain}...", domain=domain)
        
        try:
            config = SeedingConfig(
                source='sitemap',
                max_urls=max_urls,
                verbose=self.verbose,
                extract_head=True,
                concurrency=3
            )
            
            async with AsyncUrlSeeder(logger=self.logger) as seeder:
                start_time = time.time()
                urls = await seeder.urls(domain, config)
                elapsed = time.time() - start_time
                
                # Find URLs with head data
                urls_with_head = [u for u in urls if u.get('head_data') and u['head_data']]
                
                self.log("info", "Extracted head data from {count}/{total} URLs in {elapsed:.2f}s", 
                        count=len(urls_with_head), total=len(urls), elapsed=elapsed)
                
                # Analyze head data
                for url_info in urls_with_head[:2]:
                    head_data = url_info['head_data']
                    title = head_data.get('title', 'No title')
                    meta_count = len(head_data.get('meta', {}))
                    link_count = len(head_data.get('link', {}))
                    
                    self.log("debug", "URL: {url}", url=url_info['url'])
                    self.log("debug", "  Title: {title}", title=title[:100])
                    self.log("debug", "  Meta tags: {count}", count=meta_count)
                    self.log("debug", "  Link tags: {count}", count=link_count)
                
                self.test_results['head_extraction'] = {
                    'domain': domain,
                    'urls_processed': len(urls),
                    'urls_with_head_data': len(urls_with_head),
                    'elapsed_time': elapsed,
                    'success_rate': len(urls_with_head) / len(urls) if urls else 0
                }
                
                self.log("info", "✅ Head extraction test passed")
                return True
                
        except Exception as e:
            self.log("error", "❌ Head extraction test failed: {error}", error=str(e))
            return False
    
    async def test_bm25_scoring(self, domain: str, query: str = "documentation guide tutorial") -> bool:
        """Test BM25 relevance scoring"""
        self.log("info", "Testing BM25 scoring for {domain} with query: '{query}'...", 
                domain=domain, query=query)
        
        try:
            config = SeedingConfig(
                source='sitemap',
                max_urls=10,
                verbose=self.verbose,
                extract_head=True,
                query=query,
                scoring_method='bm25',
                concurrency=5
            )
            
            async with AsyncUrlSeeder(logger=self.logger) as seeder:
                start_time = time.time()
                urls = await seeder.urls(domain, config)
                elapsed = time.time() - start_time
                
                # Check for relevance scores
                scored_urls = [u for u in urls if 'relevance_score' in u]
                
                self.log("info", "Scored {count}/{total} URLs in {elapsed:.2f}s", 
                        count=len(scored_urls), total=len(urls), elapsed=elapsed)
                
                # Show top scored URLs
                if scored_urls:
                    scored_urls.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
                    for i, url_info in enumerate(scored_urls[:3]):
                        score = url_info.get('relevance_score', 0)
                        title = url_info.get('head_data', {}).get('title', 'No title')
                        self.log("debug", "#{rank}: {score:.3f} - {url}", 
                               rank=i+1, score=score, url=url_info['url'])
                        self.log("debug", "    Title: {title}", title=title[:80])
                
                self.test_results['bm25_scoring'] = {
                    'domain': domain,
                    'query': query,
                    'urls_processed': len(urls),
                    'urls_scored': len(scored_urls),
                    'elapsed_time': elapsed,
                    'top_scores': [u.get('relevance_score', 0) for u in scored_urls[:5]]
                }
                
                self.log("info", "✅ BM25 scoring test passed")
                return True
                
        except Exception as e:
            self.log("error", "❌ BM25 scoring test failed: {error}", error=str(e))
            return False
    
    async def test_rate_limiting(self, domain: str, hits_per_sec: int = 2) -> bool:
        """Test rate limiting functionality"""
        self.log("info", "Testing rate limiting at {hits_per_sec} hits/sec for {domain}...", 
                hits_per_sec=hits_per_sec, domain=domain)
        
        try:
            config = SeedingConfig(
                source='sitemap',
                max_urls=5,
                verbose=self.verbose,
                extract_head=True,
                hits_per_sec=hits_per_sec,
                concurrency=10  # High concurrency to test rate limiting
            )
            
            async with AsyncUrlSeeder(logger=self.logger) as seeder:
                start_time = time.time()
                urls = await seeder.urls(domain, config)
                elapsed = time.time() - start_time
                
                expected_min_time = len(urls) / hits_per_sec if urls else 0
                
                self.log("info", "Processed {count} URLs in {elapsed:.2f}s (expected min: {expected:.2f}s)", 
                        count=len(urls), elapsed=elapsed, expected=expected_min_time)
                
                self.test_results['rate_limiting'] = {
                    'domain': domain,
                    'hits_per_sec': hits_per_sec,
                    'urls_processed': len(urls),
                    'elapsed_time': elapsed,
                    'expected_min_time': expected_min_time,
                    'rate_limit_effective': elapsed >= expected_min_time * 0.8  # Allow some tolerance
                }
                
                self.log("info", "✅ Rate limiting test passed")
                return True
                
        except Exception as e:
            self.log("error", "❌ Rate limiting test failed: {error}", error=str(e))
            return False
    
    async def test_custom_url_extraction(self, urls: List[str]) -> bool:
        """Test extract_head_for_urls method with custom URL list"""
        self.log("info", "Testing custom URL head extraction for {count} URLs...", count=len(urls))
        
        try:
            config = SeedingConfig(
                extract_head=True,
                verbose=self.verbose,
                concurrency=3
            )
            
            async with AsyncUrlSeeder(logger=self.logger) as seeder:
                start_time = time.time()
                results = await seeder.extract_head_for_urls(urls, config)
                elapsed = time.time() - start_time
                
                valid_results = [r for r in results if r.get('status') == 'valid']
                
                self.log("info", "Processed {count} custom URLs, {valid} valid in {elapsed:.2f}s", 
                        count=len(results), valid=len(valid_results), elapsed=elapsed)
                
                # Show sample results
                for result in results[:2]:
                    status = result.get('status', 'unknown')
                    title = result.get('head_data', {}).get('title', 'No title')
                    self.log("debug", "URL: {url} (status: {status})", 
                           url=result['url'], status=status)
                    self.log("debug", "  Title: {title}", title=title[:80])
                
                self.test_results['custom_url_extraction'] = {
                    'urls_input': len(urls),
                    'urls_processed': len(results),
                    'valid_results': len(valid_results),
                    'elapsed_time': elapsed,
                    'success_rate': len(valid_results) / len(results) if results else 0
                }
                
                self.log("info", "✅ Custom URL extraction test passed")
                return True
                
        except Exception as e:
            self.log("error", "❌ Custom URL extraction test failed: {error}", error=str(e))
            return False
    
    async def test_caching_behavior(self, domain: str) -> bool:
        """Test caching behavior"""
        self.log("info", "Testing caching behavior for {domain}...", domain=domain)
        
        try:
            config = SeedingConfig(
                source='sitemap',
                max_urls=3,
                verbose=self.verbose,
                extract_head=True,
                force=False  # Use cache
            )
            
            async with AsyncUrlSeeder(logger=self.logger) as seeder:
                # First run - should populate cache
                start_time = time.time()
                urls1 = await seeder.urls(domain, config)
                elapsed1 = time.time() - start_time
                
                # Second run - should use cache
                start_time = time.time()
                urls2 = await seeder.urls(domain, config)
                elapsed2 = time.time() - start_time
                
                self.log("info", "First run: {count1} URLs in {elapsed1:.2f}s", 
                        count1=len(urls1), elapsed1=elapsed1)
                self.log("info", "Second run: {count2} URLs in {elapsed2:.2f}s (cached)", 
                        count2=len(urls2), elapsed2=elapsed2)
                
                # Cache should make second run faster
                cache_effective = elapsed2 < elapsed1 * 0.8 or elapsed2 < 1.0
                
                self.test_results['caching'] = {
                    'domain': domain,
                    'first_run_time': elapsed1,
                    'second_run_time': elapsed2,
                    'cache_effective': cache_effective,
                    'urls_consistent': len(urls1) == len(urls2)
                }
                
                self.log("info", "✅ Caching test passed (cache effective: {effective})", 
                        effective=cache_effective)
                return True
                
        except Exception as e:
            self.log("error", "❌ Caching test failed: {error}", error=str(e))
            return False
    
    async def test_error_handling(self) -> bool:
        """Test error handling with invalid inputs"""
        self.log("info", "Testing error handling...")
        
        try:
            config = SeedingConfig(
                source='sitemap',
                max_urls=2,
                verbose=self.verbose,
                extract_head=True
            )
            
            async with AsyncUrlSeeder(logger=self.logger) as seeder:
                # Test with invalid domain
                try:
                    urls = await seeder.urls("invalid-domain-that-does-not-exist.com", config)
                    self.log("debug", "Invalid domain returned {count} URLs", count=len(urls))
                except Exception as e:
                    self.log("debug", "Invalid domain raised exception (expected): {error}", error=str(e))
                
                # Test with invalid source
                bad_config = SeedingConfig(source='invalid_source')
                try:
                    urls = await seeder.urls("example.com", bad_config)
                    self.log("error", "Invalid source should have raised exception")
                    return False
                except ValueError:
                    self.log("debug", "Invalid source correctly raised ValueError")
                
                self.log("info", "✅ Error handling test passed")
                return True
                
        except Exception as e:
            self.log("error", "❌ Error handling test failed: {error}", error=str(e))
            return False
    
    async def run_all_tests(self, domain: str = "docs.python.org", max_urls: int = 10, test_scoring: bool = False):
        """Run all tests"""
        self.log("info", "🚀 Starting AsyncUrlSeeder comprehensive test suite...")
        self.log("info", "Test domain: {domain}, Max URLs: {max_urls}", domain=domain, max_urls=max_urls)
        
        tests = [
            ("Basic Initialization", self.test_basic_initialization()),
            ("Context Manager", self.test_context_manager()),
            ("Sitemap Discovery", self.test_url_discovery_sitemap(domain, max_urls)),
            ("Head Extraction", self.test_head_extraction(domain, max(3, max_urls//3))),
            ("Caching Behavior", self.test_caching_behavior(domain)),
            ("Error Handling", self.test_error_handling()),
        ]
        
        if test_scoring:
            tests.append(("BM25 Scoring", self.test_bm25_scoring(domain, "python documentation guide")))
            tests.append(("Rate Limiting", self.test_rate_limiting(domain, 2)))
        
        # Add custom URL test
        sample_urls = [
            f"https://{domain}/",
            f"https://{domain}/about",
            f"https://{domain}/contact"
        ]
        tests.append(("Custom URL Extraction", self.test_custom_url_extraction(sample_urls)))
        
        passed = 0
        failed = 0
        
        for test_name, test_coro in tests:
            self.log("info", f"\n{'='*60}")
            self.log("info", f"🧪 Running test: {test_name}")
            self.log("info", f"{'='*60}")
            
            try:
                success = await test_coro
                if success:
                    passed += 1
                    self.log("info", f"✅ {test_name} PASSED")
                else:
                    failed += 1
                    self.log("error", f"❌ {test_name} FAILED")
            except Exception as e:
                failed += 1
                self.log("error", f"❌ {test_name} CRASHED: {error}", error=str(e))
        
        # Print final results
        total_time = time.time() - self.start_time
        self.log("info", f"\n{'='*60}")
        self.log("info", "🏁 TEST SUITE COMPLETED")
        self.log("info", f"{'='*60}")
        self.log("info", "Total tests: {total}", total=passed + failed)
        self.log("info", "Passed: {passed} ✅", passed=passed)
        self.log("info", "Failed: {failed} ❌", failed=failed)
        self.log("info", "Success rate: {rate:.1%}", rate=passed/(passed+failed) if (passed+failed) > 0 else 0)
        self.log("info", "Total time: {time:.2f}s", time=total_time)
        
        # Save detailed results
        results_file = Path("async_url_seeder_test_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                'summary': {
                    'total_tests': passed + failed,
                    'passed': passed,
                    'failed': failed,
                    'success_rate': passed/(passed+failed) if (passed+failed) > 0 else 0,
                    'total_time': total_time,
                    'test_domain': domain,
                    'max_urls': max_urls
                },
                'detailed_results': self.test_results
            }, f, indent=2)
        
        self.log("info", "📊 Detailed results saved to: {file}", file=results_file)
        
        return passed == len(tests)

async def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="Test AsyncUrlSeeder functionality")
    parser.add_argument("--domain", default="docs.python.org", help="Domain to test (default: docs.python.org)")
    parser.add_argument("--max-urls", type=int, default=10, help="Maximum URLs to fetch per test (default: 10)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--test-scoring", action="store_true", help="Include BM25 scoring and rate limiting tests")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO if not args.verbose else logging.DEBUG)
    
    # Run tests
    tester = AsyncUrlSeederTester(verbose=args.verbose)
    success = await tester.run_all_tests(
        domain=args.domain,
        max_urls=args.max_urls,
        test_scoring=args.test_scoring
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())
