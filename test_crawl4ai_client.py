#!/usr/bin/env python3
"""
Test script for the Crawl4AI Client

This script demonstrates various usage patterns of the Crawl4AI client,
including single page crawling, multi-page crawling, sitemap crawling,
and advanced features like extraction strategies.
"""

import asyncio
import json
import time
from pathlib import Path

# Import our custom client
from src.crawl4ai_client import (
    Crawl4AIClient, 
    CrawlConfig, 
    CrawlMode, 
    OutputFormat,
    quick_crawl,
    quick_crawl_async
)

# Import Crawl4AI strategies for advanced examples
try:
    from crawl4ai.extraction_strategy import LLMExtractionStrategy, JsonCssExtractionStrategy
    from crawl4ai.chunking_strategy import RegexChunking, NlpSentenceChunking
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
    print("⚠️ Advanced extraction strategies not available. Install crawl4ai with full dependencies.")
    ADVANCED_FEATURES_AVAILABLE = False


async def test_basic_crawling():
    """Test basic crawling functionality"""
    print("🔍 Testing Basic Crawling")
    print("=" * 50)
    
    # Test 1: Quick single page crawl
    print("\n1. Quick single page crawl (async)...")
    start_time = time.time()
    result = await quick_crawl_async("https://example.com")
    elapsed = time.time() - start_time
    
    if result.success:
        print(f"✅ Success! Crawled in {elapsed:.2f}s")
        print(f"📄 Content length: {len(result.extracted_content or '')} chars")
        print(f"🔗 Links found: {len(result.links)}")
        print(f"📸 Media found: {len(result.media)}")
        print(f"📊 Status code: {result.status_code}")
    else:
        print(f"❌ Failed: {result.error_message}")
    
    # Test 2: Quick single page crawl (async)
    print("\n2. Quick single page crawl (async)...")
    start_time = time.time()
    result = await quick_crawl_async("https://httpbin.org/html", output_format=OutputFormat.HTML)
    elapsed = time.time() - start_time
    
    if result.success:
        print(f"✅ Success! Crawled in {elapsed:.2f}s")
        print(f"📄 HTML length: {len(result.html or '')} chars")
        print(f"📝 Markdown length: {len(result.markdown or '')} chars")
    else:
        print(f"❌ Failed: {result.error_message}")


async def test_multi_page_crawling():
    """Test multi-page crawling"""
    print("\n\n🌐 Testing Multi-Page Crawling")
    print("=" * 50)
    
    # Configure for multi-page crawling
    config = CrawlConfig(
        max_pages=5,
        max_depth=2,
        same_domain_only=True,
        delay_before_return_html=1.0,  # Faster for testing
        timeout=20
    )
    
    async with Crawl4AIClient(config) as client:
        print(f"\n1. Multi-page crawl of example.com (max {config.max_pages} pages)...")
        start_time = time.time()
        
        results = await client.crawl_async(
            "https://example.com",
            mode=CrawlMode.MULTI_PAGE,
            output_format=OutputFormat.MARKDOWN
        )
        
        elapsed = time.time() - start_time
        
        if isinstance(results, list):
            successful = [r for r in results if r.success]
            failed = [r for r in results if not r.success]
            
            print(f"✅ Completed in {elapsed:.2f}s")
            print(f"📊 Success: {len(successful)}/{len(results)} pages")
            print(f"📄 Total content: {sum(len(r.extracted_content or '') for r in successful):,} chars")
            
            if failed:
                print(f"❌ Failed URLs:")
                for r in failed[:3]:  # Show first 3 failures
                    print(f"   • {r.url}: {r.error_message}")
            
            # Show crawling stats
            stats = client.get_stats()
            print(f"📈 Success rate: {stats['success_rate']:.1%}")
            print(f"⏱️ Avg processing time: {stats['average_processing_time']:.2f}s")
        else:
            print(f"❌ Multi-page crawl failed: {results.error_message}")


async def test_sitemap_crawling():
    """Test sitemap-based crawling"""
    print("\n\n🗺️ Testing Sitemap Crawling")
    print("=" * 50)
    
    config = CrawlConfig(
        max_pages=3,  # Limit for testing
        timeout=15
    )
    
    async with Crawl4AIClient(config) as client:
        print(f"\n1. Sitemap crawl (max {config.max_pages} pages)...")
        start_time = time.time()
        
        # Test with a site that likely has a sitemap
        results = await client.crawl_async(
            "https://httpbin.org",
            mode=CrawlMode.SITEMAP,
            output_format=OutputFormat.TEXT
        )
        
        elapsed = time.time() - start_time
        
        if isinstance(results, list) and results:
            successful = [r for r in results if r.success]
            print(f"✅ Found and crawled {len(successful)} URLs from sitemap in {elapsed:.2f}s")
            
            for i, result in enumerate(successful[:3], 1):
                content_preview = (result.extracted_content or '')[:100]
                print(f"   {i}. {result.url}: {len(result.extracted_content or '')} chars")
                print(f"      Preview: {content_preview}...")
        else:
            print("⚠️ No sitemap URLs found or all failed")


async def test_different_output_formats():
    """Test different output formats"""
    print("\n\n📝 Testing Output Formats")
    print("=" * 50)
    
    test_url = "https://httpbin.org/html"
    formats = [
        (OutputFormat.MARKDOWN, "Markdown"),
        (OutputFormat.HTML, "HTML"),
        (OutputFormat.TEXT, "Plain Text"),
        (OutputFormat.JSON, "JSON"),
        (OutputFormat.STRUCTURED, "Structured")
    ]
    
    for output_format, format_name in formats:
        print(f"\n{format_name} format...")
        result = await quick_crawl_async(test_url, output_format=output_format)
        
        if result.success:
            if output_format == OutputFormat.STRUCTURED:
                content_size = len(str(result.extracted_content))
            else:
                content_size = len(result.extracted_content or '')
            
            print(f"✅ {format_name}: {content_size} chars")
            
            # Show a preview for text formats
            if output_format in [OutputFormat.MARKDOWN, OutputFormat.TEXT, OutputFormat.JSON]:
                preview = (result.extracted_content or '')[:200]
                print(f"   Preview: {preview}...")
        else:
            print(f"❌ {format_name} failed: {result.error_message}")


async def test_advanced_features():
    """Test advanced extraction features"""
    if not ADVANCED_FEATURES_AVAILABLE:
        print("\n\n⚠️ Skipping Advanced Features (dependencies not available)")
        return
    
    print("\n\n🧠 Testing Advanced Features")
    print("=" * 50)
    
    # Test 1: CSS Selector extraction
    print("\n1. CSS Selector extraction...")
    config = CrawlConfig(
        css_selector="h1, h2, p",  # Extract only headings and paragraphs
        timeout=15
    )
    
    async with Crawl4AIClient(config) as client:
        result = await client.crawl_async(
            "https://httpbin.org/html",
            output_format=OutputFormat.MARKDOWN
        )
        
        if result.success:
            print(f"✅ CSS selector extraction: {len(result.extracted_content or '')} chars")
            preview = (result.extracted_content or '')[:300]
            print(f"   Preview: {preview}...")
        else:
            print(f"❌ CSS selector extraction failed: {result.error_message}")
    
    # Test 2: Custom chunking strategy
    print("\n2. Custom chunking strategy...")
    try:
        config = CrawlConfig(
            chunking_strategy=RegexChunking(patterns=[r'\n\n', r'\. ']),
            timeout=15
        )
        
        async with Crawl4AIClient(config) as client:
            result = await client.crawl_async(
                "https://httpbin.org/html",
                output_format=OutputFormat.TEXT
            )
            
            if result.success:
                print(f"✅ Custom chunking: {len(result.extracted_content or '')} chars")
            else:
                print(f"❌ Custom chunking failed: {result.error_message}")
    except Exception as e:
        print(f"⚠️ Custom chunking not available: {str(e)}")


async def test_error_handling():
    """Test error handling and edge cases"""
    print("\n\n🚨 Testing Error Handling")
    print("=" * 50)
    
    error_tests = [
        ("Invalid URL", "not-a-valid-url"),
        ("Non-existent domain", "https://this-domain-definitely-does-not-exist-12345.com"),
        ("404 Page", "https://httpbin.org/status/404"),
        ("Timeout", "https://httpbin.org/delay/30"),  # Will timeout with default settings
    ]
    
    config = CrawlConfig(timeout=5)  # Short timeout for testing
    
    async with Crawl4AIClient(config) as client:
        for test_name, test_url in error_tests:
            print(f"\n{test_name}: {test_url}")
            start_time = time.time()
            
            result = await client.crawl_async(test_url)
            elapsed = time.time() - start_time
            
            if result.success:
                print(f"✅ Unexpectedly succeeded in {elapsed:.2f}s")
            else:
                print(f"❌ Failed as expected in {elapsed:.2f}s: {result.error_message}")


async def test_export_functionality():
    """Test result export functionality"""
    print("\n\n💾 Testing Export Functionality")
    print("=" * 50)
    
    # Crawl a few pages
    config = CrawlConfig(max_pages=3, max_depth=1)
    
    async with Crawl4AIClient(config) as client:
        results = await client.crawl_async(
            "https://httpbin.org",
            mode=CrawlMode.MULTI_PAGE
        )
        
        if isinstance(results, list) and results:
            # Test JSON export
            json_file = "test_results.json"
            try:
                client.export_results(results, json_file, format='json')
                
                # Verify the file was created
                if Path(json_file).exists():
                    file_size = Path(json_file).stat().st_size
                    print(f"✅ JSON export successful: {json_file} ({file_size} bytes)")
                    
                    # Clean up
                    Path(json_file).unlink()
                else:
                    print(f"❌ JSON export failed: file not created")
                    
            except Exception as e:
                print(f"❌ JSON export failed: {str(e)}")
        else:
            print("⚠️ No results to export")


async def test_performance_stats():
    """Test performance statistics tracking"""
    print("\n\n📊 Testing Performance Statistics")
    print("=" * 50)
    
    config = CrawlConfig(max_pages=5, timeout=10)
    
    async with Crawl4AIClient(config) as client:
        # Perform multiple crawls
        test_urls = [
            "https://httpbin.org/html",
            "https://httpbin.org/json",
            "https://example.com",
        ]
        
        print(f"\nCrawling {len(test_urls)} URLs...")
        
        for url in test_urls:
            result = await client.crawl_async(url)
            status = "✅" if result.success else "❌"
            print(f"   {status} {url}")
        
        # Show comprehensive stats
        stats = client.get_stats()
        print(f"\n📈 Session Statistics:")
        print(f"   Total requests: {stats['total_requests']}")
        print(f"   Successful: {stats['successful_requests']}")
        print(f"   Failed: {stats['failed_requests']}")
        print(f"   Success rate: {stats['success_rate']:.1%}")
        print(f"   Total time: {stats['total_processing_time']:.2f}s")
        print(f"   Average time: {stats['average_processing_time']:.2f}s")
        print(f"   Session started: {stats['session_start']}")


async def main():
    """Run all tests"""
    print("🕷️ Crawl4AI Client Test Suite")
    print("=" * 60)
    print("Testing comprehensive Crawl4AI client functionality...")
    print("Based on: https://github.com/unclecode/crawl4ai")
    
    try:
        await test_basic_crawling()
        await test_multi_page_crawling()
        await test_sitemap_crawling()
        await test_different_output_formats()
        await test_advanced_features()
        await test_error_handling()
        await test_export_functionality()
        await test_performance_stats()
        
        print("\n\n🎉 All tests completed!")
        print("The Crawl4AI client is ready for use.")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Tests interrupted by user")
    except Exception as e:
        print(f"\n\n💥 Test suite failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Run the comprehensive test suite
    asyncio.run(main())
