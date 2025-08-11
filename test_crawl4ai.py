#!/usr/bin/env python3
"""
Test script for the new Crawl4AI implementation.
"""
import asyncio
import sys
import os
import logging
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.url_seeder import discover_urls
from src.llms_text import extract_website_content

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_url_seeding():
    """Test the URL seeding functionality."""
    print("🌱 Testing URL Seeding...")
    print("=" * 50)
    
    test_url = "https://example.com"
    
    try:
        urls = await discover_urls(test_url, max_pages=5, max_depth=2)
        
        print(f"✅ URL Seeding Success!")
        print(f"📊 Discovered {len(urls)} URLs:")
        for i, url in enumerate(urls, 1):
            print(f"  {i}. {url}")
        
        return urls
    
    except Exception as e:
        print(f"❌ URL Seeding Failed: {e}")
        return []

async def test_content_extraction():
    """Test the content extraction functionality."""
    print("\n🕷️ Testing Content Extraction...")
    print("=" * 50)
    
    test_url = "https://example.com"
    
    try:
        start_time = datetime.now()
        
        content = await extract_website_content(
            url=test_url,
            max_urls=3,
            show_full_text=True
        )
        
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        print(f"✅ Content Extraction Success!")
        print(f"⏱️ Time taken: {elapsed:.2f} seconds")
        print(f"📊 Statistics:")
        print(f"  - URLs discovered: {len(content.get('discovered_urls', []))}")
        print(f"  - URLs processed: {len(content.get('processed_urls', []))}")
        print(f"  - URLs failed: {len(content.get('failed_urls', []))}")
        print(f"  - Content length: {len(content.get('llmstxt', ''))} characters")
        
        if content.get('processed_urls'):
            print(f"📄 Processed URLs:")
            for url in content['processed_urls']:
                print(f"  ✓ {url}")
        
        if content.get('failed_urls'):
            print(f"❌ Failed URLs:")
            for url in content['failed_urls']:
                print(f"  ✗ {url}")
        
        # Show content preview
        if content.get('llmstxt'):
            print(f"\n📝 Content Preview (first 500 chars):")
            print("-" * 50)
            print(content['llmstxt'][:500] + "..." if len(content['llmstxt']) > 500 else content['llmstxt'])
            print("-" * 50)
        
        return content
    
    except Exception as e:
        print(f"❌ Content Extraction Failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_full_workflow():
    """Test the complete workflow."""
    print("\n🔄 Testing Full Workflow...")
    print("=" * 50)
    
    # Test with a simple, reliable website
    test_urls = [
        "https://httpbin.org/html",  # Simple HTML page
        "https://example.com",       # Basic example site
    ]
    
    for test_url in test_urls:
        print(f"\n🎯 Testing with: {test_url}")
        print("-" * 30)
        
        try:
            # Step 1: URL Discovery
            print("Step 1: URL Discovery")
            urls = await discover_urls(test_url, max_pages=3, max_depth=1)
            print(f"  ✓ Discovered {len(urls)} URLs")
            
            # Step 2: Content Extraction
            print("Step 2: Content Extraction")
            content = await extract_website_content(test_url, max_urls=3)
            
            if content:
                success_rate = len(content['processed_urls']) / len(content['discovered_urls']) * 100 if content['discovered_urls'] else 0
                print(f"  ✓ Success rate: {success_rate:.1f}%")
                print(f"  ✓ Content extracted: {len(content['llmstxt'])} chars")
            else:
                print("  ❌ No content extracted")
        
        except Exception as e:
            print(f"  ❌ Failed: {e}")

def check_dependencies():
    """Check if all required dependencies are available."""
    print("🔍 Checking Dependencies...")
    print("=" * 50)
    
    dependencies = {
        'crawl4ai': 'AsyncWebCrawler',
        'aiohttp': 'ClientSession',
        'beautifulsoup4': 'BeautifulSoup',
        'asyncio': 'gather'
    }
    
    all_good = True
    
    for package, component in dependencies.items():
        try:
            if package == 'crawl4ai':
                from crawl4ai import AsyncWebCrawler
                print(f"  ✓ {package}: {component} available")
            elif package == 'aiohttp':
                import aiohttp
                print(f"  ✓ {package}: {component} available")
            elif package == 'beautifulsoup4':
                from bs4 import BeautifulSoup
                print(f"  ✓ {package}: {component} available")
            elif package == 'asyncio':
                import asyncio
                print(f"  ✓ {package}: {component} available")
        except ImportError as e:
            print(f"  ❌ {package}: Missing or error - {e}")
            all_good = False
    
    return all_good

async def main():
    """Main test function."""
    print("🧪 Crawl4AI Implementation Test Suite")
    print("=" * 60)
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Some dependencies are missing. Please install them first:")
        print("pip install crawl4ai aiohttp beautifulsoup4")
        return
    
    print("\n✅ All dependencies available!")
    
    # Run tests
    try:
        # Test URL seeding
        await test_url_seeding()
        
        # Test content extraction  
        await test_content_extraction()
        
        # Test full workflow
        await test_full_workflow()
        
        print("\n🎉 All tests completed!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the test suite
    asyncio.run(main())

