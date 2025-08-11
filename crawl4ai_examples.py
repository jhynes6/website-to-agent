#!/usr/bin/env python3
"""
Crawl4AI Client - Practical Usage Examples

This file demonstrates real-world usage scenarios for the Crawl4AI client,
showing how to integrate it into various applications and workflows.

Based on the official Crawl4AI repository: https://github.com/unclecode/crawl4ai
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from src.crawl4ai_client import (
    Crawl4AIClient,
    CrawlConfig, 
    CrawlMode,
    OutputFormat,
    CrawlResult,
    quick_crawl,
    quick_crawl_async
)


# Example 1: Simple Content Extraction
async def example_simple_extraction():
    """Extract content from a single webpage"""
    print("📄 Example 1: Simple Content Extraction")
    print("-" * 40)
    
    # Quick and simple - just get the content
    result = await quick_crawl_async(
        "https://example.com",
        output_format=OutputFormat.MARKDOWN
    )
    
    if result.success:
        print(f"✅ Successfully extracted content from {result.url}")
        print(f"📊 Content length: {len(result.extracted_content or '')} characters")
        print(f"🔗 Found {len(result.links)} links")
        print(f"⏱️ Processing time: {result.processing_time:.2f}s")
        
        # Show content preview
        content = result.extracted_content or ''
        preview = content[:500] + "..." if len(content) > 500 else content
        print(f"\n📝 Content preview:\n{preview}")
    else:
        print(f"❌ Failed to extract content: {result.error_message}")


# Example 2: News Website Scraping
async def example_news_scraping():
    """Scrape multiple articles from a news website"""
    print("\n\n📰 Example 2: News Website Scraping")
    print("-" * 40)
    
    # Configure for news sites
    config = CrawlConfig(
        max_pages=5,
        max_depth=2,
        same_domain_only=True,
        css_selector="article, .article, .post, .news-item",  # Common news selectors
        exclude_external_links=True,
        delay_before_return_html=2.0,  # Allow dynamic content to load
        timeout=30
    )
    
    async with Crawl4AIClient(config) as client:
        # Example with a test site (replace with actual news site)
        results = await client.crawl_async(
            "https://httpbin.org",  # Replace with news site
            mode=CrawlMode.MULTI_PAGE,
            output_format=OutputFormat.MARKDOWN
        )
        
        if isinstance(results, list):
            articles = [r for r in results if r.success]
            print(f"✅ Scraped {len(articles)} articles")
            
            # Process each article
            for i, article in enumerate(articles[:3], 1):
                content = article.extracted_content or ''
                word_count = len(content.split())
                
                print(f"\n📰 Article {i}: {article.url}")
                print(f"   📊 Word count: {word_count}")
                print(f"   ⏱️ Processing time: {article.processing_time:.2f}s")
                
                # Extract title (first line or heading)
                lines = content.split('\n')
                title = next((line.strip('# ') for line in lines if line.strip()), 'No title')[:100]
                print(f"   📝 Title: {title}")


# Example 3: E-commerce Product Scraping
async def example_ecommerce_scraping():
    """Scrape product information from an e-commerce site"""
    print("\n\n🛒 Example 3: E-commerce Product Scraping")
    print("-" * 40)
    
    # Configure for product pages
    config = CrawlConfig(
        max_pages=10,
        css_selector=".product, .item, .product-info, .price, .description",
        exclude_external_links=True,
        same_domain_only=True,
        timeout=20
    )
    
    # Simulate product scraping (replace with actual e-commerce site)
    products = []
    
    async with Crawl4AIClient(config) as client:
        # Example URLs (replace with actual product URLs)
        product_urls = [
            "https://httpbin.org/html",  # Replace with product URLs
            "https://example.com",
        ]
        
        for url in product_urls:
            result = await client.crawl_async(
                url,
                output_format=OutputFormat.STRUCTURED
            )
            
            if result.success:
                # Extract product information
                content = result.extracted_content
                if isinstance(content, dict):
                    product_info = {
                        'url': result.url,
                        'title': extract_title(content.get('markdown', '')),
                        'content': content.get('markdown', ''),
                        'links': result.links,
                        'images': result.media,
                        'scraped_at': result.timestamp
                    }
                    products.append(product_info)
                    print(f"✅ Scraped product: {product_info['title'][:50]}...")
    
    print(f"\n📦 Total products scraped: {len(products)}")
    
    # Save products to JSON
    if products:
        with open('products.json', 'w') as f:
            json.dump(products, f, indent=2)
        print(f"💾 Saved products to products.json")


# Example 4: Documentation Site Crawling
async def example_documentation_crawling():
    """Crawl a documentation website for knowledge extraction"""
    print("\n\n📚 Example 4: Documentation Site Crawling")
    print("-" * 40)
    
    config = CrawlConfig(
        max_pages=15,
        max_depth=3,
        css_selector="main, .content, .documentation, .doc-content, article",
        exclude_external_links=True,
        same_domain_only=True,
        timeout=25
    )
    
    async with Crawl4AIClient(config) as client:
        # Example with a documentation site
        results = await client.crawl_async(
            "https://httpbin.org",  # Replace with docs site
            mode=CrawlMode.SITEMAP,  # Try sitemap first for better coverage
            output_format=OutputFormat.MARKDOWN
        )
        
        if isinstance(results, list):
            docs = [r for r in results if r.success]
            
            # Create a knowledge base structure
            knowledge_base = {
                'site_url': results[0].url if results else '',
                'scraped_at': datetime.now().isoformat(),
                'total_pages': len(docs),
                'sections': []
            }
            
            for doc in docs:
                content = doc.extracted_content or ''
                
                # Extract sections based on headers
                sections = extract_sections(content)
                
                page_info = {
                    'url': doc.url,
                    'title': extract_title(content),
                    'content': content,
                    'sections': sections,
                    'word_count': len(content.split()),
                    'links': doc.links[:10],  # Limit links
                }
                
                knowledge_base['sections'].append(page_info)
            
            print(f"✅ Created knowledge base with {len(docs)} documentation pages")
            
            # Save knowledge base
            with open('knowledge_base.json', 'w') as f:
                json.dump(knowledge_base, f, indent=2)
            print(f"💾 Saved knowledge base to knowledge_base.json")
            
            # Show statistics
            total_words = sum(page['word_count'] for page in knowledge_base['sections'])
            print(f"📊 Total words in knowledge base: {total_words:,}")


# Example 5: Competitive Analysis
async def example_competitive_analysis():
    """Analyze competitor websites"""
    print("\n\n🔍 Example 5: Competitive Analysis")
    print("-" * 40)
    
    competitors = [
        "https://example.com",
        "https://httpbin.org",
        # Add more competitor URLs
    ]
    
    config = CrawlConfig(
        max_pages=5,
        css_selector="main, .hero, .features, .pricing, .about",
        timeout=20
    )
    
    analysis_results = []
    
    async with Crawl4AIClient(config) as client:
        for competitor_url in competitors:
            print(f"\n🔍 Analyzing {competitor_url}...")
            
            results = await client.crawl_async(
                competitor_url,
                mode=CrawlMode.MULTI_PAGE,
                output_format=OutputFormat.TEXT
            )
            
            if isinstance(results, list):
                successful_pages = [r for r in results if r.success]
                
                # Analyze content
                all_content = ' '.join(r.extracted_content or '' for r in successful_pages)
                
                analysis = {
                    'url': competitor_url,
                    'pages_analyzed': len(successful_pages),
                    'total_content_length': len(all_content),
                    'word_count': len(all_content.split()),
                    'unique_links': len(set(link for r in successful_pages for link in r.links)),
                    'analysis_date': datetime.now().isoformat(),
                    'key_topics': extract_key_topics(all_content),
                    'sample_content': all_content[:1000]  # First 1000 chars
                }
                
                analysis_results.append(analysis)
                
                print(f"   ✅ Analyzed {analysis['pages_analyzed']} pages")
                print(f"   📊 Total words: {analysis['word_count']:,}")
                print(f"   🔗 Unique links: {analysis['unique_links']}")
    
    # Save competitive analysis
    if analysis_results:
        with open('competitive_analysis.json', 'w') as f:
            json.dump(analysis_results, f, indent=2)
        print(f"\n💾 Competitive analysis saved to competitive_analysis.json")
        
        # Print summary
        print(f"\n📈 Competitive Analysis Summary:")
        for analysis in analysis_results:
            print(f"   {analysis['url']}: {analysis['word_count']:,} words, {analysis['pages_analyzed']} pages")


# Example 6: Monitoring Website Changes
async def example_website_monitoring():
    """Monitor a website for changes"""
    print("\n\n📊 Example 6: Website Change Monitoring")
    print("-" * 40)
    
    monitor_url = "https://httpbin.org/html"
    
    # First crawl - baseline
    print("📸 Taking baseline snapshot...")
    baseline = await quick_crawl_async(monitor_url, output_format=OutputFormat.MARKDOWN)
    
    if not baseline.success:
        print(f"❌ Failed to get baseline: {baseline.error_message}")
        return
    
    # Save baseline
    baseline_file = 'website_baseline.json'
    with open(baseline_file, 'w') as f:
        json.dump({
            'url': baseline.url,
            'content': baseline.extracted_content,
            'content_hash': hash(baseline.extracted_content or ''),
            'links': baseline.links,
            'timestamp': baseline.timestamp,
            'content_length': len(baseline.extracted_content or '')
        }, f, indent=2)
    
    print(f"💾 Baseline saved: {len(baseline.extracted_content or '')} chars")
    
    # Simulate monitoring (in real scenario, this would run periodically)
    print("\n🔄 Simulating change detection...")
    await asyncio.sleep(2)  # Wait a bit
    
    # Second crawl - check for changes
    current = await quick_crawl_async(monitor_url, output_format=OutputFormat.MARKDOWN)
    
    if current.success:
        # Compare with baseline
        baseline_content = baseline.extracted_content or ''
        current_content = current.extracted_content or ''
        
        content_changed = hash(baseline_content) != hash(current_content)
        length_diff = len(current_content) - len(baseline_content)
        
        print(f"🔍 Change Detection Results:")
        print(f"   Content changed: {'Yes' if content_changed else 'No'}")
        print(f"   Length difference: {length_diff:+d} chars")
        print(f"   Links changed: {len(current.links) != len(baseline.links)}")
        
        if content_changed:
            print("   📝 Content has been modified!")
        else:
            print("   ✅ No changes detected")


# Helper functions
def extract_title(content: str) -> str:
    """Extract title from markdown content"""
    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('# '):
            return line[2:].strip()
        elif line.startswith('## '):
            return line[3:].strip()
    return "Untitled"


def extract_sections(content: str) -> List[Dict[str, str]]:
    """Extract sections from markdown content"""
    sections = []
    current_section = None
    current_content = []
    
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('#'):
            # Save previous section
            if current_section:
                sections.append({
                    'title': current_section,
                    'content': '\n'.join(current_content).strip()
                })
            
            # Start new section
            current_section = line.lstrip('#').strip()
            current_content = []
        else:
            current_content.append(line)
    
    # Save last section
    if current_section:
        sections.append({
            'title': current_section,
            'content': '\n'.join(current_content).strip()
        })
    
    return sections


def extract_key_topics(content: str) -> List[str]:
    """Extract key topics from content (simple word frequency analysis)"""
    import re
    from collections import Counter
    
    # Simple keyword extraction
    words = re.findall(r'\b[a-zA-Z]{4,}\b', content.lower())
    
    # Filter out common words
    stop_words = {'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'were', 'said', 'each', 'which', 'their', 'time', 'more', 'very', 'what', 'know', 'just', 'first', 'into', 'over', 'think', 'also', 'your', 'work', 'life', 'only', 'can', 'still', 'should', 'after', 'being', 'now', 'made', 'before', 'here', 'through', 'when', 'where', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'than', 'too', 'very', 'can', 'will', 'just', 'don', 'should', 'now'}
    
    filtered_words = [word for word in words if word not in stop_words and len(word) > 4]
    
    # Get top 10 most common words
    word_counts = Counter(filtered_words)
    return [word for word, count in word_counts.most_common(10)]


async def main():
    """Run all examples"""
    print("🕷️ Crawl4AI Client - Practical Examples")
    print("=" * 50)
    print("Demonstrating real-world usage scenarios...")
    print("Based on: https://github.com/unclecode/crawl4ai")
    
    try:
        await example_simple_extraction()
        await example_news_scraping()
        await example_ecommerce_scraping()
        await example_documentation_crawling()
        await example_competitive_analysis()
        await example_website_monitoring()
        
        print("\n\n🎉 All examples completed!")
        print("Check the generated JSON files for detailed results.")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Examples interrupted by user")
    except Exception as e:
        print(f"\n\n💥 Examples failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Run all examples
    asyncio.run(main())
