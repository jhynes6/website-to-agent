# Crawl4AI Client

A comprehensive Python client for the [Crawl4AI](https://github.com/unclecode/crawl4ai) web crawler, providing both synchronous and asynchronous interfaces with advanced crawling capabilities including **official async URL seeding**.

## 🚀 Features

### Core Capabilities
- **Single Page Crawling**: Extract content from individual web pages
- **Multi-Page Crawling**: Discover and crawl related pages automatically
- **Sitemap Crawling**: Parse sitemaps for comprehensive site coverage
- **Official Async URL Seeding**: Integration with Crawl4AI's advanced URL discovery system
- **Multiple Output Formats**: Markdown, HTML, Text, JSON, and Structured data
- **Advanced Extraction**: CSS selectors, custom strategies, and content filtering
- **Performance Monitoring**: Built-in statistics and progress tracking
- **Export Capabilities**: Save results to JSON and CSV formats

### Advanced URL Seeding (Official Integration)
When the official Crawl4AI async seeding is available, the client provides:
- **Common Crawl Integration**: Discover URLs from Common Crawl archives
- **Sitemap Discovery**: Automatic sitemap parsing and URL extraction
- **BM25 Relevance Scoring**: Query-based URL ranking and filtering
- **Live URL Verification**: Optional HEAD requests to verify URL availability
- **Metadata Extraction**: Extract page titles, descriptions, and other metadata
- **Rate Limiting**: Configurable request rate limiting to be respectful
- **Caching**: Intelligent caching to avoid redundant requests

## 📦 Installation

```bash
# Install basic dependencies
pip install crawl4ai aiohttp beautifulsoup4

# For advanced seeding features (optional)
pip install crawl4ai[all]  # Includes all optional dependencies
```

## 🔧 Quick Start

### Simple Single Page Crawling

```python
import asyncio
from src.crawl4ai_client import quick_crawl_async, OutputFormat

async def main():
    # Quick async crawl
    result = await quick_crawl_async("https://example.com")
    
    if result.success:
        print(f"✅ Successfully crawled {result.url}")
        print(f"📄 Content: {len(result.extracted_content)} characters")
        print(f"🔗 Links: {len(result.links)} found")
    else:
        print(f"❌ Failed: {result.error_message}")

asyncio.run(main())
```

### Multi-Page Crawling with Seeding

```python
import asyncio
from src.crawl4ai_client import seeded_crawl_async

async def main():
    # Advanced seeded crawl with query-based relevance
    results = await seeded_crawl_async(
        domain="python.org",
        query="tutorial documentation guide",
        max_pages=10,
        source="sitemap+cc",  # Use both sitemap and Common Crawl
        score_threshold=0.3   # Minimum relevance score
    )
    
    for result in results:
        if result.success:
            score = result.relevance_score or 0
            print(f"✅ {result.url} (relevance: {score:.3f})")
        else:
            print(f"❌ {result.url}: {result.error_message}")

asyncio.run(main())
```

## 📖 Comprehensive Usage Guide

### 1. Basic Configuration

```python
from src.crawl4ai_client import Crawl4AIClient, CrawlConfig, CrawlMode, OutputFormat

# Configure crawling behavior
config = CrawlConfig(
    # Basic settings
    max_pages=20,
    max_depth=3,
    timeout=30,
    
    # Content filtering
    css_selector="article, .content, .post",
    exclude_external_links=True,
    
    # Official seeding settings (when available)
    seeding_source="sitemap+cc",  # "sitemap", "cc", or "sitemap+cc"
    seeding_query="python programming tutorial",
    seeding_score_threshold=0.2,
    seeding_concurrency=100,
    seeding_hits_per_sec=20
)
```

### 2. Advanced Crawling Modes

```python
async def advanced_crawling_example():
    async with Crawl4AIClient(config) as client:
        
        # Mode 1: Single page
        single_result = await client.crawl_async(
            "https://example.com",
            mode=CrawlMode.SINGLE_PAGE,
            output_format=OutputFormat.MARKDOWN
        )
        
        # Mode 2: Seeded crawling (uses official AsyncUrlSeeder)
        seeded_results = await client.crawl_async(
            "example.com",  # Can use domain or full URL
            mode=CrawlMode.SEEDED,
            output_format=OutputFormat.STRUCTURED
        )
        
        # Mode 3: Multi-page (fallback method)
        multi_results = await client.crawl_async(
            "https://example.com",
            mode=CrawlMode.MULTI_PAGE,
            output_format=OutputFormat.JSON
        )
        
        # Mode 4: Sitemap-based
        sitemap_results = await client.crawl_async(
            "https://example.com",
            mode=CrawlMode.SITEMAP,
            output_format=OutputFormat.TEXT
        )
```

### 3. Output Formats

```python
from src.crawl4ai_client import OutputFormat

# Available formats
formats = [
    OutputFormat.MARKDOWN,    # Clean markdown content
    OutputFormat.HTML,        # Raw HTML
    OutputFormat.TEXT,        # Plain text extraction
    OutputFormat.JSON,        # Structured JSON with metadata
    OutputFormat.STRUCTURED   # Full structured data with all fields
]

for format_type in formats:
    result = await quick_crawl_async(
        "https://example.com", 
        output_format=format_type
    )
    print(f"{format_type.value}: {len(str(result.extracted_content))} chars")
```

### 4. Advanced Seeding Configuration

```python
from src.crawl4ai_client import CrawlConfig

# Comprehensive seeding configuration
config = CrawlConfig(
    # URL Discovery
    seeding_source="sitemap+cc",           # Discovery sources
    seeding_pattern="*.example.com/blog/*", # URL pattern filter
    seeding_live_check=True,               # Verify URLs are accessible
    seeding_extract_head=True,             # Extract metadata for scoring
    
    # Relevance Scoring
    seeding_query="machine learning AI tutorial",  # Search query
    seeding_score_threshold=0.3,           # Minimum relevance score
    
    # Performance Tuning
    seeding_concurrency=200,               # Concurrent requests
    seeding_hits_per_sec=50,               # Rate limiting
    seeding_force=False,                   # Use cached results
    
    # Content Processing
    max_pages=50,                          # Maximum pages to crawl
    css_selector="main, article, .content" # Content extraction selector
)
```

### 5. Real-World Examples

#### News Website Scraping
```python
async def scrape_news_site():
    config = CrawlConfig(
        max_pages=20,
        seeding_source="sitemap",
        seeding_query="breaking news politics economy",
        seeding_score_threshold=0.4,
        css_selector="article, .news-content, .story-body",
        exclude_external_links=True
    )
    
    async with Crawl4AIClient(config) as client:
        results = await client.crawl_async(
            "news-website.com",
            mode=CrawlMode.SEEDED
        )
        
        articles = [r for r in results if r.success]
        print(f"📰 Scraped {len(articles)} news articles")
        
        # Sort by relevance score
        articles.sort(key=lambda x: x.relevance_score or 0, reverse=True)
        
        for i, article in enumerate(articles[:5], 1):
            score = article.relevance_score or 0
            content_length = len(article.extracted_content or '')
            print(f"{i}. {article.url} (score: {score:.3f}, {content_length} chars)")
```

#### Documentation Crawling
```python
async def crawl_documentation():
    config = CrawlConfig(
        max_pages=100,
        seeding_source="sitemap+cc",
        seeding_query="API reference tutorial guide documentation",
        seeding_score_threshold=0.2,
        css_selector="main, .documentation, .doc-content, .api-doc",
        seeding_concurrency=50
    )
    
    async with Crawl4AIClient(config) as client:
        results = await client.crawl_async(
            "docs.example.com",
            mode=CrawlMode.SEEDED,
            output_format=OutputFormat.STRUCTURED
        )
        
        # Create knowledge base
        knowledge_base = []
        for result in results:
            if result.success:
                knowledge_base.append({
                    'url': result.url,
                    'title': extract_title(result.extracted_content),
                    'content': result.extracted_content,
                    'relevance_score': result.relevance_score,
                    'word_count': len((result.extracted_content or '').split()),
                    'links': result.links[:10]  # Top 10 links
                })
        
        # Export knowledge base
        client.export_results(results, 'knowledge_base.json')
        print(f"📚 Created knowledge base with {len(knowledge_base)} documents")
```

#### Competitive Analysis
```python
async def competitive_analysis():
    competitors = [
        "competitor1.com",
        "competitor2.com", 
        "competitor3.com"
    ]
    
    config = CrawlConfig(
        max_pages=15,
        seeding_source="sitemap",
        seeding_query="pricing features products services",
        css_selector="main, .features, .pricing, .products"
    )
    
    analysis_results = {}
    
    async with Crawl4AIClient(config) as client:
        for competitor in competitors:
            print(f"🔍 Analyzing {competitor}...")
            
            results = await client.crawl_async(
                competitor,
                mode=CrawlMode.SEEDED
            )
            
            successful_pages = [r for r in results if r.success]
            total_content = ' '.join(r.extracted_content or '' for r in successful_pages)
            
            analysis_results[competitor] = {
                'pages_analyzed': len(successful_pages),
                'total_words': len(total_content.split()),
                'unique_links': len(set(link for r in successful_pages for link in r.links)),
                'avg_relevance': sum(r.relevance_score or 0 for r in successful_pages) / len(successful_pages) if successful_pages else 0
            }
    
    # Print analysis summary
    for competitor, data in analysis_results.items():
        print(f"📊 {competitor}:")
        print(f"   Pages: {data['pages_analyzed']}")
        print(f"   Words: {data['total_words']:,}")
        print(f"   Links: {data['unique_links']}")
        print(f"   Avg Relevance: {data['avg_relevance']:.3f}")
```

### 6. Performance Monitoring

```python
async def monitor_performance():
    async with Crawl4AIClient() as client:
        # Perform multiple crawls
        urls = ["https://example1.com", "https://example2.com", "https://example3.com"]
        
        for url in urls:
            await client.crawl_async(url)
        
        # Get detailed statistics
        stats = client.get_stats()
        
        print("📈 Crawling Statistics:")
        print(f"   Total requests: {stats['total_requests']}")
        print(f"   Success rate: {stats['success_rate']:.1%}")
        print(f"   Average time: {stats['average_processing_time']:.2f}s")
        print(f"   URLs seeded: {stats.get('urls_seeded', 0)}")
        print(f"   Official seeding: {stats['official_seeding_available']}")
```

### 7. Error Handling and Resilience

```python
async def robust_crawling():
    config = CrawlConfig(
        max_pages=20,
        timeout=30,
        seeding_concurrency=10,  # Lower concurrency for stability
        seeding_hits_per_sec=5   # Conservative rate limiting
    )
    
    async with Crawl4AIClient(config) as client:
        try:
            results = await client.crawl_async(
                "example.com",
                mode=CrawlMode.SEEDED
            )
            
            successful = [r for r in results if r.success]
            failed = [r for r in results if not r.success]
            
            print(f"✅ Successful: {len(successful)}")
            print(f"❌ Failed: {len(failed)}")
            
            # Log failed URLs for retry
            for failure in failed:
                print(f"   Failed: {failure.url} - {failure.error_message}")
                
        except Exception as e:
            print(f"💥 Crawling failed: {str(e)}")
```

## 🔧 Configuration Reference

### CrawlConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_pages` | int | 10 | Maximum pages to crawl |
| `max_depth` | int | 2 | Maximum crawl depth |
| `timeout` | int | 30 | Request timeout in seconds |
| `css_selector` | str | None | CSS selector for content extraction |
| `seeding_source` | str | "sitemap+cc" | URL discovery sources |
| `seeding_query` | str | None | Query for BM25 relevance scoring |
| `seeding_score_threshold` | float | None | Minimum relevance score |
| `seeding_concurrency` | int | 100 | Concurrent seeding requests |
| `seeding_hits_per_sec` | int | 10 | Rate limiting for requests |

### Seeding Sources

| Source | Description |
|--------|-------------|
| `"sitemap"` | Discover URLs from sitemap.xml files |
| `"cc"` | Use Common Crawl archive data |
| `"sitemap+cc"` | Combine both methods for comprehensive coverage |

## 📊 Performance Tips

1. **Use Seeding for Large Sites**: The official AsyncUrlSeeder is much more efficient than traditional crawling for discovering relevant pages.

2. **Optimize Concurrency**: Balance `seeding_concurrency` and `seeding_hits_per_sec` based on target site capacity.

3. **Query-Based Filtering**: Use `seeding_query` and `seeding_score_threshold` to focus on relevant content.

4. **CSS Selectors**: Use specific CSS selectors to extract only relevant content sections.

5. **Caching**: Set `seeding_force=False` to use cached discovery results.

## 🐛 Troubleshooting

### Common Issues

**1. "Official async seeding not available"**
```bash
# Install full Crawl4AI dependencies
pip install crawl4ai[all]

# Or install specific dependencies
pip install rank-bm25 lxml brotli
```

**2. Rate Limiting Errors**
```python
# Reduce request rate
config = CrawlConfig(
    seeding_hits_per_sec=5,
    seeding_concurrency=50
)
```

**3. Memory Issues with Large Crawls**
```python
# Process in smaller batches
config = CrawlConfig(
    max_pages=20,  # Smaller batches
    seeding_concurrency=30  # Lower concurrency
)
```

**4. Timeout Errors**
```python
# Increase timeouts
config = CrawlConfig(
    timeout=60,
    delay_before_return_html=3.0
)
```

## 🔗 Integration Examples

### With Existing Website-to-Agent Project

```python
# Replace the existing URL seeding in src/llms_text.py
from src.crawl4ai_client import seeded_crawl_async

async def extract_website_content(url: str, max_urls: int = 10):
    """Enhanced content extraction with official seeding"""
    
    # Use seeded crawling for better URL discovery
    results = await seeded_crawl_async(
        domain=url,
        max_pages=max_urls,
        source="sitemap+cc",
        query=None,  # No query filtering for general extraction
        output_format=OutputFormat.MARKDOWN
    )
    
    # Combine results into the expected format
    successful_results = [r for r in results if r.success]
    
    combined_content = "\n\n---\n\n".join(
        r.extracted_content for r in successful_results
    )
    
    return {
        'llmstxt': combined_content,
        'llmsfulltxt': combined_content,
        'processed_urls': [r.url for r in successful_results],
        'failed_urls': [r.url for r in results if not r.success],
        'discovered_urls': [r.url for r in results],
        'extraction_timestamp': datetime.now().isoformat(),
        'total_processing_time': sum(r.processing_time for r in results)
    }
```

## 📄 License

This client is based on the open-source [Crawl4AI](https://github.com/unclecode/crawl4ai) project. Please refer to the original project for licensing information.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📞 Support

- **Issues**: Report bugs and feature requests via GitHub issues
- **Documentation**: Check the [official Crawl4AI documentation](https://crawl4ai.com)
- **Community**: Join the [Crawl4AI Discord](https://discord.gg/jP8KfhDhyN)

---

**Built with ❤️ using [Crawl4AI](https://github.com/unclecode/crawl4ai) - The open-source LLM-friendly web crawler**
