echo "🔍 Monitoring deployment logs - waiting for app to start without critical errors..." && doctl apps logs c363f830-2d40-4648-bedc-a5a6256f3bf6 --type run --follow



How to launch the app: 

export OPENAI_API_KEY=your_openai_api_key_here && streamlit run app.py


# WebToAgent

WebToAgent is a Streamlit application that extracts domain knowledge from websites and creates specialized AI agents capable of answering questions about that domain. By leveraging asynchronous web crawling with Crawl4AI and advanced language models, WebToAgent transforms website content into interactive, conversational agents.

![WebToAgent Screenshot](https://placeholder-for-screenshot.png)

## Features

- **Asynchronous website crawling**: Crawl websites using Crawl4AI with intelligent URL discovery
- **Smart URL seeding**: Automatically discover relevant pages within a website domain
- **Knowledge model generation**: Process extracted content to build domain-specific knowledge models
- **AI agent creation**: Create conversational agents specialized in the crawled domain
- **Interactive chat interface**: Engage with your domain agent through a real-time chat interface
- **Streaming responses**: Get real-time streaming responses from the agent as it generates them
- **Progress tracking**: Visual progress indicators for all stages of agent creation

## Prerequisites

Before installing WebToAgent, make sure you have:

- Python 3.10+ installed
- Pip (Python package installer)
- Access to OpenAI API credentials

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repo/website-to-agent.git
   cd website-to-agent
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Or using Poetry:

   ```bash
   poetry install
   poetry shell
   ```

4. Configure your API credentials (see Configuration section)

## Configuration

1. Create a `.env` file in the project root with your API credentials:

   ```bash
   OPENAI_API_KEY=your_openai_api_key
   ```

2. You can customize default settings in the `src/config.py` file, including:
   - `DEFAULT_MAX_URLS`: Maximum number of URLs to crawl (default: 10)
   - `DEFAULT_CRAWL_DEPTH`: Maximum crawl depth (default: 2)
   - `DEFAULT_BATCH_SIZE`: Batch size for concurrent crawling (default: 3)

## Usage

1. Start the Streamlit application:

   ```bash
   streamlit run src/ui.py
   ```

2. Open your browser and navigate to `http://localhost:8501`

3. Enter a website URL in the sidebar

4. Configure crawling options:
   - Adjust the maximum pages to analyze
   - Comprehensive text extraction is always enabled with Crawl4AI

5. Click "Create agent" to start the process

6. Monitor progress through the visual indicators:
   - URL discovery phase
   - Content extraction phase  
   - Knowledge analysis phase
   - Agent creation phase

7. Once the agent is created, you can ask questions about the website's domain in the chat interface

## How it works

WebToAgent operates in three main phases:

1. **Asynchronous URL Discovery**: Starting from the provided URL, the system intelligently discovers related pages within the same domain using breadth-first crawling with configurable depth limits.

2. **Content Extraction**: Using Crawl4AI, the application processes discovered URLs in batches, extracting clean, structured content from each page. The system handles JavaScript-rendered content and removes navigation elements for clean text extraction.

3. **Knowledge Model Creation**: The extracted content is processed using advanced language models to build a domain-specific knowledge representation, identifying key concepts, terminology, and insights.

4. **Agent Creation**: The domain knowledge is used to create a specialized AI agent that can provide conversational responses about the website's domain with contextual understanding.

## Testing

Run the test suite to verify the installation:

```bash
python test_crawl4ai.py
```

This will test:
- URL seeding functionality
- Content extraction with Crawl4AI
- Full workflow integration

## Deployment

To deploy WebToAgent to a production environment:

### Using Streamlit Cloud

1. Push your code to a GitHub repository
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub account
4. Select your repository and the `src/ui.py` file
5. Configure your secrets (API keys) in the Streamlit Cloud dashboard
6. Deploy

### On a server

1. Set up a server with Python 3.10+
2. Clone the repository and install dependencies
3. Configure environment variables for API keys
4. Use a process manager like Supervisor to run:

   ```bash
   streamlit run src/ui.py --server.port=80 --server.address=0.0.0.0
   ```

5. (Optional) Set up Nginx as a reverse proxy for better security and performance

## Troubleshooting

### Common issues

- **Streaming responses disappearing**: If streaming responses disappear when submitting a new message, restart the application. The app utilizes a mechanism to maintain streaming state across Streamlit reruns.

- **Crawl timeouts**: If crawling times out, reduce the maximum pages to analyze or check your internet connection. The system includes built-in timeout handling.

- **Memory issues**: For large websites, consider increasing the available memory for the Python process or reducing the scope of crawling.

- **URL discovery fails**: Some websites may block automated crawling. Try with a different website or reduce the crawl depth.

## Technical details

### Asynchronous Architecture

WebToAgent uses modern async/await patterns for efficient crawling:

1. **URL Discovery**: Concurrent discovery of website pages using aiohttp
2. **Content Extraction**: Batch processing with Crawl4AI for optimal performance
3. **Error Handling**: Robust error handling with retry mechanisms
4. **Rate Limiting**: Built-in delays and batch processing to respect server limits

### Stream handling in Streamlit

WebToAgent implements a solution for maintaining streaming responses when the Streamlit app reruns:

1. **Global state management**: Uses a global variable to maintain the streaming state outside of Streamlit's session state.

2. **Thread-safe implementation**: Separates UI and streaming concerns:
   - Background threads handle API calls and collect tokens
   - Main thread manages UI and session state

3. **Cross-rerun state transfer**: Stores completed responses in the session state to be added to the chat history on the next Streamlit run.

## Performance Optimizations

- **Concurrent Processing**: URLs are processed in configurable batches for optimal speed
- **Content Filtering**: Intelligent filtering of non-content URLs (images, documents, admin pages)
- **Caching**: Crawl4AI includes built-in caching to avoid re-processing pages
- **Memory Management**: Efficient memory usage through streaming and batch processing

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

## Acknowledgments

- [Crawl4AI](https://crawl4ai.com) for advanced web crawling capabilities
- [Streamlit](https://streamlit.io) for the web application framework
- [OpenAI](https://openai.com) for language model capabilities
- [aiohttp](https://aiohttp.readthedocs.io/) for asynchronous HTTP client capabilities
