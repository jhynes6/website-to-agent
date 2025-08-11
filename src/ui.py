import streamlit as st
import asyncio
import threading
import queue
import sys
import os
import logging
import time
import re

# Add the parent directory to the Python path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DEFAULT_MAX_URLS, DEFAULT_USE_FULL_TEXT

# Set up logging for UI workflow tracking
logger = logging.getLogger('website-to-agent')
from src.llms_text import extract_website_content
from src.agents import extract_domain_knowledge, create_domain_agent

def sanitize_markdown_content(content):
    """Sanitize markdown content to prevent ReactMarkdown parsing errors."""
    if not content:
        return ""
    
    # Convert to string if not already
    content = str(content)
    
    # Remove or escape problematic markdown directives that might cause parsing errors
    # Remove HTML-like directives that aren't supported
    content = re.sub(r'<[^>]+>', '', content)
    
    # Remove any null bytes or other control characters
    content = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', content)
    
    # Ensure content ends with proper whitespace
    content = content.strip()
    
    return content

# Initialize session state
def init_session_state():
    if 'domain_agent' not in st.session_state:
        st.session_state.domain_agent = None
    if 'domain_knowledge' not in st.session_state:
        st.session_state.domain_knowledge = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'extraction_status' not in st.session_state:
        st.session_state.extraction_status = None
    if 'pending_response' not in st.session_state:
        st.session_state.pending_response = None

def run_app():
    # Initialize session state
    init_session_state()
    
    # Check if we have a pending response to add to the message history
    if st.session_state.pending_response is not None:
        st.session_state.messages.append({"role": "assistant", "content": st.session_state.pending_response})
        st.session_state.pending_response = None
    
    # App title and description in main content area
    st.title("WebToAgent")
    st.subheader("Extract domain knowledge from any website and create specialized AI agents.")
    
    # Display welcome message using AI chat message component
    if not st.session_state.domain_agent:
        with st.chat_message("assistant"):
            st.markdown("👋 Welcome! Enter a website URL in the sidebar, and I'll transform it into an AI agent you can chat with.")
    
    # Form elements in sidebar
    st.sidebar.title("Create your agent")
    
    website_url = st.sidebar.text_input("Enter website URL", placeholder="https://example.com")
    
    max_pages = st.sidebar.slider("Maximum pages to analyze", 1, 50, DEFAULT_MAX_URLS, 
                         help="More pages means more comprehensive knowledge but longer processing time.")
    
    use_full_text = st.sidebar.checkbox("Use comprehensive text extraction", value=DEFAULT_USE_FULL_TEXT,
                                help="Extract full contents of each page (always enabled with Crawl4AI)")
    
    submit_button = st.sidebar.button("Create agent", type="primary")
    
    # Process form submission
    if submit_button and website_url:
        logger.info("🚀 WORKFLOW START: User submitted form")
        logger.info(f"📝 Parameters: URL={website_url}, max_pages={max_pages}, use_full_text={use_full_text}")
        st.session_state.extraction_status = "extracting"
        
        try:
            # Create progress containers
            progress_container = st.container()
            
            with progress_container:
                st.info("🕷️ Starting website analysis with Crawl4AI...")
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: URL Discovery
                status_text.text("🌱 Step 1/3: Discovering website pages...")
                progress_bar.progress(10)
                
                logger.info("🌐 STEP 1: Starting Crawl4AI extraction...")
                
                # Run the async extraction in a thread
                content = run_async_extraction(
                    website_url, 
                    max_pages, 
                    use_full_text,
                    progress_bar,
                    status_text
                )
                
                if content is None:
                    st.error("❌ Failed to extract content from the website. Please check the URL and try again.")
                    st.session_state.extraction_status = "failed"
                    return
                
                progress_bar.progress(60)
                status_text.text("✅ Content extraction completed!")
                
                logger.info(f"✅ STEP 1 COMPLETE: Extracted content from {len(content.get('processed_urls', []))} pages")
                
                # Show extraction results
                with st.expander("📊 Extraction Results", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Pages Discovered", len(content.get('discovered_urls', [])))
                    with col2:
                        st.metric("Pages Processed", len(content.get('processed_urls', [])))
                    with col3:
                        st.metric("Content Length", f"{len(content.get('llmstxt', '')):,} chars")
                    
                    if content.get('failed_urls'):
                        st.warning(f"⚠️ Failed to process {len(content['failed_urls'])} URLs")
                        with st.expander("Failed URLs"):
                            for url in content['failed_urls']:
                                st.text(url)
                    
                    # Show content preview
                    if content.get('llmstxt'):
                        st.text_area(
                            "Content Preview (first 1000 characters):",
                            value=content['llmstxt'][:1000] + "..." if len(content['llmstxt']) > 1000 else content['llmstxt'],
                            height=200
                        )
                
                # Step 2: Knowledge Extraction
                progress_bar.progress(70)
                status_text.text("🧠 Step 2/3: Analyzing content and extracting knowledge...")
                
                logger.info("🧠 STEP 2: Starting knowledge extraction with OpenAI...")
                
                # Use the appropriate content based on user choice
                content_to_analyze = content['llmsfulltxt'] if use_full_text else content['llmstxt']
                
                domain_knowledge = asyncio.run(extract_domain_knowledge(
                    content_to_analyze,
                    website_url
                ))
                
                logger.info(f"✅ STEP 2 COMPLETE: Knowledge extracted - {len(domain_knowledge.core_concepts)} concepts, {len(domain_knowledge.terminology)} terms")
                
                # Store in session state
                st.session_state.domain_knowledge = domain_knowledge
                
                progress_bar.progress(90)
                status_text.text("🤖 Step 3/3: Creating specialized agent...")
                
                # Step 3: Create Agent
                logger.info("🤖 STEP 3: Creating specialized domain agent...")
                domain_agent = create_domain_agent(domain_knowledge)
                
                # Store in session state
                st.session_state.domain_agent = domain_agent
                logger.info("✅ STEP 3 COMPLETE: Domain agent created and ready")
                
                progress_bar.progress(100)
                status_text.text("🎉 Agent creation complete!")
                
                st.session_state.extraction_status = "complete"
                logger.info("🎉 WORKFLOW COMPLETE: Agent successfully created and ready for chat")
                
                # Success message with knowledge summary
                st.success("🎉 Agent created successfully!")
                
                # Display knowledge summary
                with st.expander("🧠 Extracted Knowledge Summary", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📋 Core Concepts")
                        for concept in domain_knowledge.core_concepts[:5]:  # Show top 5
                            st.write(f"• **{concept.name}**: {concept.description[:100]}...")
                        
                        if len(domain_knowledge.core_concepts) > 5:
                            st.write(f"... and {len(domain_knowledge.core_concepts) - 5} more concepts")
                    
                    with col2:
                        st.subheader("📖 Key Terms")
                        for term in domain_knowledge.terminology[:5]:  # Show top 5
                            st.write(f"• **{term.term}**: {term.definition[:100]}...")
                        
                        if len(domain_knowledge.terminology) > 5:
                            st.write(f"... and {len(domain_knowledge.terminology) - 5} more terms")
                
                # Clear progress indicators after a moment
                import time
                time.sleep(2)
                progress_container.empty()
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            logger.error(f"❌ WORKFLOW ERROR: {str(e)}")
            st.session_state.extraction_status = "failed"
    
    # Chat interface
    if st.session_state.domain_agent:
        display_chat_interface()

def run_async_extraction(url: str, max_pages: int, use_full_text: bool, progress_bar, status_text):
    """
    Run the async extraction in a separate thread.
    """
    result_queue = queue.Queue()
    error_queue = queue.Queue()
    
    def run_extraction():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run the extraction without Streamlit updates (avoid context issues)
            async def extraction_task():
                result = await extract_website_content(
                    url=url,
                    max_urls=max_pages,
                    show_full_text=use_full_text
                )
                return result
            
            result = loop.run_until_complete(extraction_task())
            result_queue.put(result)
            
        except Exception as e:
            logger.error(f"❌ Extraction thread error: {str(e)}")
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"❌ Full traceback: {error_details}")
            error_queue.put(str(e))
            result_queue.put(None)
        finally:
            try:
                loop.close()
            except:
                pass
    
    # Update progress in main thread
    status_text.text("🌱 Step 1/3: Discovering website pages...")
    progress_bar.progress(20)
    
    # Run extraction in thread
    thread = threading.Thread(target=run_extraction)
    thread.daemon = True
    thread.start()
    
    # Wait for result with timeout and progress updates
    timeout = 300  # 5 minutes
    start_time = time.time()
    
    while thread.is_alive():
        elapsed = time.time() - start_time
        if elapsed > timeout:
            st.error("⏰ Extraction timed out. Please try with fewer pages or a simpler website.")
            return None
        
        # Update progress periodically
        progress = min(90, 20 + (elapsed / timeout) * 70)  # 20% to 90%
        progress_bar.progress(int(progress))
        
        # Check for result
        try:
            result = result_queue.get(timeout=1)  # Check every second
            progress_bar.progress(100)
            status_text.text("✅ Content extraction completed!")
            return result
        except queue.Empty:
            continue
    
    # Thread finished, get final result
    try:
        result = result_queue.get(timeout=1)
        if result is None and not error_queue.empty():
            error_msg = error_queue.get()
            st.error(f"❌ Extraction failed: {error_msg}")
        return result
    except queue.Empty:
        st.error("❌ Extraction failed: No result returned")
        return None

def stream_agent_response(agent, prompt):
    """Stream agent response using the new DomainAgent interface."""
    
    def get_agent_response():
        """Get response from agent in a thread."""
        result_queue = queue.Queue()
        
        def async_runner():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                async def get_response():
                    response = await agent.chat(prompt)
                    return response
                
                response = loop.run_until_complete(get_response())
                result_queue.put(response)
                
            except Exception as e:
                logger.error(f"❌ Agent response error: {str(e)}")
                result_queue.put(f"❌ Error: {str(e)}")
            finally:
                loop.close()
        
        thread = threading.Thread(target=async_runner)
        thread.daemon = True
        thread.start()
        
        # Wait for result
        try:
            response = result_queue.get(timeout=60)  # 1 minute timeout
            return response
        except queue.Empty:
            return "⏰ Response timed out. Please try again."
    
    # Get the full response
    full_response = get_agent_response()
    
    # Store complete response for session state
    st.session_state.pending_response = full_response
    
    # Simulate streaming by yielding words with small delays
    words = full_response.split()
    for i, word in enumerate(words):
        yield word + " "
        if i % 10 == 0:  # Add slight pause every 10 words for visual effect
            time.sleep(0.02)

def get_non_streaming_response(agent, prompt):
    """Fallback function for non-streaming response."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        response_text = loop.run_until_complete(agent.chat(prompt))
        
        # Store for the next Streamlit run to pick up
        st.session_state.pending_response = response_text
        
        return response_text
    except Exception as e:
        logger.error(f"❌ Non-streaming response error: {str(e)}")
        return f"❌ Error: {str(e)}"
    finally:
        loop.close()

def display_chat_interface():
    """Display chat interface for interacting with the domain agent."""
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            try:
                # Sanitize content to prevent markdown parsing errors
                safe_content = sanitize_markdown_content(message["content"])
                if safe_content:
                    st.markdown(safe_content)
                else:
                    st.text("(Empty message)")
            except Exception as e:
                logger.warning(f"⚠️ Markdown rendering error: {str(e)}")
                # Fallback to plain text display
                st.text(str(message["content"]))
    
    # Chat input
    if prompt := st.chat_input("Ask a question about this domain..."):
        logger.info(f"💬 CHAT START: User asked - '{prompt}'")
        logger.info(f"📊 Chat history: {len(st.session_state.messages)} previous messages")
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            try:
                safe_prompt = sanitize_markdown_content(prompt)
                st.markdown(safe_prompt)
            except Exception as e:
                logger.warning(f"⚠️ User prompt markdown error: {str(e)}")
                st.text(prompt)
        
        # Reset any pending response
        st.session_state.pending_response = None
        
        # Get agent response with streaming
        with st.chat_message("assistant"):
            try:
                logger.info("🔄 CHAT PROCESSING: Streaming response from domain agent...")
                # Stream the response tokens
                token_stream = stream_agent_response(st.session_state.domain_agent, prompt)
                st.write_stream(token_stream)
                logger.info("✅ CHAT COMPLETE: Response delivered successfully")
                
            except Exception as e:
                logger.warning(f"⚠️ CHAT FALLBACK: Streaming failed - {str(e)}")
                # Fallback to non-streaming response if streaming fails
                st.warning(f"Streaming failed ({str(e)}), using standard response method.")
                try:
                    logger.info("🔄 CHAT FALLBACK: Using non-streaming response...")
                    full_response = get_non_streaming_response(st.session_state.domain_agent, prompt)
                    try:
                        safe_response = sanitize_markdown_content(full_response)
                        st.markdown(safe_response)
                    except Exception as markdown_error:
                        logger.warning(f"⚠️ Fallback markdown error: {str(markdown_error)}")
                        st.text(full_response)
                    logger.info("✅ CHAT FALLBACK COMPLETE: Non-streaming response delivered")
                except Exception as e2:
                    logger.error(f"❌ CHAT ERROR: Failed to generate response - {str(e2)}")
                    st.error(f"Error generating response: {str(e2)}")

if __name__ == "__main__":
    run_app()
