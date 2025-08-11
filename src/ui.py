import streamlit as st
import asyncio
import threading
import queue
import sys
import os
import logging
import time
import re
import traceback

# Add the parent directory to the Python path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DEFAULT_MAX_URLS, DEFAULT_USE_FULL_TEXT

# Set up logging for UI workflow tracking
logger = logging.getLogger('website-to-agent')

# URGENT: Add comprehensive error logging
def log_error_with_traceback(error_msg, exception=None):
    """Log errors with full traceback to server logs"""
    logger.error(f"🚨 UI ERROR: {error_msg}")
    if exception:
        logger.error(f"🚨 EXCEPTION TYPE: {type(exception).__name__}")
        logger.error(f"🚨 EXCEPTION MSG: {str(exception)}")
        logger.error(f"🚨 TRACEBACK: {traceback.format_exc()}")
    else:
        logger.error(f"🚨 STACK TRACE: {''.join(traceback.format_stack())}")

from src.llms_text import extract_website_content
from src.agents import extract_domain_knowledge, create_domain_agent

def sanitize_markdown_content(content):
    """ULTRA-AGGRESSIVE sanitization to prevent ReactMarkdown parsing errors."""
    try:
        if not content:
            return ""
        
        # Convert to string if not already
        content = str(content)
        
        # LOG THE RAW CONTENT FOR DEBUGGING
        logger.error(f"🔍 SANITIZING CONTENT (first 200 chars): {repr(content[:200])}")
        
        # NUCLEAR OPTION: Remove EVERYTHING that could possibly break ReactMarkdown
        
        # Remove ALL HTML/XML tags and attributes  
        content = re.sub(r'<[^>]*?>', ' ', content)
        content = re.sub(r'&[a-zA-Z0-9#]+;', ' ', content)  # HTML entities
        content = re.sub(r'<!(?:DOCTYPE|--)[^>]*>', ' ', content)  # DOCTYPE and comments
        
        # Remove ALL markdown directives, attributes, and special syntax
        content = re.sub(r':::[^:]*?:::', ' ', content, flags=re.DOTALL)  # Admonitions
        content = re.sub(r'\{[^}]*\}', ' ', content)  # Any attributes in curly braces
        content = re.sub(r'\[[^\]]*\](?:\([^)]*\))?', ' ', content)  # Links and refs
        content = re.sub(r'---+', ' ', content)  # Horizontal rules
        content = re.sub(r'\|[^|]*\|', ' ', content)  # Table syntax
        content = re.sub(r'^#+\s*', '', content, flags=re.MULTILINE)  # Headers
        
        # Remove ALL special characters that could confuse ReactMarkdown
        content = re.sub(r'[<>{}\\|`~\[\]()_*#@$%^&+=]', ' ', content)
        
        # Remove ALL control and unicode characters
        content = re.sub(r'[\x00-\x1F\x7F-\x9F]', ' ', content)
        content = re.sub(r'[^\x20-\x7E]', ' ', content)  # Only allow basic ASCII
        
        # Remove multiple consecutive punctuation/special chars
        content = re.sub(r'[^\w\s]{2,}', ' ', content)
        
        # Normalize ALL whitespace
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()
        
        # STRICT length limit
        if len(content) > 3000:
            content = content[:3000] + " (truncated)"
        
        # FINAL NUCLEAR SAFETY CHECK - only allow basic alphanumeric + minimal punctuation
        content = re.sub(r'[^a-zA-Z0-9\s.,!?:-]', ' ', content)
        content = re.sub(r'\s+', ' ', content).strip()
        
        logger.error(f"🛡️ SANITIZED RESULT (first 200 chars): {repr(content[:200])}")
        
        return content if content else "Content processed safely"
        
    except Exception as e:
        log_error_with_traceback("CRITICAL: markdown sanitization failed", e)
        # ABSOLUTE EMERGENCY FALLBACK
        return "Error: Content could not be processed safely"

def safe_markdown_display(content, fallback_to_text=True):
    """EMERGENCY: COMPLETELY BYPASS MARKDOWN - Use only text display to prevent ReactMarkdown crashes"""
    try:
        logger.error(f"🚨 SAFE_MARKDOWN_DISPLAY: Bypassing markdown completely, using text only")
        safe_content = sanitize_markdown_content(content)
        
        # EMERGENCY: NEVER USE st.markdown() - only st.text() to eliminate ReactMarkdown crashes
        if safe_content and len(safe_content.strip()) > 0:
            st.text(safe_content[:1000])  # Text display only, truncated for safety
            return True
        else:
            st.text("(Empty or invalid content)")
            return False
            
    except Exception as e:
        log_error_with_traceback("Even text display failed", e)
        st.text("Error: Content could not be displayed safely")
        return False

def safe_error_display(error_message):
    """Safely display error messages without risking ReactMarkdown crashes"""
    try:
        # Aggressively sanitize error messages
        safe_msg = re.sub(r'[^\w\s.,!?():-]', ' ', str(error_message))
        safe_msg = re.sub(r'\s+', ' ', safe_msg).strip()[:500]
        
        # Use st.error with heavily sanitized content, fallback to warning/text if that fails
        try:
            st.error(f"❌ {safe_msg}")
        except Exception as error_display_e:
            log_error_with_traceback("st.error failed, using st.warning", error_display_e)
            try:
                st.warning(f"Error: {safe_msg}")
            except Exception as warning_display_e:
                log_error_with_traceback("st.warning failed, using st.text", warning_display_e)
                st.text(f"ERROR: {safe_msg}")
    except Exception as e:
        log_error_with_traceback("safe_error_display completely failed", e)
        st.text("ERROR: An error occurred but could not be displayed safely")

def run_async_extraction(url, max_urls, use_full_text, result_queue, error_queue):
    """Run the async extraction in a separate thread"""
    logger.info(f"🔧 THREAD START: Starting extraction thread for {url}")
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        logger.info("✅ THREAD: Event loop created successfully")
        
        async def extract():
            logger.info(f"🚀 ASYNC START: Beginning extraction for {url} (max_urls={max_urls})")
            try:
                result = await extract_website_content(
                    url=url,
                    max_urls=max_urls,
                    show_full_text=use_full_text
                )
                logger.info(f"✅ ASYNC SUCCESS: Extraction completed for {url}")
                return result
            except Exception as async_e:
                log_error_with_traceback(f"Async extraction failed for {url}", async_e)
                raise
        
        result = loop.run_until_complete(extract())
        logger.info(f"📤 THREAD: Putting result in queue (size: {len(str(result))} chars)")
        result_queue.put(result)
        logger.info("✅ THREAD: Result successfully queued")
        
    except Exception as thread_e:
        log_error_with_traceback(f"Thread extraction failed for {url}", thread_e)
        error_queue.put(str(thread_e))
        logger.error(f"📤 THREAD: Error queued: {str(thread_e)}")
    finally:
        logger.info("🔧 THREAD END: Extraction thread completed")

def run_workflow():
    """Main workflow for extracting content and creating agents"""
    logger.info("🎯 WORKFLOW START: Beginning main workflow")
    
    try:
        url = st.session_state.url
        max_urls = st.session_state.max_urls
        logger.info(f"📋 WORKFLOW PARAMS: URL={url}, max_urls={max_urls}")

        # Step 1: Content Extraction
        # EMERGENCY: Use st.text instead of st.success to avoid markdown parsing
        st.text("Starting website analysis with Crawl4AI...")
        logger.info("📊 STEP 1: Starting content extraction")
        
        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Update progress
        status_text.text("🌱 Step 1/3: Discovering website pages...")
        logger.info("🌱 PROGRESS: Step 1/3 - Discovering pages")
        progress_bar.progress(10)
        
        # Run extraction in background thread
        result_queue = queue.Queue()
        error_queue = queue.Queue()
        
        logger.info("🧵 THREAD: Starting extraction thread")
        extraction_thread = threading.Thread(
            target=run_async_extraction,
            args=(url, max_urls, DEFAULT_USE_FULL_TEXT, result_queue, error_queue)
        )
        extraction_thread.start()
        logger.info("✅ THREAD: Extraction thread started successfully")
        
        # Monitor progress
        progress_steps = [
            (20, "🔍 Step 1/3: Analyzing website structure..."),
            (40, "📄 Step 1/3: Extracting content from pages..."),
            (60, "✨ Step 1/3: Processing extracted content..."),
        ]
        
        step_idx = 0
        max_wait_time = 60  # seconds
        start_time = time.time()
        
        while extraction_thread.is_alive():
            if time.time() - start_time > max_wait_time:
                logger.error("⏰ TIMEOUT: Extraction taking too long")
                break
                
            if step_idx < len(progress_steps):
                progress, message = progress_steps[step_idx]
                progress_bar.progress(progress)
                status_text.text(message)
                logger.info(f"�� PROGRESS UPDATE: {message}")
                step_idx += 1
            
            time.sleep(2)
        
        extraction_thread.join(timeout=5)
        logger.info("🧵 THREAD: Extraction thread joined")
        
        # Check results
        result = None
        if not result_queue.empty():
            result = result_queue.get()
            logger.info(f"📥 RESULT: Got extraction result (size: {len(str(result))} chars)")
        
        if result is None and not error_queue.empty():
            error_msg = error_queue.get()
            safe_error_msg = sanitize_markdown_content(str(error_msg))
            logger.error(f"❌ EXTRACTION FAILED: {error_msg}")
            safe_error_display(f"❌ Extraction failed: {safe_error_msg}")
            return
        
        if result is None:
            logger.error("❌ NO RESULT: Extraction completed but returned no result")
            safe_error_display("❌ Extraction failed: No result returned")
            return
            
        # Update progress
        progress_bar.progress(80)
        status_text.text("✅ Step 1/3: Content extraction completed!")
        logger.info("✅ STEP 1 COMPLETE: Content extraction finished successfully")
        
        # Validate extracted content
        if not result or not result.get('content'):
            logger.error("❌ VALIDATION: No content in extraction result")
            safe_error_display("❌ Failed to extract content from the website. Please check the URL and try again.")
            return
        
        logger.info(f"✅ VALIDATION: Content extracted successfully ({len(result['content'])} chars)")
        # EMERGENCY: Use st.text instead of st.success to avoid markdown parsing  
        st.text("Content extraction completed!")
        
        # Step 2: Knowledge Extraction
        progress_bar.progress(85)
        status_text.text("🧠 Step 2/3: Extracting domain knowledge with AI...")
        logger.info("🧠 STEP 2: Starting knowledge extraction")
        
        try:
            # Run knowledge extraction in a thread since we can't await in Streamlit
            knowledge_queue = queue.Queue()
            knowledge_error_queue = queue.Queue()
            
            def run_knowledge_extraction():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    async def extract_knowledge():
                        return await extract_domain_knowledge(
                            content=result['content'],
                            url=url
                        )
                    
                    domain_knowledge = loop.run_until_complete(extract_knowledge())
                    knowledge_queue.put(domain_knowledge)
                    
                except Exception as ke:
                    log_error_with_traceback("Knowledge extraction thread failed", ke)
                    knowledge_error_queue.put(str(ke))
            
            knowledge_thread = threading.Thread(target=run_knowledge_extraction)
            knowledge_thread.start()
            knowledge_thread.join(timeout=30)
            
            if not knowledge_queue.empty():
                domain_knowledge = knowledge_queue.get()
                logger.info("✅ STEP 2 COMPLETE: Domain knowledge extracted successfully")
            elif not knowledge_error_queue.empty():
                error_msg = knowledge_error_queue.get()
                safe_error_msg = sanitize_markdown_content(str(error_msg))
                logger.error(f"❌ KNOWLEDGE EXTRACTION FAILED: {error_msg}")
                safe_error_display(f"❌ Knowledge extraction failed: {safe_error_msg}")
                return
            else:
                logger.error("❌ KNOWLEDGE EXTRACTION TIMEOUT")
                safe_error_display("❌ Knowledge extraction timed out")
                return
                
        except Exception as knowledge_e:
            log_error_with_traceback("Knowledge extraction failed", knowledge_e)
            safe_error_msg = sanitize_markdown_content(str(knowledge_e))
            safe_error_display(f"❌ Knowledge extraction failed: {safe_error_msg}")
            return
        
        # Step 3: Agent Creation  
        progress_bar.progress(95)
        status_text.text("🤖 Step 3/3: Creating your specialized agent...")
        logger.info("🤖 STEP 3: Starting agent creation")
        
        try:
            # Run agent creation in a thread since we can't await in Streamlit
            agent_queue = queue.Queue()
            agent_error_queue = queue.Queue()
            
            def run_agent_creation():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    async def create_agent():
                        return await create_domain_agent(domain_knowledge)
                    
                    agent = loop.run_until_complete(create_agent())
                    agent_queue.put(agent)
                    
                except Exception as ae:
                    log_error_with_traceback("Agent creation thread failed", ae)
                    agent_error_queue.put(str(ae))
            
            agent_thread = threading.Thread(target=run_agent_creation)
            agent_thread.start()
            agent_thread.join(timeout=30)
            
            if not agent_queue.empty():
                agent = agent_queue.get()
                logger.info("✅ STEP 3 COMPLETE: Agent created successfully")
            elif not agent_error_queue.empty():
                error_msg = agent_error_queue.get()
                safe_error_msg = sanitize_markdown_content(str(error_msg))
                logger.error(f"❌ AGENT CREATION FAILED: {error_msg}")
                safe_error_display(f"❌ Agent creation failed: {safe_error_msg}")
                return
            else:
                logger.error("❌ AGENT CREATION TIMEOUT")
                safe_error_display("❌ Agent creation timed out")
                return
                
        except Exception as agent_e:
            log_error_with_traceback("Agent creation failed", agent_e)
            safe_error_msg = sanitize_markdown_content(str(agent_e))
            safe_error_display(f"❌ Agent creation failed: {safe_error_msg}")
            return
        
        # Complete!
        progress_bar.progress(100)
        status_text.text("🎉 All steps completed! Your agent is ready.")
        logger.info("🎉 WORKFLOW COMPLETE: All steps finished successfully")
        
        # Store in session state
        st.session_state.domain_knowledge = domain_knowledge
        st.session_state.domain_agent = agent
        st.session_state.extraction_status = "completed"
        
        # Show success message
        # EMERGENCY: Use st.text instead of st.success/st.info to avoid markdown parsing
        st.text("Agent Created Successfully!")
        st.text("Your specialized AI agent is ready. You can now chat with it using the interface below!")
        logger.info("✅ WORKFLOW SUCCESS: Agent stored in session state and ready for chat")
        
    except Exception as e:
        log_error_with_traceback("Main workflow failed", e)
        safe_error_msg = sanitize_markdown_content(str(e))
        safe_error_display(f"❌ Error: {safe_error_msg}")
        logger.error(f"❌ WORKFLOW ERROR: {str(e)}")
        st.session_state.extraction_status = "failed"

def display_sidebar():
    """Display the sidebar with input controls"""
    logger.info("📋 UI: Rendering sidebar")
    
    with st.sidebar:
        st.header("Create your agent")
        
        # URL input
        url = st.text_input(
            "Enter website URL",
            value="",
            placeholder="https://example.com"
        )
        
        # Max pages slider
        max_urls = st.slider(
            "Maximum pages to analyze",
            min_value=1,
            max_value=50,
            value=5
        )
        
        # Store values in session state
        st.session_state.url = url
        st.session_state.max_urls = max_urls
        
        # Create agent button
        if st.button("🚀 Create Agent", type="primary", disabled=not url):
            logger.info(f"🚀 BUTTON CLICK: Create Agent button pressed for URL: {url}")
            try:
                st.session_state.extraction_status = "running"
                st.session_state.messages = []  # Clear previous messages
                logger.info("✅ BUTTON: Session state updated, starting workflow")
            except Exception as button_e:
                log_error_with_traceback("Button click handler failed", button_e)

def get_streaming_response(agent, message):
    """Get streaming response from agent (placeholder for now)"""
    logger.info(f"💬 CHAT: Getting streaming response for message: {message[:50]}...")
    try:
        # This would be replaced with actual streaming logic
        response = f"I understand you're asking about: {message}. Based on the website content I analyzed, I can help you with that."
        logger.info("✅ CHAT: Streaming response generated successfully")
        return response
    except Exception as e:
        log_error_with_traceback("Streaming response failed", e)
        raise

def get_non_streaming_response(agent, message):
    """Get non-streaming response from agent"""
    logger.info(f"💬 CHAT FALLBACK: Getting non-streaming response for: {message[:50]}...")
    try:
        # Placeholder implementation
        response = f"Based on the website analysis, I can help you with: {message}"
        logger.info("✅ CHAT FALLBACK: Non-streaming response generated")
        return response
    except Exception as e:
        log_error_with_traceback("Non-streaming response failed", e)
        raise

def display_chat_interface():
    """Display chat interface for interacting with the domain agent."""
    logger.info("💬 UI: Rendering chat interface")
    
    # Display chat history
    for i, message in enumerate(st.session_state.messages):
        try:
            with st.chat_message(message["role"]):
                # Use safe markdown display with automatic fallback
                if not safe_markdown_display(message["content"], fallback_to_text=True):
                    log_error_with_traceback(f"Chat message {i} could not be displayed safely", None)
        except Exception as msg_e:
            log_error_with_traceback(f"Chat message {i} display failed", msg_e)

    # Chat input
    if prompt := st.chat_input("Ask me anything about the website..."):
        logger.info(f"💬 CHAT INPUT: User sent message: {prompt[:100]}...")
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            if not safe_markdown_display(prompt, fallback_to_text=True):
                log_error_with_traceback("User prompt could not be displayed safely", None)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            try:
                logger.info("🤖 CHAT: Generating assistant response")
                message_placeholder = st.empty()
                full_response = ""
                
                try:
                    # Try streaming response first
                    logger.info("🔄 CHAT: Attempting streaming response")
                    streaming_response = get_streaming_response(st.session_state.domain_agent, prompt)
                    
                    # Simulate streaming by showing chunks (use text to avoid markdown issues)
                    words = streaming_response.split()
                    for i, word in enumerate(words):
                        full_response += word + " "
                        # Use text display for streaming to avoid markdown parsing during animation
                        message_placeholder.text(full_response + "▌")
                        time.sleep(0.05)  # Simulate streaming delay
                    
                    # Final response without cursor
                    message_placeholder.empty()
                    if not safe_markdown_display(full_response, fallback_to_text=True):
                        log_error_with_traceback("Final response could not be displayed safely", None)
                    logger.info("✅ CHAT: Streaming response completed successfully")
                    
                except Exception as streaming_e:
                    log_error_with_traceback("Streaming response failed, trying fallback", streaming_e)
                    try:
                        logger.info("🔄 CHAT FALLBACK: Using non-streaming response...")
                        full_response = get_non_streaming_response(st.session_state.domain_agent, prompt)
                        if not safe_markdown_display(full_response, fallback_to_text=True):
                            log_error_with_traceback("Fallback response could not be displayed safely", None)
                        logger.info("✅ CHAT FALLBACK COMPLETE: Non-streaming response delivered")
                    except Exception as e2:
                        log_error_with_traceback("Chat fallback failed", e2)
                        safe_error_msg = sanitize_markdown_content(str(e2))
                        safe_error_display(f"Error generating response: {safe_error_msg}")
                        full_response = "I apologize, but I encountered an error generating a response."

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                logger.info("✅ CHAT: Response added to chat history")
                
            except Exception as chat_e:
                log_error_with_traceback("Chat interface critical error", chat_e)
                safe_error_msg = sanitize_markdown_content(str(chat_e))
                safe_error_display(f"Critical chat error: {safe_error_msg}")

def run_app():
    """Main application entry point"""
    logger.info("🚀 APP START: Application starting up")
    
    try:
        # Page config is handled in app.py - don't duplicate it here
        
        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = []
            logger.info("📝 INIT: Chat messages initialized")
            
        if "extraction_status" not in st.session_state:
            st.session_state.extraction_status = "idle"
            logger.info("📝 INIT: Extraction status initialized")
            
        if "domain_knowledge" not in st.session_state:
            st.session_state.domain_knowledge = None
            logger.info("📝 INIT: Domain knowledge initialized")
            
        if "domain_agent" not in st.session_state:
            st.session_state.domain_agent = None
            logger.info("📝 INIT: Domain agent initialized")
        
        # Main header
        st.title("The Laura Natalia Gonzalez Chat Bot")
        st.subheader("A chatbot trained in the likes of the most incredible person on Earth")
        st.write("Until I figure out the training data, you can throw a website in, crawl it, and generate an SME on that website")
        
        # Display sidebar
        display_sidebar()
        
        # Handle workflow execution
        if st.session_state.extraction_status == "running":
            logger.info("🔄 WORKFLOW: Status is running, executing workflow")
            run_workflow()
        elif st.session_state.extraction_status == "completed":
            logger.info("✅ WORKFLOW: Status is completed, showing chat interface")
            # EMERGENCY: Use st.text instead of st.success to avoid any markdown parsing
            st.text("Agent Ready! You can now chat with your specialized assistant.")
            display_chat_interface()
        else:
            logger.info("💭 STANDBY: Showing welcome message")
            # EMERGENCY: Use st.text instead of st.info to avoid any markdown parsing
            st.text("Welcome! Enter a website URL in the sidebar, and I'll transform it into an AI agent you can chat with.")
            
        logger.info("✅ APP: Application rendering completed successfully")
        
    except Exception as app_e:
        log_error_with_traceback("Application critical error", app_e)
        safe_error_display("🚨 Critical application error occurred. Check server logs for details.")

if __name__ == "__main__":
    logger.info("🎬 MAIN: Starting application from main")
    run_app()
