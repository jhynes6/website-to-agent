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
        # Restore proper success messaging 
        st.success("🕷️ Starting website analysis with Simple Scraper...")
        logger.info("📊 STEP 1: Starting content extraction")
        
        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Update progress
        status_text.text("Step 1/3: Discovering website pages...")  # No emojis
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
            (20, "Step 1/3: Analyzing website structure..."),  # No emojis
            (40, "Step 1/3: Extracting content from pages..."),  # No emojis  
            (60, "Step 1/3: Processing extracted content..."),  # No emojis
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
        status_text.text("Step 1/3: Content extraction completed!")  # No emojis
        logger.info("✅ STEP 1 COMPLETE: Content extraction finished successfully")
        
        # Validate extracted content - handle graceful failures
        content = result.get('llmstxt', '') or result.get('content', '')
        if not result or not content:
            logger.error("❌ VALIDATION: No content in extraction result")
            safe_error_display("❌ Failed to extract content from the website. Please check the URL and try again.")
            return
        
        # Check if this was an extraction error (graceful failure)
        if result.get('extraction_error'):
            logger.warning(f"⚠️ EXTRACTION WARNING: {result.get('extraction_error')}")
            st.text("⚠️ Content extraction completed with warnings - proceeding with available content")
        else:
            logger.info(f"✅ VALIDATION: Content extracted successfully ({len(content)} chars)")
            # Restore proper success messaging
            st.success("✅ Content extraction completed!")
        
        # Step 2: Knowledge Extraction
        progress_bar.progress(85)
        status_text.text("Step 2/3: Extracting domain knowledge with AI...")  # No emojis
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
                            content=content,
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
                
                # 🎯 DISPLAY LLM ANALYSIS SUMMARY - RESTORED!
                st.success("✅ Domain knowledge extracted successfully!")
                
                # Show analysis metrics
                st.subheader("📊 AI Analysis Results")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Core Concepts", len(domain_knowledge.core_concepts))
                with col2:
                    st.metric("Key Terms", len(domain_knowledge.terminology))
                with col3:
                    st.metric("Insights", len(domain_knowledge.key_insights))
                
                # Debug info (can remove when stable)
                # st.write(f"DEBUG - Core concepts found: {len(domain_knowledge.core_concepts)}")
                # st.write(f"DEBUG - Key insights found: {len(domain_knowledge.key_insights)}")
                # st.write(f"DEBUG - Terminology found: {len(domain_knowledge.terminology)}")
                
                # Show key concepts found
                if domain_knowledge.core_concepts:
                    with st.expander("🎯 Key Concepts Discovered", expanded=True):
                        for i, concept in enumerate(domain_knowledge.core_concepts[:5]):
                            st.write(f"**Concept {i+1}: {concept.name}**")
                            st.write(f"Description: {concept.description[:300]}")
                            if concept.related_concepts:
                                st.write(f"Related: {', '.join(concept.related_concepts[:3])}")
                            st.write("---")
                else:
                    st.warning("⚠️ No core concepts were extracted from the website")
                
                # Show key insights 
                if domain_knowledge.key_insights:
                    with st.expander("💡 Key Insights Found", expanded=True):
                        for i, insight in enumerate(domain_knowledge.key_insights[:3]):
                            st.write(f"**Insight {i+1}:**")
                            st.write(f"{insight.content[:400]}")
                            if insight.topics:
                                st.write(f"Topics: {', '.join(insight.topics)}")
                            st.write(f"Confidence: {insight.confidence:.1%}")
                            st.write("---")
                else:
                    st.warning("⚠️ No key insights were extracted from the website")
                    
                # Show some terminology too
                if domain_knowledge.terminology:
                    with st.expander("📚 Key Terms Found", expanded=False):
                        for i, term in enumerate(domain_knowledge.terminology[:5]):
                            st.write(f"**{term.term}**: {term.definition[:200]}")
                            if term.examples:
                                st.write(f"Examples: {', '.join(term.examples[:2])}")
                            st.write("---")
                
                st.info("🤖 Now creating your specialized AI agent...")
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
        status_text.text("Step 3/3: Creating your specialized agent...")  # No emojis
        logger.info("🤖 STEP 3: Starting agent creation")
        
        try:
            # Run agent creation in a thread since we can't await in Streamlit
            agent_queue = queue.Queue()
            agent_error_queue = queue.Queue()
            
            def run_agent_creation():
                try:
                    # create_domain_agent is synchronous, no async needed
                    agent = create_domain_agent(domain_knowledge)
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
        # Restore proper success messaging
        st.success("🎉 Agent Created Successfully!")
        st.info("🤖 Your specialized AI agent is ready. You can now chat with it using the interface below!")
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
        # EMERGENCY: Replace st.header() with st.text() to avoid ReactMarkdown
        st.text("CREATE YOUR AGENT")  # No markdown processing
        
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

async def get_streaming_response(agent, message):
    """Get streaming response from agent using actual OpenAI API"""
    logger.info(f"💬 CHAT: Getting streaming response for message: {message[:50]}...")
    try:
        if hasattr(agent, 'chat'):
            response = await agent.chat(message)
            logger.info("✅ CHAT: Streaming response generated successfully")
            return response
        else:
            logger.warning("🚨 CHAT: Agent doesn't have chat method, using fallback")
            return f"I apologize, but I'm not properly configured yet. Please recreate the agent."
    except Exception as e:
        log_error_with_traceback("Streaming response failed", e)
        raise

def run_async_chat_response(agent, message, result_queue, error_queue):
    """Run the async chat response in a separate thread"""
    logger.info(f"🔧 CHAT THREAD START: Starting chat thread for message: {message[:50]}...")
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        logger.info("✅ CHAT THREAD: Event loop created successfully")
        
        async def chat():
            logger.info(f"🚀 ASYNC CHAT START: Beginning chat response")
            try:
                if hasattr(agent, 'chat'):
                    result = await agent.chat(message)
                    logger.info(f"✅ ASYNC CHAT SUCCESS: Chat completed")
                    return result
                else:
                    logger.warning("🚨 ASYNC CHAT: Agent missing chat method")
                    return "I apologize, but I'm not properly configured. Please recreate the agent."
            except Exception as async_e:
                log_error_with_traceback(f"Async chat failed", async_e)
                raise
        
        result = loop.run_until_complete(chat())
        logger.info(f"📤 CHAT THREAD: Putting result in queue (size: {len(str(result))} chars)")
        result_queue.put(result)
        logger.info("✅ CHAT THREAD: Result successfully queued")
        
    except Exception as thread_e:
        log_error_with_traceback(f"Chat thread failed", thread_e)
        error_queue.put(str(thread_e))
        logger.error(f"📤 CHAT THREAD: Error queued: {str(thread_e)}")
    finally:
        logger.info("🔧 CHAT THREAD END: Chat thread completed")

def get_non_streaming_response(agent, message):
    """Get non-streaming response from agent using threading for async calls"""
    logger.info(f"💬 CHAT FALLBACK: Getting non-streaming response for: {message[:50]}...")
    try:
        result_queue = queue.Queue()
        error_queue = queue.Queue()
        
        chat_thread = threading.Thread(
            target=run_async_chat_response,
            args=(agent, message, result_queue, error_queue)
        )
        chat_thread.start()
        chat_thread.join(timeout=30)
        
        if not result_queue.empty():
            response = result_queue.get()
            logger.info("✅ CHAT FALLBACK COMPLETE: Non-streaming response delivered")
            return response
        elif not error_queue.empty():
            error_msg = error_queue.get()
            logger.error(f"❌ CHAT ERROR: {error_msg}")
            return f"I apologize, but I encountered an error: {error_msg}"
        else:
            logger.error("❌ CHAT TIMEOUT")
            return "I apologize, but my response timed out. Please try again."
            
    except Exception as e:
        log_error_with_traceback("Non-streaming response failed", e)
        raise

def display_chat_interface():
    """Display chat interface for interacting with the domain agent (only when agent is ready)."""
    logger.info("💬 UI: Rendering chat interface")
    
    # Debug info (can remove when stable)
    # st.write("🐛 DEBUG: Chat interface is rendering!")
    # st.write(f"DEBUG: Messages in history: {len(st.session_state.messages)}")
    # st.write(f"DEBUG: Domain agent exists: {st.session_state.domain_agent is not None}")
    
    # Display chat history
    for i, message in enumerate(st.session_state.messages):
        try:
            with st.chat_message(message["role"]):
                # Use safe markdown display with automatic fallback
                if not safe_markdown_display(message["content"], fallback_to_text=True):
                    log_error_with_traceback(f"Chat message {i} could not be displayed safely", None)
        except Exception as msg_e:
            log_error_with_traceback(f"Chat message {i} display failed", msg_e)

    # Debug info (can remove when stable)
    # st.write("🐛 DEBUG: About to render chat input...")
    
    # Chat input - MAKE SURE THIS RENDERS
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
                    logger.info("🔄 CHAT: Using non-streaming response (since streaming requires async context)...")
                    full_response = get_non_streaming_response(st.session_state.domain_agent, prompt)
                    
                    if full_response:
                        # Simulate streaming by showing chunks (use text to avoid markdown issues)
                        words = full_response.split()
                        for i, word in enumerate(words):
                            partial_response = " ".join(words[:i+1])
                            # Use text display for streaming to avoid markdown parsing during animation
                            message_placeholder.text(partial_response + "▌")
                            time.sleep(0.05)  # Simulate streaming delay
                        
                        # Final response without cursor
                        message_placeholder.empty()
                        if not safe_markdown_display(full_response, fallback_to_text=True):
                            log_error_with_traceback("Final response could not be displayed safely", None)
                        logger.info("✅ CHAT: Response completed successfully")
                    else:
                        logger.error("❌ CHAT: No response received")
                        full_response = "I apologize, but I couldn't generate a response. Please try again."
                        safe_error_display("No response was generated. Please try again.")
                    
                except Exception as e:
                    log_error_with_traceback("Chat response failed", e)
                    safe_error_msg = sanitize_markdown_content(str(e))
                    safe_error_display(f"Error generating response: {safe_error_msg}")
                    full_response = "I apologize, but I encountered an error generating a response."

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                logger.info("✅ CHAT: Response added to chat history")
                
            except Exception as chat_e:
                log_error_with_traceback("Chat interface critical error", chat_e)
                safe_error_msg = sanitize_markdown_content(str(chat_e))
                safe_error_display(f"Critical chat error: {safe_error_msg}")

def display_chat_interface_always():
    """Display chat interface that's always visible but handles different states."""
    logger.info("💬 UI: Rendering always-visible chat interface")
    
    # Create a separator
    st.markdown("---")
    st.subheader("💬 Chat with your AI Agent")
    
    # Check if agent is ready
    agent_ready = (st.session_state.extraction_status == "completed" and 
                   st.session_state.domain_agent is not None)
    
    if not agent_ready:
        # Show placeholder when no agent
        if st.session_state.extraction_status == "running":
            st.info("🔄 Please wait while your agent is being created...")
            st.chat_input("Chat will be available once your agent is ready...", disabled=True)
        elif st.session_state.extraction_status == "failed":
            st.error("❌ Agent creation failed. Please try creating a new agent.")
            st.chat_input("Please create a new agent to enable chat...", disabled=True)
        else:
            st.info("👋 Create an agent using the sidebar to start chatting!")
            st.chat_input("Please create an agent first...", disabled=True)
        return
    
    # Agent is ready - show full chat interface
    st.success("🤖 Agent is ready! Ask me anything about the website content.")
    
    # Display chat history
    for i, message in enumerate(st.session_state.messages):
        try:
            with st.chat_message(message["role"]):
                if not safe_markdown_display(message["content"], fallback_to_text=True):
                    log_error_with_traceback(f"Chat message {i} could not be displayed safely", None)
        except Exception as msg_e:
            log_error_with_traceback(f"Chat message {i} display failed", msg_e)

    # Chat input - ALWAYS RENDERED when agent is ready
    if prompt := st.chat_input("Ask me anything about the website..."):
        logger.info(f"💬 CHAT INPUT: User sent message: {prompt[:100]}...")
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message immediately
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
                    logger.info("🔄 CHAT: Using non-streaming response...")
                    full_response = get_non_streaming_response(st.session_state.domain_agent, prompt)
                    
                    if full_response:
                        # Simulate streaming by showing chunks
                        words = full_response.split()
                        for i, word in enumerate(words):
                            partial_response = " ".join(words[:i+1])
                            message_placeholder.text(partial_response + "▌")
                            time.sleep(0.05)
                        
                        # Final response without cursor
                        message_placeholder.empty()
                        if not safe_markdown_display(full_response, fallback_to_text=True):
                            log_error_with_traceback("Final response could not be displayed safely", None)
                        logger.info("✅ CHAT: Response completed successfully")
                    else:
                        logger.error("❌ CHAT: No response received")
                        full_response = "I apologize, but I couldn't generate a response. Please try again."
                        safe_error_display("No response was generated. Please try again.")
                    
                except Exception as e:
                    log_error_with_traceback("Chat response failed", e)
                    safe_error_msg = sanitize_markdown_content(str(e))
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
        
        # EMERGENCY: Replace ALL markdown-processing components with st.text()
        st.text("THE LAURA NATALIA GONZALEZ CHAT BOT")  # No markdown processing
        st.text("A chatbot trained in the likes of the most incredible person on Earth")  # No markdown processing
        st.text("Until I figure out the training data, you can throw a website in, crawl it, and generate an SME on that website")  # No markdown processing
        
        # Display sidebar
        display_sidebar()
        
        # Debug info (can remove when stable)
        # st.write(f"🐛 DEBUG: Current extraction_status = '{st.session_state.extraction_status}'")
        # st.write(f"DEBUG: Domain agent exists = {st.session_state.domain_agent is not None}")
        
        # Handle workflow execution
        if st.session_state.extraction_status == "running":
            logger.info("🔄 WORKFLOW: Status is running, executing workflow")
            run_workflow()
        elif st.session_state.extraction_status == "completed":
            logger.info("✅ WORKFLOW: Status is completed, showing success message")
            st.success("🤖 Agent Ready! You can now chat with your specialized assistant.")
        else:
            logger.info("💭 STANDBY: Showing welcome message")
            st.info("👋 Welcome! Enter a website URL in the sidebar, and I'll transform it into an AI agent you can chat with.")
        
        # ALWAYS show chat interface at bottom (but handle state appropriately)
        display_chat_interface_always()
            
        logger.info("✅ APP: Application rendering completed successfully")
        
    except Exception as app_e:
        log_error_with_traceback("Application critical error", app_e)
        safe_error_display("🚨 Critical application error occurred. Check server logs for details.")

if __name__ == "__main__":
    logger.info("🎬 MAIN: Starting application from main")
    run_app()
