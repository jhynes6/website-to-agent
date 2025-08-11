from datetime import datetime
from typing import List
import logging
import json
import os
import re
from openai import AsyncOpenAI

from src.models import Concept, Terminology, Insight, DomainKnowledge
from src.config import DEFAULT_MODEL, OPENAI_API_KEY

# Set up logging
logger = logging.getLogger('website-to-agent')

# Initialize OpenAI client with official OpenAI endpoints
client = AsyncOpenAI(
    api_key=OPENAI_API_KEY
)

def estimate_token_count(text: str) -> int:
    """Estimate token count (rough approximation: ~4 chars = 1 token)"""
    return len(text) // 4

def trim_content_intelligently(content: str, max_tokens: int) -> str:
    """
    Trim content to fit within token limit while preserving important information.
    
    Strategy:
    1. Keep the beginning (usually contains key info)
    2. Keep some middle sections  
    3. Keep the end (often has conclusions)
    4. Remove excessive whitespace and repetitive content
    """
    logger.info(f"🔧 TRIMMING: Original content ~{estimate_token_count(content)} tokens, target: {max_tokens}")
    
    # Target character count (rough conversion from tokens)
    max_chars = max_tokens * 4
    
    if len(content) <= max_chars:
        return content
    
    # Clean up excessive whitespace first
    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)  # Remove excessive newlines
    content = re.sub(r' +', ' ', content)  # Remove excessive spaces
    
    if len(content) <= max_chars:
        return content
    
    # If still too long, use intelligent sectioning
    # Keep 60% from beginning, 20% from middle, 20% from end
    begin_chars = int(max_chars * 0.6)
    middle_chars = int(max_chars * 0.2) 
    end_chars = int(max_chars * 0.2)
    
    beginning = content[:begin_chars]
    
    # Get middle section
    middle_start = len(content) // 2 - middle_chars // 2
    middle_end = middle_start + middle_chars
    middle = content[middle_start:middle_end]
    
    # Get end section
    ending = content[-end_chars:]
    
    trimmed = f"{beginning}\n\n[... CONTENT TRIMMED FOR ANALYSIS ...]\n\n{middle}\n\n[... CONTENT TRIMMED FOR ANALYSIS ...]\n\n{ending}"
    
    logger.info(f"✂️ TRIMMED: Content reduced to ~{estimate_token_count(trimmed)} tokens")
    return trimmed

async def extract_domain_knowledge(content: str, url: str) -> DomainKnowledge:
    """
    Extract structured domain knowledge from website content using OpenAI.
    
    Args:
        content: The extracted website content (llmstxt or llmsfulltxt)
        url: Source URL for reference
        
    Returns:
        Structured DomainKnowledge object
    """
    logger.info(f"🔍 Starting knowledge extraction for URL: {url}")
    logger.info(f"📄 Content length: {len(content)} characters")
    
    # Create prompt for knowledge extraction
    prompt = f"""Extract comprehensive domain knowledge from the provided website content.

Identify and structure:
1. Core concepts and their relationships
2. Specialized terminology and definitions  
3. Key insights and principles

For each concept, assess its centrality/importance to the domain.
For terminology, provide clear definitions and examples when available.
For insights, evaluate confidence based on how explicitly they're stated.

Return your response as a JSON object with this exact structure:
{{
    "core_concepts": [
        {{
            "name": "concept name",
            "description": "detailed description",
            "related_concepts": ["related concept 1", "related concept 2"],
            "centrality": 0.8
        }}
    ],
    "terminology": [
        {{
            "term": "technical term",
            "definition": "clear definition",
            "context": "where this term is used",
            "examples": ["example1", "example2"]
        }}
    ],
    "key_insights": [
        {{
            "insight": "key insight or principle",
            "topics": ["topic1", "topic2"],
            "confidence": 0.9
        }}
    ]
}}

Website content to analyze:
{content}

Source: {url}"""
    
    # Run the extraction (with intelligent retry on context length error)
    logger.info("🤖 Running knowledge extraction with OpenAI...")
    
    # Check estimated token count first
    estimated_tokens = estimate_token_count(prompt)
    logger.info(f"📏 Estimated prompt tokens: {estimated_tokens}")
    
    original_content = content
    attempt = 1
    max_attempts = 3
    
    while attempt <= max_attempts:
        try:
            logger.info(f"🔄 Attempt {attempt}/{max_attempts} - Content length: {len(content)} chars")
            
            # Recreate prompt with potentially trimmed content
            if attempt > 1:
                # Calculate how much we need to trim (leave room for prompt template + response)
                max_content_tokens = 120000  # Conservative limit leaving room for prompt template
                content = trim_content_intelligently(original_content, max_content_tokens)
                
                prompt = f"""Extract comprehensive domain knowledge from the provided website content.

Identify and extract:
1. Core concepts and their relationships
2. Key terminology and definitions  
3. Important insights or principles

For each concept, assess its centrality/importance to the domain.
For terminology, provide clear definitions and examples when available.
For insights, evaluate confidence based on how explicitly they're stated.

Return your response as a JSON object with this exact structure:
{{
    "core_concepts": [
        {{
            "name": "concept name",
            "description": "detailed description",
            "related_concepts": ["related concept 1", "related concept 2"],
            "centrality": 0.8
        }}
    ],
    "terminology": [
        {{
            "term": "technical term",
            "definition": "clear definition",
            "context": "where this term is used",
            "examples": ["example1", "example2"]
        }}
    ],
    "key_insights": [
        {{
            "insight": "key insight or principle",
            "topics": ["topic1", "topic2"],
            "confidence": 0.9
        }}
    ]
}}

Website content to analyze:
{content}

Source: {url}"""
            
            response = await client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert knowledge extractor. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=4096,
            )
            
            # If we get here, the API call was successful
            break
            
        except Exception as api_error:
            error_message = str(api_error)
            
            # Check if it's a context length error
            if "context_length_exceeded" in error_message or "maximum context length" in error_message:
                logger.warning(f"⚠️ CONTEXT LENGTH ERROR (attempt {attempt}): {error_message}")
                
                if attempt < max_attempts:
                    logger.info(f"🔄 RETRYING: Will trim content and try again...")
                    attempt += 1
                    continue
                else:
                    logger.error(f"❌ FINAL ATTEMPT FAILED: Unable to fit content within context limit after {max_attempts} attempts")
                    raise api_error
            else:
                # Different error - don't retry
                logger.error(f"❌ API ERROR: {error_message}")
                raise api_error
    
    # Parse the response (outside the retry loop)
    try:
        content_text = response.choices[0].message.content.strip()
        
        # Try to extract JSON from the response
        try:
            # Find JSON in the response (in case there's extra text)
            start = content_text.find('{')
            end = content_text.rfind('}') + 1
            if start >= 0 and end > start:
                json_text = content_text[start:end]
            else:
                json_text = content_text
                
            parsed_data = json.loads(json_text)
            
            # Create domain knowledge object
            domain_knowledge = DomainKnowledge(
                core_concepts=[
                    Concept(
                        name=c.get("name", "Unknown"),
                        description=c.get("description", ""),
                        related_concepts=c.get("related_concepts", []),
                        importance_score=c.get("centrality", 0.5)
                    ) for c in parsed_data.get("core_concepts", [])
                ],
                terminology=[
                    Terminology(
                        term=t.get("term", "Unknown"),
                        definition=t.get("definition", ""),
                        context=t.get("context", ""),
                        examples=t.get("examples", [])
                    ) for t in parsed_data.get("terminology", [])
                ],
                key_insights=[
                    Insight(
                        content=i.get("insight", ""),
                        topics=i.get("topics", []),
                        confidence=i.get("confidence", 0.5)
                    ) for i in parsed_data.get("key_insights", [])
                ],
                source_url=url,
                extraction_timestamp=datetime.now().isoformat()
            )
            
            logger.info("✅ Knowledge extraction completed successfully")
            logger.info(f"📊 Extracted: {len(domain_knowledge.core_concepts)} concepts, {len(domain_knowledge.terminology)} terms, {len(domain_knowledge.key_insights)} insights")
            
            return domain_knowledge
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse JSON response: {e}")
            logger.error(f"Raw response: {content_text[:500]}...")
            
            # Return a basic structure if JSON parsing fails
            return DomainKnowledge(
                core_concepts=[Concept(name="Content Analysis", description=f"Analysis of {url}", related_concepts=[], importance_score=1.0)],
                terminology=[],
                key_insights=[Insight(content="Content successfully extracted but detailed analysis failed", topics=[], confidence=0.5)],
                source_url=url,
                extraction_timestamp=datetime.now().isoformat()
            )
            
    except Exception as e:
        logger.error(f"❌ Knowledge extraction failed: {str(e)}")
        
        # Return a basic structure on error
        return DomainKnowledge(
            core_concepts=[Concept(name="Website Content", description=f"Content from {url}", related_concepts=[], importance_score=1.0)],
            terminology=[],
            key_insights=[Insight(content="Content extraction completed but analysis failed", topics=[], confidence=0.3)],
            source_url=url,
            extraction_timestamp=datetime.now().isoformat()
        )

class DomainAgent:
    """Simple domain agent that uses OpenAI's official API."""
    
    def __init__(self, domain_knowledge: DomainKnowledge):
        self.domain_knowledge = domain_knowledge
        self.system_prompt = self._create_system_prompt()
        logger.info(f"🏗️ Creating specialized agent for domain: {domain_knowledge.source_url}")
        logger.info(f"📊 Concepts extracted: {len(domain_knowledge.core_concepts)}")
    
    def _create_system_prompt(self) -> str:
        """Create system prompt from domain knowledge."""
        domain_name = self.domain_knowledge.core_concepts[0].name if self.domain_knowledge.core_concepts else "this website"
        
        return f"""You are a knowledgeable AI assistant specializing in {domain_name}. You have been trained on comprehensive content from {self.domain_knowledge.source_url} and possess deep understanding of this domain.

## YOUR KNOWLEDGE BASE

### Core Concepts and Areas of Expertise:
{_format_concepts(self.domain_knowledge.core_concepts)}

### Specialized Terminology:
{_format_terminology(self.domain_knowledge.terminology)}

### Key Insights and Principles:
{_format_insights(self.domain_knowledge.key_insights)}

## YOUR RESPONSE GUIDELINES

1. **Be Authoritative**: You are the expert on this domain. Provide confident, detailed answers based on your knowledge.

2. **Use Your Knowledge**: Always start with information from your specialized knowledge base. Reference specific concepts, terminology, and insights that are relevant.

3. **Be Practical**: When possible, provide actionable advice, practical examples, or specific steps the user can take.

4. **Structure Your Responses**: Use clear formatting with headings, bullet points, and numbered lists to make complex information digestible.

5. **Show Your Expertise**: Reference relevant terminology and concepts naturally in your responses to demonstrate domain knowledge.

6. **Be Honest About Limitations**: If asked about something not covered in your knowledge base, acknowledge this clearly and suggest related topics you can help with.

7. **Connect Ideas**: When relevant, explain how different concepts, insights, or terminology relate to each other and to the user's question.

8. **Provide Context**: When referencing your source material, mention that this information comes from your analysis of {self.domain_knowledge.source_url}.

## RESPONSE STYLE
- Be conversational but professional
- Use specific examples when possible
- Break down complex topics into understandable parts
- Always aim to provide value and actionable information
- Make connections between different aspects of the domain

Remember: You are not just answering questions - you are sharing specialized expertise to help users understand and work with {domain_name} effectively."""

    async def chat(self, message: str) -> str:
        """Chat with the domain agent."""
        try:
            response = await client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": message}
                ],
                temperature=0.3,
                max_tokens=4096,  # Increased for more detailed responses
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"❌ Agent chat failed: {str(e)}")
            return f"I apologize, but I encountered an error while processing your question: {str(e)}"

def create_domain_agent(domain_knowledge: DomainKnowledge) -> DomainAgent:
    """
    Create a specialized agent based on extracted domain knowledge.
    
    Args:
        domain_knowledge: Structured domain knowledge
        
    Returns:
        DomainAgent instance
    """
    return DomainAgent(domain_knowledge)

def _format_concepts(concepts: List[Concept]) -> str:
    """Format concepts for agent instructions."""
    if not concepts:
        return "No specific concepts were identified from the source material."
    
    formatted = ""
    for i, concept in enumerate(concepts, 1):
        importance_indicator = "⭐" * min(3, max(1, int(concept.importance_score * 3)))
        formatted += f"{i}. **{concept.name}** {importance_indicator}\n"
        formatted += f"   {concept.description}\n"
        if concept.related_concepts:
            formatted += f"   Related topics: {', '.join(concept.related_concepts)}\n"
        formatted += "\n"
    return formatted.strip()

def _format_terminology(terminology: List[Terminology]) -> str:
    """Format terminology for agent instructions."""
    if not terminology:
        return "No specialized terminology was identified from the source material."
    
    formatted = ""
    for i, term_info in enumerate(terminology, 1):
        formatted += f"{i}. **{term_info.term}**\n"
        formatted += f"   Definition: {term_info.definition}\n"
        if term_info.context:
            formatted += f"   Context: {term_info.context}\n"
        if term_info.examples:
            formatted += f"   Examples: {'; '.join(term_info.examples)}\n"
        formatted += "\n"
    return formatted.strip()

def _format_insights(insights: List[Insight]) -> str:
    """Format insights for agent instructions."""
    if not insights:
        return "No key insights were identified from the source material."
    
    formatted = ""
    for i, insight in enumerate(insights, 1):
        confidence_indicator = "🔥" if insight.confidence > 0.8 else "💡" if insight.confidence > 0.6 else "💭"
        formatted += f"{i}. {confidence_indicator} {insight.content}\n"
        if insight.topics:
            formatted += f"   Related to: {', '.join(insight.topics)}\n"
        formatted += "\n"
    return formatted.strip()
