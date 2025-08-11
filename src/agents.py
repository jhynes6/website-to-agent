from datetime import datetime
from typing import List
import logging
import json
import os
from openai import AsyncOpenAI

from src.models import Concept, Terminology, Insight, DomainKnowledge
from src.config import DEFAULT_MODEL, OPENAI_API_KEY

# Set up logging
logger = logging.getLogger('website-to-agent')

# Initialize OpenAI client with official OpenAI endpoints
client = AsyncOpenAI(
    api_key=OPENAI_API_KEY
)

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
    
    # Run the extraction
    logger.info("🤖 Running knowledge extraction with OpenAI...")
    try:
        response = await client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert knowledge extractor. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=4096,
        )
        
        # Parse the response
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
        return f"""You are an expert on {self.domain_knowledge.core_concepts[0].name if self.domain_knowledge.core_concepts else "this domain"} 
with specialized knowledge based on content from {self.domain_knowledge.source_url}.

DOMAIN CONCEPTS:
{_format_concepts(self.domain_knowledge.core_concepts)}

TERMINOLOGY:
{_format_terminology(self.domain_knowledge.terminology)}

KEY INSIGHTS:
{_format_insights(self.domain_knowledge.key_insights)}

When answering questions:
1. Draw on this specialized knowledge first
2. Clearly indicate when you're using information from the source material
3. If asked something outside this domain knowledge, acknowledge the limitations
4. Structure complex answers with headings and bullet points for clarity
5. Refer to the source URL when appropriate

Provide accurate, insightful responses based on this domain knowledge."""

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
                max_tokens=2048,
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
    formatted = ""
    for concept in concepts:
        formatted += f"- {concept.name}: {concept.description}\n"
        if concept.related_concepts:
            formatted += f"  Related: {', '.join(concept.related_concepts)}\n"
    return formatted

def _format_terminology(terminology: List[Terminology]) -> str:
    """Format terminology for agent instructions."""
    formatted = ""
    for term_info in terminology:
        formatted += f"- {term_info.term}: {term_info.definition}\n"
        if term_info.examples:
            formatted += f"  Examples: {'; '.join(term_info.examples)}\n"
    return formatted

def _format_insights(insights: List[Insight]) -> str:
    """Format insights for agent instructions."""
    formatted = ""
    for insight in insights:
        formatted += f"- {insight.content}\n"
    return formatted
