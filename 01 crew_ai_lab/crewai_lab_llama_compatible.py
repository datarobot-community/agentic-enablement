#!/usr/bin/env python3
"""
CrewAI Lab with LLaMA Compatibility Fix
Addresses the "last turn must be user message" error when using LLaMA models
through DataRobot LLM Gateway.
"""

import datarobot as dr
import os
from dotenv import load_dotenv
from pprint import pprint
import random
import openai
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, List, Dict, Any
from ddgs import DDGS
from datetime import datetime
import json
from crewai import Agent, Task, Crew, Process, LLM
from crewai.llm import LLM as BaseLLM
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class LlamaCompatibleLLM(BaseLLM):
    """
    LLaMA-compatible LLM wrapper that ensures conversations end with user messages.
    This fixes the "last turn must be user message" error with LLaMA models.
    """
    
    def __init__(self, model: str, api_key: str, base_url: str, custom_llm_provider: str = "openai"):
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            custom_llm_provider=custom_llm_provider
        )
        self.is_llama_model = self._is_llama_model(model)
        logger.info(f"Initialized LLM: {model}, LLaMA compatible: {self.is_llama_model}")
    
    def _is_llama_model(self, model: str) -> bool:
        """Check if the model is a LLaMA variant."""
        llama_indicators = [
            'llama', 'meta.llama', 'meta/llama', 'llama-', 'llama3', 'llama2'
        ]
        return any(indicator in model.lower() for indicator in llama_indicators)
    
    def _ensure_user_message_last(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Ensure the conversation ends with a user message for LLaMA compatibility.
        """
        if not messages:
            return messages
            
        # Check if last message is from user
        last_message = messages[-1]
        if last_message.get('role') == 'user':
            logger.debug("Conversation already ends with user message")
            return messages
        
        # If last message is from assistant, add a continuation prompt
        if last_message.get('role') == 'assistant':
            logger.info("Adding continuation prompt to ensure user message is last")
            continuation_message = {
                "role": "user",
                "content": "Please continue with your response based on the information provided above."
            }
            return messages + [continuation_message]
        
        return messages
    
    def _validate_and_fix_conversation(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate and fix conversation format for LLaMA models.
        """
        if not self.is_llama_model:
            return messages
        
        # Ensure conversation ends with user message
        fixed_messages = self._ensure_user_message_last(messages)
        
        # Log the conversation structure for debugging
        logger.debug(f"Conversation structure: {[(msg.get('role'), len(msg.get('content', ''))) for msg in fixed_messages]}")
        
        return fixed_messages
    
    def call(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """
        Override the call method to fix message format for LLaMA models.
        """
        try:
            # Fix conversation format if needed
            fixed_messages = self._validate_and_fix_conversation(messages)
            
            # Call parent implementation with fixed messages
            return super().call(fixed_messages, **kwargs)
            
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            # If we get the specific LLaMA error, try to fix it
            if "user message" in str(e).lower() and "last turn" in str(e).lower():
                logger.info("Detected LLaMA format error, attempting to fix...")
                
                # Add a simple user message at the end
                fixed_messages = messages + [{
                    "role": "user",
                    "content": "Please provide your response."
                }]
                
                try:
                    return super().call(fixed_messages, **kwargs)
                except Exception as retry_error:
                    logger.error(f"Retry failed: {str(retry_error)}")
                    raise retry_error
            else:
                raise e

class SearchInput(BaseModel):
    """Input schema for DuckDuckGo search tool."""
    query: str = Field(..., description="The search query")

class LlamaCompatibleDuckDuckGoSearchTool(BaseTool):
    """
    DuckDuckGo search tool that returns LLaMA-compatible responses.
    """
    name: str = "duckduckgo_search"
    description: str = "Search the web using DuckDuckGo for current information"
    args_schema: Type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:
        """Execute the search and return formatted results."""
        try:
            with DDGS() as ddgs:
                # Try news search first
                news_results = list(ddgs.news(query, max_results=5))
                
                if news_results:
                    formatted_results = []
                    formatted_results.append(f"Search Results for: {query}")
                    formatted_results.append(f"Search performed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    formatted_results.append("\\n=== NEWS RESULTS ===")
                    
                    for i, result in enumerate(news_results, 1):
                        title = result.get('title', 'No title')
                        body = result.get('body', '')
                        url = result.get('url', '')
                        date = result.get('date', 'Recent')
                        
                        formatted_results.append(f"\\n{i}. {title}")
                        if body:
                            # Truncate body to keep it manageable
                            body_snippet = body[:300] + "..." if len(body) > 300 else body
                            formatted_results.append(f"   Summary: {body_snippet}")
                        formatted_results.append(f"   Date: {date}")
                        formatted_results.append(f"   Source: {url}")
                    
                    # Add a continuation prompt to help with conversation flow
                    formatted_results.append("\\n\\nBased on this search information, please analyze the findings.")
                    return "\\n".join(formatted_results)
                
                # If no news results, try regular web search
                web_results = list(ddgs.text(query, max_results=5))
                
                if web_results:
                    formatted_results = []
                    formatted_results.append(f"Search Results for: {query}")
                    formatted_results.append(f"Search performed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    formatted_results.append("\\n=== WEB RESULTS ===")
                    
                    for i, result in enumerate(web_results, 1):
                        title = result.get('title', 'No title')
                        body = result.get('body', '')
                        url = result.get('href', '')
                        
                        formatted_results.append(f"\\n{i}. {title}")
                        if body:
                            # Truncate body to keep it manageable
                            body_snippet = body[:300] + "..." if len(body) > 300 else body
                            formatted_results.append(f"   Summary: {body_snippet}")
                        formatted_results.append(f"   Source: {url}")
                    
                    # Add a continuation prompt to help with conversation flow
                    formatted_results.append("\\n\\nBased on this search information, please analyze the findings.")
                    return "\\n".join(formatted_results)
                
                else:
                    return f"No search results found for: {query}. Please try a different search query."
            
        except Exception as e:
            return f"Search error for '{query}': {str(e)}. Please try again or use a different search term."

def setup_datarobot_connection():
    """Set up DataRobot client and get available models."""
    # Initialize DataRobot client
    dr_client = dr.Client(
        endpoint=os.getenv('DATAROBOT_ENDPOINT'), 
        token=os.getenv('DATAROBOT_API_TOKEN')
    )
    
    # Get available models
    response = dr_client.get(url="genai/llmgw/catalog/")
    data = response.json()["data"]
    supported_llms = [llm_model["model"] for llm_model in data]
    
    print("Available LLMs in DataRobot Gateway:")
    pprint(supported_llms)
    
    return dr_client, supported_llms

def test_model_compatibility(dr_client, model):
    """Test if a model works with basic OpenAI client."""
    try:
        client = openai.OpenAI(
            api_key=dr_client.token,
            base_url=dr_client.endpoint + "/genai/llmgw"
        )
        
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user", 
                "content": f"Hello, please introduce yourself. Model: {model}"
            }],
            max_tokens=100
        )
        
        print(f"✅ Model {model} is working!")
        print(f"Response: {response.choices[0].message.content[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ Model {model} failed: {str(e)}")
        return False

def create_llama_compatible_crew(dr_client, model):
    """Create a crew with LLaMA-compatible configuration."""
    
    # Create LLaMA-compatible LLM
    llm = LlamaCompatibleLLM(
        model=model,
        api_key=dr_client.token,
        base_url=dr_client.endpoint + "/genai/llmgw",
        custom_llm_provider="openai"
    )
    
    # Create LLaMA-compatible search tool
    search_tool = LlamaCompatibleDuckDuckGoSearchTool()
    
    # Create agents with LLaMA-compatible settings
    researcher = Agent(
        role='Senior Research Analyst',
        goal='Uncover cutting-edge developments in AI and data science',
        backstory='''You work at a leading tech think tank.
        Your expertise lies in identifying emerging trends.
        You have a knack for dissecting complex data and presenting actionable insights.''',
        verbose=False,
        allow_delegation=False,
        tools=[search_tool],
        llm=llm
    )
    
    writer = Agent(
        role='Tech Content Strategist',
        goal='Craft compelling content on tech advancements',
        backstory='''You are a renowned Content Strategist, known for your insightful 
        and engaging articles. You transform complex concepts into compelling narratives.''',
        verbose=False,
        allow_delegation=False,  # Set to False to avoid delegation issues with LLaMA
        llm=llm
    )
    
    # Create tasks
    research_task = Task(
        description='''Conduct a comprehensive analysis of the latest advancements in AI for 2025.
        Identify key trends, breakthrough technologies, and potential industry impacts.
        Use the search tool to find recent information about AI developments.
        Your final answer MUST be a full analysis report.''',
        expected_output='A comprehensive 3-paragraph report on the latest AI advancements.',
        agent=researcher
    )
    
    writing_task = Task(
        description='''Using the research analyst's report, compose an engaging blog post.
        The post should be easy to understand, insightful, and positive in tone.
        Make it sound cool, avoid complex words so it doesn't sound like AI.
        Your final answer MUST be the full blog post of at least 4 paragraphs.''',
        expected_output='A 4-paragraph blog post on AI advancements, formatted in markdown.',
        agent=writer
    )
    
    # Create crew
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        process=Process.sequential,
        verbose=False
    )
    
    return crew

def main():
    """Main function to run the LLaMA-compatible crew."""
    print("🚀 Starting LLaMA-Compatible CrewAI Lab...")
    
    # Setup DataRobot connection
    dr_client, supported_llms = setup_datarobot_connection()
    
    # Filter for LLaMA models
    llama_models = [model for model in supported_llms if 'llama' in model.lower()]
    
    if not llama_models:
        print("❌ No LLaMA models found in DataRobot Gateway")
        return
    
    print(f"\\nFound {len(llama_models)} LLaMA models:")
    for model in llama_models:
        print(f"  - {model}")
    
    # Use the first available LLaMA model or specify one
    model = llama_models[0]  # You can change this to test different models
    # model = "bedrock/meta.llama3-70b-instruct-v1:0"  # Uncomment to specify
    
    print(f"\\n🔧 Using model: {model}")
    
    # Test basic compatibility
    if not test_model_compatibility(dr_client, model):
        print("❌ Model compatibility test failed!")
        return
    
    # Create and run crew
    print("\\n🤖 Creating LLaMA-compatible crew...")
    crew = create_llama_compatible_crew(dr_client, model)
    
    print("\\n🚀 Running crew workflow...")
    try:
        result = crew.kickoff()
        
        print("\\n" + "="*50)
        print("📋 CREW RESULTS:")
        print("="*50)
        print(result)
        print("="*50)
        
    except Exception as e:
        print(f"❌ Crew execution failed: {str(e)}")
        print("💡 This might be a conversation format issue. Check the logs above.")
        
        # Try to get partial results
        if hasattr(crew, 'tasks'):
            print("\\n📋 Checking for partial results...")
            for i, task in enumerate(crew.tasks):
                if hasattr(task, 'output') and task.output:
                    print(f"Task {i+1} output: {task.output}")

if __name__ == "__main__":
    main()