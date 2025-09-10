# Copyright 2025 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from typing import Any, Optional, Union

from dotenv import load_dotenv
from crewai import LLM, Agent, Crew, CrewOutput, Task, Process
from helpers import CrewAIEventListener, create_inputs_from_completion_params
from openai.types.chat import CompletionCreateParams
from ragas.messages import AIMessage
from search_tool import DuckDuckGoSearchTool

# Load environment variables from .env file in project root
import pathlib
env_path = pathlib.Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class MyAgent:
    """Simplified CrewAI agent implementing a two-agent research workflow.
    
    This agent creates a streamlined research crew consisting of:
    1. Senior Research Analyst - Uses web search to gather current information
    2. Tech Content Strategist - Transforms research into engaging content
    
    The workflow is based on the CrewAI lab notebook and teaches deployment
    of CrewAI workflows in DataRobot.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
        verbose: Optional[Union[bool, str]] = False,
        **kwargs: Any,
    ):
        """Initialize the simplified MyAgent with DataRobot LLM Gateway configuration.

        Args:
            api_key: DataRobot API token (defaults to DATAROBOT_API_TOKEN env var)
            api_base: DataRobot API endpoint (defaults to DATAROBOT_ENDPOINT env var)
            model: LLM model to use (defaults to vertex_ai/gemini-2.0-flash-001)
            verbose: Enable verbose logging (defaults to False for DataRobot compatibility)
            **kwargs: Additional parameters from CompletionCreateParams
        """
        self.api_key = api_key or os.environ.get("DATAROBOT_API_TOKEN")
        self.api_base = api_base or os.environ.get("DATAROBOT_ENDPOINT")
        self.model = model or "bedrock/meta.llama3-70b-instruct-v1:0"
        
        if isinstance(verbose, str):
            self.verbose = verbose.lower() == "true"
        elif isinstance(verbose, bool):
            self.verbose = verbose
        
        self.event_listener = CrewAIEventListener()
        self.search_tool = DuckDuckGoSearchTool()

    @property
    def api_base_litellm(self) -> str:
        """Get API base URL formatted for LiteLLM compatibility."""
        if self.api_base:
            # Extract the base domain and construct the correct LLM Gateway path
            base = re.sub(r"/api/v2/?$", "", self.api_base)
            return f"{base}/api/v2/genai/llmgw"
        return "https://app.datarobot.com/api/v2/genai/llmgw"

    # @property
    # def llm(self) -> LLM:
    #     """Configure LLM for DataRobot LLM Gateway."""
    #     return LLM(
    #         model=self.model,
    #         api_key=self.api_key,
    #        # base_url=f"{self.api_base_litellm}genai/llmgw",
    #         base_url=self.api_base_litellm,
    #         custom_llm_provider="openai",
    #         num_retries=3,
    #         timeout=60
    #     )

    @property
    def llm(self) -> LLM:
        """Returns a CrewAI LLM instance configured to use DataRobot's LLM Gateway.

        This property can serve as a primary LLM backend for the agents. You can optionally
        have multiple LLMs configured, such as one for DataRobot's LLM Gateway
        and another for a specific DataRobot deployment, or even multiple deployments or
        third-party LLMs.
        """
        return LLM(
            #model='datarobot/azure/gpt-4o-2024-11-20',
            #model='vertex_ai/claude-3-7-sonnet@20250219',
            model='vertex_ai/gemini-2.5-flash',
            api_base=self.api_base_litellm,
            api_key=self.api_key,
            temperature=0.1,
            max_tokens=2000,
            timeout=120,

            # Try these parameters for better compatibility
            custom_llm_provider="openai",
            # Rate limiting configurations
            #rpm=60,  # Requests per minute
            #tpm=10000,  # Tokens per minute
            # Add system message handling for Llama
            #system_message="You are a helpful AI assistant.",
        )

    @property
    def researcher(self) -> Agent:
        """Senior Research Analyst agent with web search capabilities."""
        return Agent(
            role="Senior Research Analyst",
            goal="Uncover cutting-edge developments and provide comprehensive analysis on {topic}",
            backstory="You work at a leading tech think tank. "
                     "Your expertise lies in identifying emerging trends. "
                     "You have a knack for dissecting complex data and presenting "
                     "actionable insights."
                    "Always end your responses with a clear conclusion or recommendation.",
            #verbose=self.verbose,
            verbose=True,
            allow_delegation=False,
            tools=[self.search_tool],
            llm=self.llm,
        )

    @property
    def writer(self) -> Agent:
        """Tech Content Strategist agent for creating engaging content."""
        return Agent(
            role="Tech Content Strategist",
            goal="Craft compelling content on tech advancements about {topic}",
            backstory="You are a renowned Content Strategist, known for "
                     "your insightful and engaging articles. "
                     "You transform complex concepts into compelling narratives.",
            #verbose=self.verbose,
            verbose=True,
            allow_delegation=True,
            llm=self.llm,
        )

    @property
    def research_task(self) -> Task:
        """Research task for the analyst to gather current information."""
        return Task(
            description=(
                "Conduct a comprehensive analysis of {topic}. "
                "Use the search tool to find recent information and developments. "
                "Identify key trends, breakthrough technologies, and potential impacts. "
                "Your final answer MUST be a full analysis report."
            ),
            expected_output="A comprehensive 3-paragraph report with current information and analysis.",
            agent=self.researcher,
        )

    @property
    def writing_task(self) -> Task:
        """Writing task to create engaging content from research."""
        return Task(
            description=(
                "Using the research analyst's report, compose an engaging blog post about {topic}. "
                "The post should be easy to understand, insightful, and positive in tone. "
                "Make it sound cool, avoid complex words so it doesn't sound like AI. "
                "Your final answer MUST be the full blog post of at least 4 paragraphs."
            ),
            expected_output="A 4-paragraph blog post formatted in markdown.",
            agent=self.writer,
        )

    def create_crew(self) -> Crew:
        """Create and return the two-agent research crew."""
        return Crew(
            agents=[self.researcher, self.writer],
            tasks=[self.research_task, self.writing_task],
            process=Process.sequential,
            #verbose=self.verbose,
            verbose=True,
        )

    def run(
        self, completion_create_params: CompletionCreateParams
    ) -> tuple[list[Any], CrewOutput]:
        """Run the simplified two-agent research workflow.

        [THIS METHOD IS REQUIRED FOR THE AGENT TO WORK WITH DRUM SERVER]

        Args:
            completion_create_params: Parameters containing the research topic

        Returns:
            tuple: (list of events, CrewOutput with the final blog post)
        """
        inputs = create_inputs_from_completion_params(completion_create_params)

        if isinstance(inputs, str):
            inputs = {"topic": inputs}

        print("Running simplified research crew with inputs:", inputs, flush=True)

        crew_output = self.create_crew().kickoff(inputs=inputs)
        response_text = str(crew_output.raw)

        events = self.event_listener.messages
        if len(events) > 0:
            last_message = events[-1].content
            if last_message != response_text:
                events.append(AIMessage(content=response_text))
        else:
            events = None
            
        return crew_output, events