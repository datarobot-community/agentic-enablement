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

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from ddgs import DDGS
from datetime import datetime


class SearchInput(BaseModel):
    """Input schema for DuckDuckGo search tool."""
    query: str = Field(..., description="The search query")


class DuckDuckGoSearchTool(BaseTool):
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
                    formatted_results.append("\n=== NEWS RESULTS ===")

                    for i, result in enumerate(news_results, 1):
                        title = result.get('title', 'No title')
                        body = result.get('body', '')
                        url = result.get('url', '')
                        date = result.get('date', 'Recent')

                        formatted_results.append(f"\n{i}. {title}")
                        if body:
                            # Truncate body to keep it manageable
                            body_snippet = body[:300] + "..." if len(body) > 300 else body
                            formatted_results.append(f"   Summary: {body_snippet}")
                        formatted_results.append(f"   Date: {date}")
                        formatted_results.append(f"   Source: {url}")

                    return "\n".join(formatted_results)

                # If no news results, try regular web search
                web_results = list(ddgs.text(query, max_results=5))

                if web_results:
                    formatted_results = []
                    formatted_results.append(f"Search Results for: {query}")
                    formatted_results.append(f"Search performed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    formatted_results.append("\n=== WEB RESULTS ===")

                    for i, result in enumerate(web_results, 1):
                        title = result.get('title', 'No title')
                        body = result.get('body', '')
                        url = result.get('href', '')

                        formatted_results.append(f"\n{i}. {title}")
                        if body:
                            # Truncate body to keep it manageable
                            body_snippet = body[:300] + "..." if len(body) > 300 else body
                            formatted_results.append(f"   Summary: {body_snippet}")
                        formatted_results.append(f"   Source: {url}")

                    return "\n".join(formatted_results)

                else:
                    return f"No search results found for: {query}"

        except Exception as e:
            return f"Search error for '{query}': {str(e)}"