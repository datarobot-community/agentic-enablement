#!/usr/bin/env python3
"""
Simple Single Agent Workflow with External Tool
A basic example of using crewAI with a hypothetical external class as a tool.
"""

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Dict, Any
import datarobot as dr

# Load environment variables
load_dotenv()

# Hypothetical external class that could be a database, API, or any service
class DataProcessor:
    """
    Hypothetical external class that processes data.
    This could represent any external service, database, or API.
    """
    
    def __init__(self):
        self.data_store = {
            "users": ["alice", "bob", "charlie"],
            "products": ["laptop", "mouse", "keyboard"],
            "orders": [
                {"user": "alice", "product": "laptop", "quantity": 1},
                {"user": "bob", "product": "mouse", "quantity": 2},
                {"user": "charlie", "product": "keyboard", "quantity": 1}
            ]
        }
    
    def get_data(self, data_type: str) -> Dict[str, Any]:
        """Retrieve data by type."""
        if data_type in self.data_store:
            return {"success": True, "data": self.data_store[data_type]}
        return {"success": False, "error": f"Data type '{data_type}' not found"}
    
    def analyze_data(self, data_type: str) -> Dict[str, Any]:
        """Perform analysis on data."""
        if data_type == "orders":
            orders = self.data_store.get("orders", [])
            total_orders = len(orders)
            total_quantity = sum(order["quantity"] for order in orders)
            return {
                "success": True,
                "analysis": {
                    "total_orders": total_orders,
                    "total_quantity": total_quantity,
                    "average_quantity": total_quantity / total_orders if total_orders > 0 else 0
                }
            }
        return {"success": False, "error": f"Analysis not available for '{data_type}'"}

# Create tool input schema
class DataProcessorInput(BaseModel):
    """Input schema for the data processor tool."""
    action: str = Field(..., description="Action to perform: 'get' or 'analyze'")
    data_type: str = Field(..., description="Type of data to work with: 'users', 'products', or 'orders'")

# Create crewAI tool that wraps the external class
class DataProcessorTool(BaseTool):
    name: str = "data_processor"
    description: str = "Access and analyze data from the external data processor service"
    args_schema: Type[BaseModel] = DataProcessorInput
    
    def __init__(self):
        super().__init__()
        self.processor = DataProcessor()
    
    def _run(self, action: str, data_type: str) -> str:
        """Execute the data processor action."""
        try:
            if action == "get":
                result = self.processor.get_data(data_type)
            elif action == "analyze":
                result = self.processor.analyze_data(data_type)
            else:
                return f"Unknown action: {action}. Use 'get' or 'analyze'"
            
            if result["success"]:
                if "data" in result:
                    return f"Data retrieved successfully: {result['data']}"
                elif "analysis" in result:
                    analysis = result["analysis"]
                    return f"Analysis completed: {analysis}"
            else:
                return f"Error: {result['error']}"
                
        except Exception as e:
            return f"Tool error: {str(e)}"

def setup_llm():
    """Setup LLM connection to DataRobot Gateway."""
    # Initialize DataRobot client
    dr_client = dr.Client(
        endpoint=os.getenv('DATAROBOT_ENDPOINT'), 
        token=os.getenv('DATAROBOT_API_TOKEN')
    )
    
    # Get available models (you could also hardcode a preferred model)
    response = dr_client.get(url="genai/llmgw/catalog/")
    data = response.json()["data"]
    supported_llms = [llm_model["model"] for llm_model in data]
    
    # Use a reliable model (or you could randomize like in the notebook)
    model = "anthropic/claude-3-5-sonnet-20241022"  # Fallback to first if preferred not available
    if model not in supported_llms and supported_llms:
        model = supported_llms[0]
    
    # Create LLM configuration
    llm = LLM(
        model=model,
        api_key=dr_client.token,
        base_url=dr_client.endpoint + "/genai/llmgw",
        custom_llm_provider="openai"
    )
    
    return llm

def main():
    """Main function to run the single agent workflow."""
    print("🚀 Starting Simple Single Agent Workflow...")
    
    # Setup LLM
    llm = setup_llm()
    
    # Create the data processor tool
    data_tool = DataProcessorTool()
    
    # Create the agent
    data_analyst = Agent(
        role='Data Analyst',
        goal='Analyze business data and provide insights',
        backstory='''You are a skilled data analyst who specializes in 
        extracting meaningful insights from business data. You use various 
        tools to access and analyze data, then provide clear, actionable 
        recommendations.''',
        verbose=True,
        allow_delegation=False,
        tools=[data_tool],
        llm=llm
    )
    
    # Create the task
    analysis_task = Task(
        description='''Use the data processor tool to:
        1. First get the orders data
        2. Then analyze the orders data
        3. Provide a summary of the findings and any business insights
        Your final answer should include the raw data and analysis results.''',
        expected_output='A comprehensive report with data findings and business insights.',
        agent=data_analyst
    )
    
    # Create the crew
    crew = Crew(
        agents=[data_analyst],
        tasks=[analysis_task],
        process=Process.sequential,
        verbose=True
    )
    
    # Run the workflow
    print("\n📊 Running data analysis workflow...")
    result = crew.kickoff()
    
    print("\n" + "="*50)
    print("📋 WORKFLOW RESULTS:")
    print("="*50)
    print(result)
    print("="*50)
    
    print("\n✅ Workflow completed successfully!")

if __name__ == "__main__":
    main()