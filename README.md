# What Are AI Agents and How Do They Work?

## For Data Scientists!! 
As a data scientist, you are an expert at extracting insights from data and building predictive models. Now, imagine you could build autonomous assistants that not only build models but also use them, monitor them, and report on them. 

That is the world of AI Agents.

These labs will introduce you to the core concepts of AI agents, using the **CrewAI** framework as our guide. By the end, you'll understand how to build and orchestrate your own AI agents and deploy them into Datarobot. 

## 1. What is an AI Agent?

An AI Agent is more than just a model—it's an autonomous system designed to achieve specific goals. Think of it not as a passive tool (like a calculator) but as an active team member.

A sophisticated AI agent can:
- **Reason and Plan:** It can break down a high-level goal into a series of steps.
- **Use Tools:** It can interact with external software, APIs, and databases (like the DataRobot API).
- **Collaborate:** It can work with other agents, delegating tasks and sharing information to solve complex problems.
- **Remember:** It maintains a memory or context to inform its decisions over time.

 ## 2. How Do They Work? Introducing The CrewAI Framework

CrewAI provides a powerful and intuitive structure for building agentic systems. It breaks down the complexity into four core, easy-to-understand components:

- **Agents:** Individual AI entities with specific roles, goals, and tools
- **Tasks:** Specific objectives assigned to agents or requiring collaboration
- **Crews:** Groups of agents working toward common goals with defined workflows
- **Tools:** External capabilities agents can utilize (APIs, search, file operations)

### Core Concept 1: The Agent (The 'Who')

An Agent is the fundamental actor in the system. It's a specialized worker with a defined persona. In CrewAI, an agent is defined by three key attributes:

- **`role`**: What is the agent's job title? (e.g., 'Senior Financial Analyst', 'MLOps Specialist'). This sets the context for its behavior.
- **`goal`**: What is its primary objective? (e.g., 'Analyze market trends for tech stocks', 'Monitor production models for data drift'). This is the agent's driving motivation.
- **`backstory`**: A brief narrative that gives the agent its personality and expertise. This helps the underlying LLM adopt the correct persona for higher-quality reasoning.


### Core Concept 2: The Tool (The 'How')

An Agent on its own can only think and write. To **act**, it needs Tools. A tool is simply a function that an agent can decide to use to get information or perform an action.

- A tool can be a connection to a database, a web search function, or a custom API call.
- For our purposes, a tool could be a function named `get_datarobot_leaderboard()` or `deploy_model_to_production()`.
- By giving an agent a set of tools, you are defining its capabilities and empowering it to interact with the outside world.

### Core Concept 3: The Task (The 'What')

A Task is the specific assignment you give to an agent. It's a detailed instruction that guides the agent's work. A task is defined by:

- **`description`**: A clear, detailed explanation of what needs to be done and the steps involved.
- **`expected_output`**: A precise description of what the final result should look like (e.g., 'A JSON object with the top 5 models and their validation scores', 'A summary report in markdown format'). This helps the agent understand what a successful outcome is.
- **`agent`**: Which agent is assigned to this task.
### Core Concept 4: The Crew (The 'Orchestra')

A Crew is the collaborative team of agents and the set of tasks they need to accomplish. The Crew is responsible for:

- **Assembling the team:** You define which agents are part of the crew.
- **Assigning the work:** You provide the list of tasks to be completed.
- **Orchestrating the process:** The Crew manages the workflow, allowing agents to work sequentially or delegate tasks to one another, and ultimately produces the final result.