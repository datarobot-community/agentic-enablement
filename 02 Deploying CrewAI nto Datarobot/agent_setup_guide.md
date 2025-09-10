# DataRobot Agentic Lab Deployment Guide

> **Note:** This setup guide is a companion to, and shows you how to deploy a CrewAI agentic framework in DataRobot, utilizing templates in this git repo: https://github.com/datarobot-community/datarobot-agent-templates.git

## Understanding the Three Virtual Environments

Before we start, it's important to understand that there are three virtual environments (.venv) that are utilized.
```
project-root/
├── .venv/                          # 1. Project Root Environment
│                                   #    - Used for running quickstart.py
│
├── agent_crewai/
│   └── .venv/                      # 2. Local Agent Environment (agent_crewai.venv)
│                                   #    - Used for local testing and CLI
│                                   #    - Runs when you execute: task agent:cli
│
└── infra/
    └── .venv/                      # 3. Infrastructure Environment
                                    #    - Used for model registration and deployment
                                    #    - Handles requirements.in → requirements.txt compilation
```

### 1. Project Root Environment
The first environment is in the project root folder. This is used for running the quickstart.

### 2. Local Agent Environment
Second is the local agent folder. In this case, it is `agent_crewai.venv`
- This environment is for running and testing locally
- When you run `task agent:cli` from the project root folder, task runs the command line interface in this `.venv`

### 3. Infrastructure Environment
Third is the infra folder. This environment is used for registering and deploying the model, and thus the workflow.
- Note the step below, where we modify the `requirements.in` file, and then run a script to update the `requirements.txt` and other resources

## Important Notes About Package Management

It's important to know that there are three separate `.venv` environments. When we need to add packages to these environments, we navigate to the parent folder of that `.venv` and use the uv utilities to add the appropriate package.

For example, we will use the `uv add` command to add the ddg package to our local model. When we add to the registered and deployed models, we use the `uv pip compile` command - but more on that later.

**Important:** You cannot pip install packages, as you will get an error message indicating that it is a managed environment.

## Initial Setup

Ok, now with all that... you're ready to get this going, right? Not so fast... you probably have a shiny new `.venv` that was created by your codespace, pycharm, vs code, or whatever. You don't need that. Let's deactivate that:

From a command prompt, type `deactivate` and hit enter.

Now you're ready.

## Creating the Environment File

Let's create the `.env` file.

From terminal:
```bash
cp .env.sample .env
```

Now, in your favorite text editor open that shiny new `.env` file:
- Add your pulumi password, API credentials (if necessary), and set a use case ID, if you desire
- Remark or remove the last line, that mentions the base environment

## Running the Quickstart

Now, drop to a terminal again:

From project root:
```bash
python quickstart.py
```

Choose Agent framework. In this case, it will be crew ai.

The quickstart will remove the directories for the agentic frameworks that you are not using.

## Required Files

Copy the following files to your `\agent_crewai\custom_model\` directory:
- `agent.py`
- `search_tool.py`

- These files can be obtained from the following sources:

**Google Drive:**
https://drive.google.com/drive/folders/1Q605vPeLomkuITgBWXRyhVQ2UGro2ZMk

**Git Repository:**
https://github.com/datarobot/agentic-enablement.git
(Available in the "02 Deploying crew_ai into Datarobot" folder)

## Adding Required Packages

Once you have the files copied, you need to add the required package(s) to the local model. In this case, we will be adding "ddgs" to the agent_crewai virtual environment.

### For Local Environment

To do so, navigate to the `/agent_crewai/` directory in your terminal, then run this command:
```bash
uv add ddgs
```

### For Registered and Deployed Models

Next, we need to add the package to the registered and deployed models:

1. Open the `agent_crewai/docker_context/requirements.in` file
2. Once this is done, you will need to run this command from the same folder:
```bash
uv pip compile --no-annotate --no-emit-index-url --output-file=requirements.txt requirements.in
```

## Testing the Model Locally

Now that the package requirements are set, let's test the model locally. This step will execute your code locally, using the `.venv` that is located in the `\agent_crewai\.venv` directory.

In your terminal, from the project root directory, run this command:
```bash
task cli -- execute --user_prompt 'developments in AI over the past 3 months'
```

Your agent will run locally, which you can watch and view the output. After completing this, let's build and deploy the model.

## Building the Model

From your project root folder, run the following command:
```bash
task build
```

You will be asked to choose a name for your stack. Name it whatever you want, and hit enter.

If asked whether you want to make the changes it detects, choose yes.

**(It's a good idea to copy the green text output to a notes file, for future reference)**

## Deploying the Model

Once this is complete, run this command from the same directory:
```bash
task deploy
```

Once this is completed, copy the new green output text to your notes file, from above.

## Using the Workflow

Now you're ready to use the workflow in an agentic playground. Open the DR GUI and create a new agentic playground in your use case. It will be available as a workflow.