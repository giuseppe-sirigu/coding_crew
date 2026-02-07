Complete Setup: Continue.dev + MCP Server

Phase 1: Install the Local LLM Runtime
Option A: vLLM (Best Performance)
bash# Install vLLM (requires Python 3.8+)
pip install vllm

# Or use conda if you prefer
conda create -n vllm python=3.10
conda activate vllm
pip install vllm
Option B: Ollama (Easiest Setup)
bash# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Verify installation
ollama --version

Phase 2: Download and Run Qwen3 Coder
If using vLLM:
bash# Install huggingface-cli if needed
pip install huggingface-hub

# Download Qwen2.5-Coder model (latest coding model from Qwen family)
# For RTX 5080 (16GB VRAM), use the 14B or 7B parameter model
huggingface-cli download Qwen/Qwen2.5-Coder-14B-Instruct-AWQ

Step 1: Install Continue in VS Code
bash# Install Continue extension
code --install-extension continue.continue

# Or install via VS Code:
# 1. Open Extensions (Ctrl+Shift+X)
# 2. Search for "Continue"
# 3. Click Install

Step 2: Install Required Dependencies
bash# Create a virtual environment for the MCP server
python3 -m venv ~/coding-swarm-env
source ~/coding-swarm-env/bin/activate

# Install dependencies
pip install mcp litellm crewai crewai-tools langchain-community langchain-openai

Step 3: Create the MCP Server
Create a directory for your swarm:
bashmkdir -p ~/coding-swarm
cd ~/coding-swarm
Create swarm_mcp_server.py:
python#!/usr/bin/env python3
"""
MCP Server for Local Coding Agent Swarm
Provides coding tasks to a crew of specialized AI agents
"""
import asyncio
import json
import os
from typing import Any
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

# Configure to use local vLLM endpoint
os.environ["OPENAI_API_BASE"] = "http://localhost:8000/v1"
os.environ["OPENAI_API_KEY"] = "dummy-key-not-needed"
os.environ["OPENAI_MODEL_NAME"] = "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ"

# Initialize the local LLM
local_llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key",
    model="Qwen/Qwen2.5-Coder-14B-Instruct-AWQ",
    temperature=0.1
)

# Define the agent swarm
class CodingSwarm:
    def __init__(self):
        self.architect = Agent(
            role='Software Architect',
            goal='Design clean, scalable, and maintainable code architecture',
            backstory="""You are an expert software architect with 15 years of experience.
            You excel at breaking down complex problems into elegant solutions.
            You always consider scalability, maintainability, and best practices.""",
            verbose=True,
            allow_delegation=True,
            llm=local_llm
        )
        
        self.coder = Agent(
            role='Senior Developer',
            goal='Write high-quality, efficient, and well-documented code',
            backstory="""You are a senior developer who writes clean, tested code.
            You follow best practices, write comprehensive docstrings, and consider
            edge cases. You're proficient in multiple programming languages.""",
            verbose=True,
            allow_delegation=False,
            llm=local_llm
        )
        
        self.reviewer = Agent(
            role='Code Reviewer',
            goal='Review code for bugs, security issues, performance problems, and style',
            backstory="""You are a meticulous code reviewer with a security-first mindset.
            You catch subtle bugs, identify security vulnerabilities, and ensure
            code follows best practices and team standards.""",
            verbose=True,
            allow_delegation=False,
            llm=local_llm
        )
        
        self.tester = Agent(
            role='QA Engineer',
            goal='Write comprehensive tests and ensure code quality',
            backstory="""You are a QA engineer who believes in test-driven development.
            You write unit tests, integration tests, and consider edge cases.
            You ensure code is robust and reliable.""",
            verbose=True,
            allow_delegation=False,
            llm=local_llm
        )
    
    def execute_task(self, description: str, context: str = "") -> str:
        """Execute a coding task using the full swarm"""
        full_description = description
        if context:
            full_description += f"\n\nAdditional Context:\n{context}"
        
        # Create tasks for the crew
        design_task = Task(
            description=f"Analyze and design a solution for: {full_description}",
            agent=self.architect,
            expected_output="Architecture design and implementation plan"
        )
        
        implement_task = Task(
            description=f"Implement the solution based on the architecture: {full_description}",
            agent=self.coder,
            expected_output="Complete, working code with documentation"
        )
        
        review_task = Task(
            description="Review the implemented code for issues, bugs, and improvements",
            agent=self.reviewer,
            expected_output="Code review findings and suggested improvements"
        )
        
        test_task = Task(
            description="Write comprehensive tests for the implemented code",
            agent=self.tester,
            expected_output="Complete test suite"
        )
        
        # Create and run the crew
        crew = Crew(
            agents=[self.architect, self.coder, self.reviewer, self.tester],
            tasks=[design_task, implement_task, review_task, test_task],
            process=Process.sequential,
            verbose=2
        )
        
        result = crew.kickoff()
        return str(result)
    
    def quick_code(self, description: str, context: str = "") -> str:
        """Quick coding task - just architect and coder"""
        full_description = description
        if context:
            full_description += f"\n\nContext:\n{context}"
        
        task = Task(
            description=full_description,
            agent=self.coder,
            expected_output="Working code implementation"
        )
        
        crew = Crew(
            agents=[self.coder],
            tasks=[task],
            verbose=1
        )
        
        result = crew.kickoff()
        return str(result)
    
    def review_code(self, code: str, language: str = "python") -> str:
        """Review existing code"""
        task = Task(
            description=f"Review this {language} code for bugs, security issues, and improvements:\n\n{code}",
            agent=self.reviewer,
            expected_output="Detailed code review with specific suggestions"
        )
        
        crew = Crew(
            agents=[self.reviewer],
            tasks=[task],
            verbose=1
        )
        
        result = crew.kickoff()
        return str(result)
    
    def write_tests(self, code: str, language: str = "python") -> str:
        """Generate tests for existing code"""
        task = Task(
            description=f"Write comprehensive tests for this {language} code:\n\n{code}",
            agent=self.tester,
            expected_output="Complete test suite with multiple test cases"
        )
        
        crew = Crew(
            agents=[self.tester],
            tasks=[task],
            verbose=1
        )
        
        result = crew.kickoff()
        return str(result)

# Initialize the swarm
swarm = CodingSwarm()

# Create MCP server
server = Server("local-coding-swarm")

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools for the MCP client"""
    return [
        Tool(
            name="full_coding_task",
            description="""Execute a complete coding task using all agents (architect, coder, reviewer, tester).
            Use this for new features, complex implementations, or when you want thorough analysis and testing.
            The swarm will design, implement, review, and test the solution.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_description": {
                        "type": "string",
                        "description": "Detailed description of what you want to build"
                    },
                    "context": {
                        "type": "string",
                        "description": "Additional context: existing code, requirements, constraints"
                    }
                },
                "required": ["task_description"]
            }
        ),
        Tool(
            name="quick_code",
            description="""Quick code generation for simple tasks. Uses only the coder agent.
            Best for small functions, quick fixes, or when you already know the design.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_description": {
                        "type": "string",
                        "description": "What code to generate"
                    },
                    "context": {
                        "type": "string",
                        "description": "Any relevant context or requirements"
                    }
                },
                "required": ["task_description"]
            }
        ),
        Tool(
            name="review_code",
            description="""Have the code reviewer analyze existing code for bugs, security issues,
            performance problems, and style improvements.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The code to review"
                    },
                    "language": {
                        "type": "string",
                        "description": "Programming language (e.g., python, javascript, rust)"
                    }
                },
                "required": ["code"]
            }
        ),
        Tool(
            name="write_tests",
            description="""Generate comprehensive tests for existing code.
            Creates unit tests, integration tests, and edge case tests.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The code to write tests for"
                    },
                    "language": {
                        "type": "string",
                        "description": "Programming language"
                    }
                },
                "required": ["code"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls from the MCP client"""
    try:
        if name == "full_coding_task":
            result = swarm.execute_task(
                arguments["task_description"],
                arguments.get("context", "")
            )
        elif name == "quick_code":
            result = swarm.quick_code(
                arguments["task_description"],
                arguments.get("context", "")
            )
        elif name == "review_code":
            result = swarm.review_code(
                arguments["code"],
                arguments.get("language", "python")
            )
        elif name == "write_tests":
            result = swarm.write_tests(
                arguments["code"],
                arguments.get("language", "python")
            )
        else:
            result = f"Unknown tool: {name}"
        
        return [TextContent(
            type="text",
            text=result
        )]
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Error executing {name}: {str(e)}"
        )]

async def main():
    """Run the MCP server"""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
Make it executable:
bashchmod +x swarm_mcp_server.py

Step 4: Configure Continue to Use Your Swarm
Open Continue's config file:
bash# In VS Code, press Ctrl+Shift+P (or Cmd+Shift+P on Mac)
# Type: "Continue: Open config.json"
# Or manually edit: ~/.continue/config.json
Replace the contents with this configuration:
json{
  "models": [
    {
      "title": "Qwen Local Coder",
      "provider": "openai",
      "model": "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ",
      "apiBase": "http://localhost:8000/v1",
      "apiKey": "dummy-key"
    }
  ],
  "tabAutocompleteModel": {
    "title": "Qwen Autocomplete",
    "provider": "openai",
    "model": "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ",
    "apiBase": "http://localhost:8000/v1",
    "apiKey": "dummy-key"
  },
  "embeddingsProvider": {
    "provider": "openai",
    "model": "text-embedding-ada-002",
    "apiBase": "http://localhost:8000/v1",
    "apiKey": "dummy-key"
  },
  "mcpServers": {
    "coding-swarm": {
      "command": "/home/YOUR_USERNAME/coding-swarm-env/bin/python3",
      "args": ["/home/YOUR_USERNAME/coding-swarm/swarm_mcp_server.py"],
      "env": {
        "OPENAI_API_BASE": "http://localhost:8000/v1",
        "OPENAI_API_KEY": "dummy-key"
      }
    }
  },
  "experimental": {
    "modelContextProtocolServers": ["coding-swarm"]
  }
}
Important: Replace YOUR_USERNAME with your actual username.

Step 5: Create a Startup Script
Create start_swarm.sh in your home directory:
bash#!/bin/bash

# Start vLLM server in the background
echo "Starting vLLM server..."
source ~/coding_crew/bin/activate
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-Coder-14B-Instruct-AWQ \
  --quantization awq \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --gpu-memory-utilization 0.85 \
  --max-model-len 12000 &

VLLM_PID=$!
echo "vLLM started with PID: $VLLM_PID"

# Wait for vLLM to be ready
echo "Waiting for vLLM to be ready..."
sleep 30

echo "Swarm is ready! Open VS Code and use Continue."
echo "To stop: kill $VLLM_PID"

# Keep script running
wait
Make it executable:
bashchmod +x ~/start_swarm.sh