"""
MCP Server for Local Coding Agent Swarm
"""
import asyncio
import os
import sys
import logging

# Set up logging FIRST
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/swarm_mcp_server.log'),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

logger.info("="*50)
logger.info("MCP Server script starting...")
logger.info("="*50)

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from crewai import Agent, Task, Crew, Process, LLM

# Read configuration from environment variables (set by Continue config.yaml or start_swarm.sh)
api_base = os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1")
api_key = os.environ.get("OPENAI_API_KEY", "dummy-key")
model_name = os.environ.get("OPENAI_MODEL_NAME", "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ")

# Ensure env vars are set for CrewAI internals
os.environ["OPENAI_API_BASE"] = api_base
os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_MODEL_NAME"] = model_name

logger.info("Initializing local LLM (model=%s, base_url=%s)...", model_name, api_base)

# Initialize the local LLM using CrewAI's LLM class
llm_model = f"openai/{model_name}" if not model_name.startswith("openai/") else model_name
local_llm = LLM(
    model=llm_model,
    base_url=api_base,
    api_key=api_key,
    temperature=0.1
)

logger.info("LLM initialized successfully")

# Define the agent swarm
class CodingSwarm:
    def __init__(self):
        logger.info("Initializing CodingSwarm agents...")
        
        self.architect = Agent(
            role='Software Architect',
            goal='Design clean, scalable code architecture',
            backstory='Expert software architect with 15 years of experience',
            verbose=False,
            allow_delegation=True,
            llm=local_llm
        )
        
        self.coder = Agent(
            role='Senior Developer',
            goal='Write high-quality, efficient code',
            backstory='Senior developer who writes clean, tested code',
            verbose=False,
            allow_delegation=False,
            llm=local_llm
        )
        
        self.reviewer = Agent(
            role='Code Reviewer',
            goal='Review code for bugs and best practices',
            backstory='Meticulous code reviewer with security focus',
            verbose=False,
            allow_delegation=False,
            llm=local_llm
        )
        
        self.tester = Agent(
            role='QA Engineer',
            goal='Write comprehensive tests',
            backstory='QA engineer who believes in test-driven development',
            verbose=False,
            allow_delegation=False,
            llm=local_llm
        )
        
        logger.info("All agents initialized successfully")
    
    def execute_task(self, description: str, context: str = "") -> str:
        """Execute a coding task using the full swarm"""
        logger.info(f"Executing full task: {description[:100]}...")
        
        full_description = description
        if context:
            full_description += f"\n\nContext:\n{context}"
        
        design_task = Task(
            description=f"Design solution for: {full_description}",
            agent=self.architect,
            expected_output="Architecture design"
        )
        
        implement_task = Task(
            description=f"Implement: {full_description}",
            agent=self.coder,
            expected_output="Working code"
        )
        
        review_task = Task(
            description="Review the code",
            agent=self.reviewer,
            expected_output="Code review"
        )
        
        test_task = Task(
            description="Write tests",
            agent=self.tester,
            expected_output="Test suite"
        )
        
        crew = Crew(
            agents=[self.architect, self.coder, self.reviewer, self.tester],
            tasks=[design_task, implement_task, review_task, test_task],
            process=Process.sequential,
            verbose=False  # Ensure verbose is a boolean
        )
        
        result = crew.kickoff()
        logger.info("Task completed successfully")
        return str(result)
    
    def quick_code(self, description: str, context: str = "") -> str:
        """Quick coding task"""
        logger.info(f"Quick code: {description[:100]}...")
        
        full_description = description
        if context:
            full_description += f"\n\nContext:\n{context}"
        
        task = Task(
            description=full_description,
            agent=self.coder,
            expected_output="Code"
        )
        
        crew = Crew(
            agents=[self.coder],
            tasks=[task],
            verbose=False
        )
        
        result = crew.kickoff()
        logger.info("Quick code completed")
        return str(result)
    
    def review_code(self, code: str, language: str = "python") -> str:
        """Review code"""
        logger.info(f"Reviewing {language} code...")
        
        task = Task(
            description=f"Review this {language} code:\n\n{code}",
            agent=self.reviewer,
            expected_output="Code review"
        )
        
        crew = Crew(
            agents=[self.reviewer],
            tasks=[task],
            verbose=False
        )
        
        result = crew.kickoff()
        logger.info("Code review completed")
        return str(result)
    
    def write_tests(self, code: str, language: str = "python") -> str:
        """Write tests"""
        logger.info(f"Writing tests for {language} code...")
        
        task = Task(
            description=f"Write tests for:\n\n{code}",
            agent=self.tester,
            expected_output="Tests"
        )
        
        crew = Crew(
            agents=[self.tester],
            tasks=[task],
            verbose=False
        )
        
        result = crew.kickoff()
        logger.info("Tests completed")
        return str(result)

# Initialize the swarm
logger.info("Creating swarm instance...")
swarm = CodingSwarm()
logger.info("Swarm created successfully")

# Create MCP server
server = Server("local-coding-swarm")
logger.info("MCP Server instance created")

@server.list_tools()
async def list_tools() -> list[Tool]:
    logger.info("list_tools called")
    return [
        Tool(
            name="full_coding_task",
            description="Execute complete coding task with all agents",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_description": {"type": "string"},
                    "context": {"type": "string"}
                },
                "required": ["task_description"]
            }
        ),
        Tool(
            name="quick_code",
            description="Quick code generation",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_description": {"type": "string"},
                    "context": {"type": "string"}
                },
                "required": ["task_description"]
            }
        ),
        Tool(
            name="review_code",
            description="Review code",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "language": {"type": "string"}
                },
                "required": ["code"]
            }
        ),
        Tool(
            name="write_tests",
            description="Write tests",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "language": {"type": "string"}
                },
                "required": ["code"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    logger.info(f"Tool called: {name}")
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
        
        return [TextContent(type="text", text=result)]
    except Exception as e:
        logger.error(f"Tool execution error: {e}", exc_info=True)
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def main():
    logger.info("Starting MCP server main loop")
    try:
        async with stdio_server() as (read_stream, write_stream):
            logger.info("MCP server stdio initialized - ready for connections")
            await server.run(read_stream, write_stream, server.create_initialization_options())
    except Exception as e:
        logger.error(f"MCP server error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        logger.info("Running asyncio.run(main())")
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)