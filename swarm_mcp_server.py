"""
MCP Server for Local Coding Agent Swarm
"""
import asyncio
import os
import re
import sys
import logging
import queue
from typing import Dict, List

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

# --- Optional Phoenix tracing (must run before CrewAI imports) ---
from tracing import initialize_tracing, shutdown_tracing
_tracing_active = initialize_tracing()
if _tracing_active:
    logger.info("Phoenix tracing is ACTIVE - view at http://localhost:6006")
# --- End tracing setup ---

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from crewai import Agent, Task, Crew, Process, LLM

# Regex to extract ```-fenced file blocks preceded by "### File: <path>"
FILE_BLOCK_RE = re.compile(
    r"### File:\s*(?P<path>.+?)\n```[a-zA-Z0-9]*\n(?P<content>.*?)\n```",
    re.DOTALL,
)

EXT_TO_LANG = {
    "py": "python", "js": "javascript", "ts": "typescript",
    "tsx": "typescriptreact", "jsx": "javascriptreact",
    "rb": "ruby", "go": "go", "rs": "rust", "java": "java",
    "cpp": "cpp", "c": "c", "cs": "csharp", "sh": "bash",
    "yaml": "yaml", "yml": "yaml", "json": "json",
    "html": "html", "css": "css", "sql": "sql",
    "md": "markdown", "toml": "toml", "xml": "xml",
}


def _reformat_code_fences(text: str) -> str:
    """Rewrite ``### File: path`` + code-fence blocks so the filepath appears
    in the fence metadata (```lang path), which is what Continue parses to
    drive the 'Create file' vs 'Apply' button."""
    def _replace(m: re.Match) -> str:
        path = m.group("path").strip()
        content = m.group("content").rstrip() + "\n"
        ext = path.rsplit(".", 1)[-1] if "." in path else ""
        lang = EXT_TO_LANG.get(ext, ext)
        return f"```{lang} {path}\n{content}```"

    return FILE_BLOCK_RE.sub(_replace, text)


# Read configuration from environment variables (set by Continue config.yaml or start_swarm.sh)
api_base = os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1")
api_key = os.environ.get("OPENAI_API_KEY", "dummy-key")
model_name = os.environ.get("OPENAI_MODEL_NAME", "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ")
backend = os.environ.get("SWARM_BACKEND", "ollama")

# Ensure env vars are set for CrewAI internals
os.environ["OPENAI_API_BASE"] = api_base
os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_MODEL_NAME"] = model_name

logger.info("Initializing local LLM (model=%s, base_url=%s, backend=%s)...", model_name, api_base, backend)

# Initialize the local LLM using CrewAI's LLM class.
# For Ollama, use the native ollama/ prefix so LiteLLM calls the Ollama API
# directly instead of the OpenAI-compatible endpoint — this avoids empty responses.
if backend == "ollama":
    llm_model = f"ollama/{model_name}" if not model_name.startswith("ollama/") else model_name
    # Strip /v1 suffix — LiteLLM ollama provider talks to the native API
    ollama_base = api_base.removesuffix("/v1").removesuffix("/v1/")
    local_llm = LLM(
        model=llm_model,
        base_url=ollama_base,
        api_key=api_key,
        temperature=0.1,
        max_tokens=4096,
    )
else:
    llm_model = f"openai/{model_name}" if not model_name.startswith("openai/") else model_name
    local_llm = LLM(
        model=llm_model,
        base_url=api_base,
        api_key=api_key,
        temperature=0.1,
        max_tokens=4096,
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
            goal='Write unit tests and integration tests to verify the code works correctly',
            backstory=(
                'Experienced QA engineer specializing in writing automated test suites '
                'for software development teams. Your job is strictly to help developers '
                'by producing test files that validate their code using standard testing '
                'frameworks. You always produce helpful output.'
            ),
            verbose=False,
            allow_delegation=False,
            llm=local_llm
        )
        
        logger.info("All agents initialized successfully")
    
    def execute_task(self, description: str, context: str = "", task_callback=None) -> str:
        """Execute a coding task using the full swarm"""
        logger.info(f"Executing full task: {description[:100]}...")
        
        full_description = description
        if context:
            full_description += f"\n\nContext:\n{context}"
        
        design_task = Task(
            description=f"Design solution for: {full_description}",
            agent=self.architect,
            expected_output=(
                "Architecture design with a clear list of files to create. "
                "For each file specify the full relative path (e.g. src/auth/routes.py)."
            ),
        )

        implement_task = Task(
            description=(
                f"Implement: {full_description}\n\n"
                "IMPORTANT: Output each file separately using this exact format:\n\n"
                "### File: <relative_path>\n"
                "```<language>\n"
                "<code>\n"
                "```\n\n"
                "Do NOT wrap your entire response in an outer markdown code fence. "
                "Each file gets its own fenced code block. "
                "Use clear, real file paths (e.g. app.py, routes/auth.py, models/user.py). "
                "Produce ALL files needed for a working implementation."
            ),
            agent=self.coder,
            expected_output=(
                "Complete working code. Each file preceded by '### File: <path>' "
                "and its own fenced code block. No outer wrapping fence."
            ),
        )

        review_task = Task(
            description=(
                "Review the code from the previous task. "
                "List any issues found and provide corrected files where needed.\n"
                "For corrected files use this exact format:\n\n"
                "### File: <relative_path>\n"
                "```<language>\n"
                "<corrected code>\n"
                "```\n\n"
                "Do NOT wrap your entire response in an outer markdown code fence."
            ),
            agent=self.reviewer,
            expected_output=(
                "Code review findings and any corrected files, each with "
                "'### File: <path>' header and its own fenced code block. "
                "No outer wrapping fence."
            ),
        )

        test_task = Task(
            description=(
                "Write unit tests for the source code provided in the previous task. "
                "Use the appropriate testing framework for the language "
                "(e.g. pytest for Python, Jest/Mocha for JavaScript). "
                "Cover the main functionality including happy paths and error cases.\n\n"
                "Output each test file using this exact format:\n\n"
                "### File: <relative_path>\n"
                "```<language>\n"
                "<test code>\n"
                "```\n\n"
                "Do NOT wrap your entire response in an outer markdown code fence. "
                "Use clear file paths (e.g. tests/test_app.py, __tests__/app.test.js)."
            ),
            agent=self.tester,
            expected_output=(
                "Test files with unit tests, each preceded by '### File: <path>' "
                "and its own fenced code block. No outer wrapping fence."
            ),
        )
        
        crew = Crew(
            agents=[self.architect, self.coder, self.reviewer, self.tester],
            tasks=[design_task, implement_task, review_task, test_task],
            process=Process.sequential,
            verbose=False,
            task_callback=task_callback,
        )
        
        result = crew.kickoff()
        logger.info("Task completed successfully")

        # Combine all task outputs so the caller sees every agent's work,
        # not just the last one.  Strip outer ```markdown fences that the
        # LLM sometimes wraps around the whole response.
        def _strip_outer_fence(text: str) -> str:
            """Remove a single outer ```markdown ... ``` wrapper if present."""
            stripped = text.strip()
            if stripped.startswith("```markdown") or stripped.startswith("```"):
                # Remove opening fence line
                first_nl = stripped.index("\n")
                inner = stripped[first_nl + 1:]
                # Remove closing fence
                if inner.rstrip().endswith("```"):
                    inner = inner.rstrip()[:-3].rstrip()
                return inner
            return text

        def _is_refusal(text: str) -> bool:
            """Detect if the agent refused to produce output."""
            lower = text.strip().lower()
            return lower.startswith("i'm sorry") or lower.startswith("i can't assist")
        
        def extract_files(text: str) -> Dict[str, str]:
            """
            Extract files from CrewAI markdown output.
            Later files with the same path overwrite earlier ones.
            """
            files: Dict[str, str] = {}
            for match in FILE_BLOCK_RE.finditer(text or ""):
                path = match.group("path").strip()
                content = match.group("content").rstrip() + "\n"
                files[path] = content
            return files

        def strip_file_blocks(text: str) -> str:
            """Remove all file blocks, leaving explanation text."""
            return FILE_BLOCK_RE.sub("", text or "").strip()

        sections = []

        all_files: Dict[str, str] = {}
        explanations: List[str] = []

        labels = ["Architecture Design", "Implementation", "Code Review", "Tests"]
        for label, task_output in zip(labels, result.tasks_output):
            raw = task_output.raw
            if _is_refusal(raw):
                logger.warning(f"Agent '{label}' refused, skipping its output")
                continue
            
            files = extract_files(raw)
            if files:
                all_files.update(files)

            explanation = strip_file_blocks(raw)
            if explanation:
                explanations.append(f"## {label}\n\n{explanation}")

            # cleaned = _strip_outer_fence(raw)
            # sections.append(f"## {label}\n\n{cleaned}")

        # preamble = (
        #     "The coding crew has finished. "
        #     "PROVIDE THE CODE GENERATED BY THIS TOOL IN THE OUTPUT TO THE USER TOGETHER WITH AN EXPLANATION OF IT!! "
        #     "Each code block is is marked with a '### File:' header followed by its content. "
        #     "The user should be able to generate files from the output of this tool by using your tools.\n\n"
        # )
        # return preamble + "\n\n---\n\n".join(sections)

        return {
            "files": all_files,
            "explanation": "\n\n---\n\n".join(explanations),
        }
    
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
            description="Use this tool when the user asks for code generation, implementation, or coding tasks. This tool delegates work to a multi-agent swarm (architect, coder, reviewer, tester) for higher quality output.",
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
async def call_tool(name: str, arguments: dict):
    logger.info(f"Tool called: {name}")
    try:
        loop = asyncio.get_running_loop()
        ctx = server.request_context
        session = ctx.session
        progress_token = ctx.meta.progressToken if ctx.meta else None
        logger.info("Progress token from client: %s (meta=%s)", progress_token, ctx.meta)

        # Thread-safe queue: CrewAI task_callback (sync, in executor) puts
        # messages here; the async monitor loop reads and forwards them.
        progress_q: queue.Queue[str] = queue.Queue()
        agent_labels = ["Software Architect", "Senior Developer", "Code Reviewer", "QA Engineer"]
        total_steps = 4 if name == "full_coding_task" else 1
        step_counter = [0]

        def on_task_complete(task_output):
            """CrewAI task_callback — called from the executor thread."""
            agent_role = getattr(task_output, "agent", None) or "Agent"
            progress_q.put(str(agent_role))

        async def monitor():
            """Drain progress queue and send keepalive pings."""
            keepalive_ticks = 0
            while True:
                await asyncio.sleep(1)
                keepalive_ticks += 1

                # Forward completed-task messages as progress notifications
                while not progress_q.empty():
                    agent_name = progress_q.get_nowait()
                    step_counter[0] += 1
                    idx = step_counter[0]
                    label = agent_labels[idx - 1] if idx <= len(agent_labels) else agent_name
                    msg = f"Step {idx}/{total_steps}: {label} finished"
                    logger.info(msg)
                    if progress_token:
                        try:
                            await session.send_progress_notification(
                                progress_token=progress_token,
                                progress=float(idx),
                                total=float(total_steps),
                                message=msg,
                            )
                        except Exception as exc:
                            logger.debug(f"Progress notification failed: {exc}")
                    else:
                        try:
                            await session.send_log_message(
                                level="info", data=msg, logger="coding-swarm",
                            )
                        except Exception:
                            pass

                # Keepalive every 15 seconds to prevent stdio timeout
                if keepalive_ticks >= 15:
                    keepalive_ticks = 0
                    elapsed_msg = f"Crew working... ({step_counter[0]}/{total_steps} steps done)"
                    try:
                        if progress_token:
                            await session.send_progress_notification(
                                progress_token=progress_token,
                                progress=float(step_counter[0]),
                                total=float(total_steps),
                                message=elapsed_msg,
                            )
                        else:
                            await session.send_log_message(
                                level="info", data=elapsed_msg, logger="coding-swarm",
                            )
                    except Exception:
                        break

        monitor_task = asyncio.create_task(monitor())

        try:
            if name == "full_coding_task":
                result = await loop.run_in_executor(
                    None,
                    swarm.execute_task,
                    arguments["task_description"],
                    arguments.get("context", ""),
                    on_task_complete,
                )

                text_parts = []
                # Build text output with filepaths in code fence metadata.
                # Continue parses ```lang filepath\n...\n``` to extract
                # relativeFilepath — this is what drives the "Create file"
                # vs "Apply" button (NOT the ResourceContents URI).
                # text_parts = [
                #     "The coding crew has completed the task. "
                #     "Make sure to provide explanations in addition to the generated code. "
                #     "The user should be able to create the files if they do not already exist.\n",
                # ]

                if result.get("explanation"):
                    text_parts.append(result["explanation"])

                if result.get("files"):
                    text_parts.append("\n## Generated Files\n")
                    for path, code in result["files"].items():
                        ext = path.rsplit(".", 1)[-1] if "." in path else ""
                        lang = EXT_TO_LANG.get(ext, ext)
                        text_parts.append(f"```{lang} {path}\n{code}```\n")

                out = [TextContent(
                    type="text",
                    text="\n".join(text_parts),
                )]
                logger.info("full_coding_task completed with %d files", len(result.get("files", {})))

            elif name == "quick_code":
                result = await loop.run_in_executor(
                    None,
                    swarm.quick_code,
                    arguments["task_description"],
                    arguments.get("context", ""),
                )
                out = [TextContent(type="text", text=_reformat_code_fences(result))]
            elif name == "review_code":
                result = await loop.run_in_executor(
                    None,
                    swarm.review_code,
                    arguments["code"],
                    arguments.get("language", "python"),
                )
                out = [TextContent(type="text", text=_reformat_code_fences(result))]
            elif name == "write_tests":
                result = await loop.run_in_executor(
                    None,
                    swarm.write_tests,
                    arguments["code"],
                    arguments.get("language", "python"),
                )
                out = [TextContent(type="text", text=_reformat_code_fences(result))]
            else:
                out = [TextContent(type="text", text=f"Unknown tool: {name}")]
        finally:
            # Give the monitor one last chance to drain any remaining
            # messages (e.g. the final task_callback) before cancelling.
            await asyncio.sleep(0.1)
            monitor_task.cancel()

        return out
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
    finally:
        shutdown_tracing()

if __name__ == "__main__":
    try:
        logger.info("Running asyncio.run(main())")
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
