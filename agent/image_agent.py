from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from dotenv import load_dotenv
import asyncio
from typing import List
from pydantic_ai.mcp import MCPServerStdio

load_dotenv()


class ImageResult(BaseModel):
    object_count: int
    detected_objects: List[str]


class CountObjectsResult(BaseModel):
    object_count: int


class DetectObjectResult(BaseModel):
    detected_objects: List[str]


class DateDifferenceResult(BaseModel):
    days_between: int

server = MCPServerStdio()

agent = Agent(
    "openai:gpt-4o",
    result_type=ImageResult,
    mcp_servers=[server],
    system_prompt="You will be given an image description and you will need to call the appropriate tools to get the image recognition results. "
    "Use 'count_objects' to get the object count and 'detect_object' to get the detected objects."
    "Use 'run_mcp_client' to find the number of days between two dates",
)


@agent.tool
async def count_objects(ctx: RunContext) -> CountObjectsResult:
    return "Detect the number of objects based on the image description."


@agent.tool
async def detect_object(ctx: RunContext) -> DetectObjectResult:
    return "Detect the object based on the image description."


@agent.tool
async def run_mcp_client(
    ctx: RunContext, start_date: str, end_date: str
) -> DateDifferenceResult:
    """Calculate the number of days between two dates using the MCP server.

    Args:
        start_date: The start date in YYYY-MM-DD format
        end_date: The end date in YYYY-MM-DD format
    """
    async with agent.run_mcp_servers():
        result = await agent.run(f"How many days between {start_date} and {end_date}?")
    return DateDifferenceResult(days_between=int(result.output))


async def main(prompt: str):
    nodes = []
    # Begin an AgentRun, which is an async-iterable over the nodes of the agent's graph
    async with agent.iter(prompt) as agent_run:
        async for node in agent_run:
            # Each node represents a step in the agent's execution
            nodes.append(node)
    print(nodes)


if __name__ == "__main__":
    prompt = "How many days between 2000-01-01 and 2025-03-18?"
    asyncio.run(main(prompt))
