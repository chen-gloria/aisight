from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from dotenv import load_dotenv
import asyncio
from typing import List

load_dotenv()

class ImageResult(BaseModel):
    object_count: int
    detected_objects: List[str]

class CountObjectsResult(BaseModel):
    object_count: int

class DetectObjectResult(BaseModel):
    detected_objects: List[str]

agent = Agent(
    "openai:gpt-4o",
    result_type=ImageResult,
    system_prompt=
    "You will be given an image description and you will need to call the appropriate tools to get the image recognition results. "
    "Use 'count_objects' to get the object count and 'detect_object' to get the detected objects.",
)

@agent.tool
async def count_objects(ctx: RunContext) -> CountObjectsResult:
    return "Detect the number of objects based on the image description."

@agent.tool
async def detect_object(ctx: RunContext) -> DetectObjectResult:
    return "Detect the object based on the image description."

async def main(prompt: str):
    nodes = []
    # Begin an AgentRun, which is an async-iterable over the nodes of the agent's graph
    async with agent.iter(prompt) as agent_run:
        async for node in agent_run:
            # Each node represents a step in the agent's execution
            nodes.append(node)
    print(nodes)


if __name__ == "__main__":
    prompt = "There is a land that has four cars in it."
    asyncio.run(main(prompt))

