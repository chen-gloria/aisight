import asyncio
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai import Agent
from dotenv import load_dotenv
import logfire
from dataclasses import dataclass

logfire.configure()

load_dotenv()

server = MCPServerStdio(
    command="uv",
    args=["run", "server.py", "server"],
)

@dataclass
class AgentDeps:
    image_path: str

agent = Agent(
    "openai:gpt-4o",
    mcp_servers=[server],
    deps_type=AgentDeps,
    system_prompt="You will be given an image path and a computer vision task. You will need to call the appropriate server tools to get the results of the computer vision task. "
    "Use 'detect_objects' to get the detected objects."
    "Use 'classify_image' to get the classification results."
    "Use 'start_camera_detection' to start the camera detection."
    "Use 'stop_camera_detection' to stop the camera detection."
    "Use 'get_camera_detections' to get the latest detections from the camera."
    "Use 'test_connection' to test the connection to the YOLO MCP service.",
    instrument=True
)


def run_agent_loop(file_path: str):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(run_agent(file_path))


async def run_agent(file_path: str):
    deps = AgentDeps(file_path)
    async with agent.run_mcp_servers():
        return await agent.run(f"Classify this image: {deps.image_path} using the 'yolov8n-cls.pt' model" , deps=deps)


if __name__ == "__main__":
    image_path = "C:/Users/Gunee/Projects/aisight/public/images/dog.jpg"
    result = run_agent_loop(image_path)
    print(result.data)


