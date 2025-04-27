import os
import asyncio
from typing import List, Dict, Any, Union
from pathlib import Path
from pydantic import BaseModel
from pydantic_ai.mcp import MCPServerStdio
from models import ClassificationResponse, DetectionResponse
import base64
from pydantic_ai import Agent, RunContext
from dotenv import load_dotenv
import logfire
from dataclasses import dataclass

logfire.configure()

load_dotenv()

server = MCPServerStdio(
    command="uv",
    args=["run", "yolo_server.py", "server"],
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

def main():
    '''
    Implementation explained here: https://www.slingacademy.com/article/python-asyncio-runtimeerror-event-loop-closed-fixing-guide/
    :return:
    '''
    loop = asyncio.get_event_loop()
    if loop.is_closed():
        asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_agent())

async def run_agent():
    image_path = "C:/Users/Gunee/Projects/aisight/agent/public/images/dog.jpg"
    deps = AgentDeps(image_path)
    async with agent.run_mcp_servers():
        await agent.run(f'Classify this image: {deps.image_path}', deps=deps)
        # await agent.run('Classify what you detect in the camera', deps=deps)

if __name__ == "__main__":
    main()

