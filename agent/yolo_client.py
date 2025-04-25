from typing import List, Dict, Any, Union
import asyncio
import os
import base64
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class YOLOClient:
    def __init__(self, server_command: str = "uv", server_args: List[str] = None):
        """
        Initialize YOLO MCP client

        Args:
            server_command: Command to start the server (default: "uv")
            server_args: Arguments for the server command (default: ["run", "server.py", "server"])
        """
        if server_args is None:
            server_args = ["run", "server.py", "server"]

        self.server_params = StdioServerParameters(
            command=server_command, args=server_args, env=os.environ
        )

    async def _call_tool(
        self, tool_name: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Internal method to call MCP tools"""
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, params)
                return result.content[0].text

    async def detect_objects(
        self,
        image_data: Union[str, bytes, Path],
        model_name: str = "yolov8n.pt",
        confidence: float = 0.25,
        save_results: bool = False,
    ) -> Dict[str, Any]:
        """
        Detect objects in an image

        Args:
            image_data: Image data (file path, base64 string, or bytes)
            model_name: YOLO model name
            confidence: Detection confidence threshold
            save_results: Whether to save results to disk

        Returns:
            Dictionary containing detection results
        """
        # Handle different input types
        if isinstance(image_data, (str, Path)):
            is_path = True
            image_str = str(image_data)
        else:
            is_path = False
            if isinstance(image_data, bytes):
                image_str = base64.b64encode(image_data).decode("utf-8")
            else:
                image_str = image_data

        return await self._call_tool(
            "detect_objects",
            {
                "image_data": image_str,
                "model_name": model_name,
                "confidence": confidence,
                "save_results": save_results,
                "is_path": is_path,
            },
        )

    async def classify_image(
        self,
        image_data: Union[str, bytes, Path],
        model_name: str = "yolov8n-cls.pt",
        top_k: int = 5,
        save_results: bool = False,
    ) -> Dict[str, Any]:
        """
        Classify an image

        Args:
            image_data: Image data (file path, base64 string, or bytes)
            model_name: YOLO classification model name
            top_k: Number of top classifications to return
            save_results: Whether to save results to disk

        Returns:
            Dictionary containing classification results
        """
        # Handle different input types
        if isinstance(image_data, (str, Path)):
            is_path = True
            image_str = str(image_data)
        else:
            is_path = False
            if isinstance(image_data, bytes):
                image_str = base64.b64encode(image_data).decode("utf-8")
            else:
                image_str = image_data

        return await self._call_tool(
            "classify_image",
            {
                "image_data": image_str,
                "model_name": model_name,
                "top_k": top_k,
                "save_results": save_results,
                "is_path": is_path,
            },
        )

    async def train_model(
        self,
        dataset_path: str,
        model_name: str = "yolov8n.pt",
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        name: str = "yolo_custom_model",
        project: str = "runs/train",
    ) -> Dict[str, Any]:
        """
        Train a YOLO model

        Args:
            dataset_path: Path to YOLO format dataset
            model_name: Base model to start with
            epochs: Number of training epochs
            imgsz: Image size for training
            batch: Batch size
            name: Name for the training run
            project: Project directory

        Returns:
            Dictionary containing training results
        """
        return await self._call_tool(
            "train_model",
            {
                "dataset_path": dataset_path,
                "model_name": model_name,
                "epochs": epochs,
                "imgsz": imgsz,
                "batch": batch,
                "name": name,
                "project": project,
            },
        )

    async def validate_model(
        self, model_path: str, data_path: str, imgsz: int = 640, batch: int = 16
    ) -> Dict[str, Any]:
        """
        Validate a YOLO model

        Args:
            model_path: Path to YOLO model (.pt file)
            data_path: Path to YOLO format validation dataset
            imgsz: Image size for validation
            batch: Batch size

        Returns:
            Dictionary containing validation results
        """
        return await self._call_tool(
            "validate_model",
            {
                "model_path": model_path,
                "data_path": data_path,
                "imgsz": imgsz,
                "batch": batch,
            },
        )

    async def start_camera_detection(
        self,
        model_name: str = "yolov8n.pt",
        confidence: float = 0.25,
        camera_id: int = 0,
    ) -> Dict[str, Any]:
        """
        Start realtime camera detection

        Args:
            model_name: YOLO model name
            confidence: Detection confidence threshold
            camera_id: Camera device ID

        Returns:
            Dictionary containing operation status
        """
        return await self._call_tool(
            "start_camera_detection",
            {
                "model_name": model_name,
                "confidence": confidence,
                "camera_id": camera_id,
            },
        )

    async def stop_camera_detection(self) -> Dict[str, Any]:
        """
        Stop realtime camera detection

        Returns:
            Dictionary containing operation status
        """
        return await self._call_tool("stop_camera_detection", {})

    async def test_connection(self) -> Dict[str, Any]:
        """
        Test connection to YOLO MCP service

        Returns:
            Dictionary containing service status and available tools
        """
        return await self._call_tool("test_connection", {})


# Example usage
async def main():
    client = YOLOClient()

    # Test connection
    status = await client.test_connection()
    print("Service status:", status)

    # Detect objects in an image
    image_path = "C:/Users/gdbt0/PycharmProjects/aisight/agent/images/dog.jpg"
    detections = await client.detect_objects(image_path)
    print("Detections:", detections)

    # Classify an image
    classifications = await client.classify_image(image_path)
    print("Classifications:", classifications)


if __name__ == "__main__":
    asyncio.run(main())
