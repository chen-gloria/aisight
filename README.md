# AI agent for working with images and videos

- The existing mini agent is built using the PydanticAI (https://ai.pydantic.dev/) library. 
- The agent uses the gemini-2.0-flash model with multimodal capabilities for image recognition. 
- Demo built with Streamlit.
- FastAPI is used for API endpoints.

<img src="https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=Pydantic&logoColor=white" />

<img src="https://img.shields.io/badge/fastapi-109989?style=for-the-badge&logo=FASTAPI&logoColor=white" />

<img src="https://img.shields.io/badge/Google%20Gemini-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white" />

<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white" />

## How to run it locally

### Requirements

- Python 3.13
- Google Cloud Gemini API key. Add it to the `.env` file as `GEMINI_API_KEY`. Check the example file `.env.example`
- uv package manager (https://docs.astral.sh/uv/getting-started/installation/)

### Steps

1. Install the dependencies

```
uv sync
```

2. Run the demo app in the browser

```
uv run streamlit run streamlit_app.py
```

3. Run the development server using

```
uv run fastapi dev
```

or

```
docker run -p 8000:8000 backend
```
Inside the image, the uvicorn application is running on port 8000.


### Tools

To check the code for linting errors, run the following command:

```
uv run ruff check --fix



```

To format the code, run the following command:

```
uv run ruff format
```

To run the tests, run the following command:

```
uv run pytest tests.py

```

### Development Resources:
- How to initialise a FastAPI project with uv - [link](https://www.youtube.com/watch?v=igWlYl3asKw)
- Docker setup with FastAPI - [link](https://www.youtube.com/watch?v=DA6gywtTLL8)


# YOLO MCP Service

A powerful YOLO (You Only Look Once) computer vision service that integrates with Claude AI through Model Context Protocol (MCP). This service enables Claude to perform object detection, segmentation, classification, and real-time camera analysis using state-of-the-art YOLO models.

![](https://badge.mcpx.dev?type=server 'MCP Server')


## Features

- Object detection, segmentation, classification, and pose estimation
- Real-time camera integration for live object detection
- Support for model training, validation, and export
- Comprehensive image analysis combining multiple models
- Support for both file paths and base64-encoded images
- Seamless integration with Claude AI

## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- Git (optional, for cloning the repository)

### Environment Setup

1. Create a directory for the project and navigate to it:
   ```bash
   mkdir yolo-mcp-service
   cd yolo-mcp-service
   ```

2. Download the project files or clone from repository:
   ```bash
   # If you have the files, copy them to this directory
   # If using git:
   git clone https://github.com/GongRzhe/YOLO-MCP-Server.git .
   ```

3. Create a virtual environment:
   ```bash
   # On Windows
   python -m venv .venv
   
   # On macOS/Linux
   python3 -m venv .venv
   ```

4. Activate the virtual environment:
   ```bash
   # On Windows
   .venv\Scripts\activate
   
   # On macOS/Linux
   source .venv/bin/activate
   ```

5. Run the setup script:
   ```bash
   python setup.py
   ```
   
   The setup script will:
   - Check your Python version
   - Create a virtual environment (if not already created)
   - Install required dependencies
   - Generate an MCP configuration file (mcp-config.json)
   - Output configuration information for different MCP clients including Claude

6. Note the output from the setup script, which will look similar to:
   ```
   MCP configuration has been written to: /path/to/mcp-config.json
   
   MCP configuration for Cursor:
   
   /path/to/.venv/bin/python /path/to/server.py
   
   MCP configuration for Windsurf/Claude Desktop:
   {
     "mcpServers": {
       "yolo-service": {
         "command": "/path/to/.venv/bin/python",
         "args": [
           "/path/to/server.py"
         ],
         "env": {
           "PYTHONPATH": "/path/to"
         }
       }
     }
   }
   
   To use with Claude Desktop, merge this configuration into: /path/to/claude_desktop_config.json
   ```

### Downloading YOLO Models

Before using the service, you need to download the YOLO models. The service looks for models in the following directories:
- The current directory where the service is running
- A `models` subdirectory
- Any other directory configured in the `CONFIG["model_dirs"]` variable in server.py

Create a models directory and download some common models:

```bash
# Create models directory
mkdir models

# Download YOLOv8n for basic object detection
curl -L https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -o models/yolov8n.pt

# Download YOLOv8n-seg for segmentation
curl -L https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt -o models/yolov8n-seg.pt

# Download YOLOv8n-cls for classification
curl -L https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-cls.pt -o models/yolov8n-cls.pt

# Download YOLOv8n-pose for pose estimation
curl -L https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt -o models/yolov8n-pose.pt
```

For Windows PowerShell users:
```powershell
# Create models directory
mkdir models

# Download models using Invoke-WebRequest
Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt" -OutFile "models/yolov8n.pt"
Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt" -OutFile "models/yolov8n-seg.pt"
Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-cls.pt" -OutFile "models/yolov8n-cls.pt"
Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt" -OutFile "models/yolov8n-pose.pt"
```

### Configuring Claude

To use this service with Claude:

1. For Claude web: Set up the service on your local machine and use the configuration provided by the setup script in your MCP client.

2. For Claude Desktop:
   - Run the setup script and note the configuration output
   - Locate your Claude Desktop configuration file (the path is provided in the setup script output)
   - Add or merge the configuration into your Claude Desktop configuration file
   - Restart Claude Desktop

## Using YOLO Tools in Claude

### 1. First Check Available Models

Always check which models are available on your system first:

```
I'd like to use the YOLO tools. Can you first check which models are available on my system?

<function_calls>
<invoke name="list_available_models">
</invoke>
</function_calls>
```

### 2. Detecting Objects in an Image

For analyzing an image file on your computer:

```
Can you analyze this image file for objects?

<function_calls>
<invoke name="analyze_image_from_path">
<parameter name="image_path">/path/to/your/image.jpg</parameter>
<parameter name="confidence">0.3</parameter>
</invoke>
</function_calls>
```

You can also specify a different model:

```
Can you analyze this image using a different model?

<function_calls>
<invoke name="analyze_image_from_path">
<parameter name="image_path">/path/to/your/image.jpg</parameter>
<parameter name="model_name">yolov8n.pt</parameter>
<parameter name="confidence">0.4</parameter>
</invoke>
</function_calls>
```

### 3. Running Comprehensive Image Analysis

For more detailed analysis that combines object detection, classification, and more:

```
Can you perform a comprehensive analysis on this image?

<function_calls>
<invoke name="comprehensive_image_analysis">
<parameter name="image_path">/path/to/your/image.jpg</parameter>
<parameter name="confidence">0.3</parameter>
</invoke>
</function_calls>
```

### 4. Image Segmentation

For identifying object boundaries and creating segmentation masks:

```
Can you perform image segmentation on this photo?

<function_calls>
<invoke name="segment_objects">
<parameter name="image_data">/path/to/your/image.jpg</parameter>
<parameter name="is_path">true</parameter>
<parameter name="model_name">yolov8n-seg.pt</parameter>
</invoke>
</function_calls>
```

### 5. Image Classification

For classifying the entire image content:

```
What does this image show? Can you classify it?

<function_calls>
<invoke name="classify_image">
<parameter name="image_data">/path/to/your/image.jpg</parameter>
<parameter name="is_path">true</parameter>
<parameter name="model_name">yolov8n-cls.pt</parameter>
<parameter name="top_k">5</parameter>
</invoke>
</function_calls>
```

### 6. Using Your Computer's Camera

Start real-time object detection using your computer's camera:

```
Can you turn on my camera and detect objects in real-time?

<function_calls>
<invoke name="start_camera_detection">
<parameter name="model_name">yolov8n.pt</parameter>
<parameter name="confidence">0.3</parameter>
</invoke>
</function_calls>
```

Get the latest camera detections:

```
What are you seeing through my camera right now?

<function_calls>
<invoke name="get_camera_detections">
</invoke>
</function_calls>
```

Stop the camera when finished:

```
Please turn off the camera.

<function_calls>
<invoke name="stop_camera_detection">
</invoke>
</function_calls>
```

### 7. Advanced Model Operations

#### Training a Custom Model

```
I want to train a custom object detection model on my dataset.

<function_calls>
<invoke name="train_model">
<parameter name="dataset_path">/path/to/your/dataset</parameter>
<parameter name="model_name">yolov8n.pt</parameter>
<parameter name="epochs">50</parameter>
</invoke>
</function_calls>
```

#### Validating a Model

```
Can you validate the performance of my model on a test dataset?

<function_calls>
<invoke name="validate_model">
<parameter name="model_path">/path/to/your/trained/model.pt</parameter>
<parameter name="data_path">/path/to/validation/dataset</parameter>
</invoke>
</function_calls>
```

#### Exporting a Model to Different Formats

```
I need to export my YOLO model to ONNX format.

<function_calls>
<invoke name="export_model">
<parameter name="model_path">/path/to/your/model.pt</parameter>
<parameter name="format">onnx</parameter>
</invoke>
</function_calls>
```

### 8. Testing Connection

Check if the YOLO service is running correctly:

```
Is the YOLO service running correctly?

<function_calls>
<invoke name="test_connection">
</invoke>
</function_calls>
```

## Troubleshooting

### Camera Issues

If the camera doesn't work, try different camera IDs:

```
<function_calls>
<invoke name="start_camera_detection">
<parameter name="camera_id">1</parameter>  <!-- Try 0, 1, or 2 -->
</invoke>
</function_calls>
```

### Model Not Found

If a model is not found, make sure you've downloaded it to one of the configured directories:

```
<function_calls>
<invoke name="get_model_directories">
</invoke>
</function_calls>
```

### Performance Issues

For better performance with limited resources, use the smaller models (e.g., yolov8n.pt instead of yolov8x.pt)


# uv-docker-example

An example project for using uv in Docker images, with a focus on best practices for developing with
the project mounted in the local image.

See the [uv Docker integration guide](https://docs.astral.sh/uv/guides/integration/docker/) for more
background.

## Trying it out

A [`run.sh`](./run.sh) utility is provided for quickly building the image and starting a container.
This script demonstrates best practices for developing using the container, using bind mounts for
the project and virtual environment directories.

To build and run the web application in the container using `docker run`:

```console
$ ./run.sh
```

Then, check out [`http://localhost:8000`](http://localhost:8000) to see the website.

A Docker compose configuration is also provided to demonstrate best practices for developing using
the container with Docker compose. Docker compose is more complex than using `docker run`, but has
more robust support for various workflows.

To build and run the web application using Docker compose:

```
docker compose up --watch 
```

By default, the image is set up to start the web application. However, a command-line interface is
provided for demonstration purposes as well. 

To run the command-line entrypoint in the container:

```console
$ ./run.sh hello
```

## Project overview

### Dockerfile

The [`Dockerfile`](./Dockerfile) defines the image and includes:

- Installation of uv
- Installing the project dependencies and the project separately for optimal image build caching
- Placing environment executables on the `PATH`
- Running the web application

The [`multistage.Dockerfile`](./multistage.Dockerfile) example extends the `Dockerfile` example to
use multistage builds to reduce the final size of the image.

The [`standalone.Dockerfile`](./standalone.Dockerfile) example extends the `multistage.Dockerfile`
example to use a managed Python interpreter in a multistage build instead of the system interpreter
that comes with the base image.

### Dockerignore file

The [`.dockerignore`](./.dockerignore) file includes an entry for the `.venv` directory to ensure the
`.venv` is not included in image builds. Note that the `.dockerignore` file is not applied to volume
mounts during container runs.

### Run script

The [`run.sh`](./run.sh) script includes an example of invoking `docker run` for local development,
mounting the source code for the project into the container so that edits are reflected immediately.

### Docker compose file

The [compose.yml](./compose.yml) file includes a Docker compose definition for the web application.
It includes a [`watch`
directive](https://docs.docker.com/compose/file-watch/#compose-watch-versus-bind-mounts) for Docker
compose, which is a best-practice method for updating the container on local changes.

### Application code

The Python application code for the project is at
[`src/uv_docker_example/__init__.py`](./src/uv_docker_example/__init__.py) — there's a command line
entrypoint and a basic FastAPI application — both of which just display "hello world" output.

### Project definition

The project at [`pyproject.toml`](./pyproject.toml) includes Ruff as an example development
dependency, includes FastAPI as a dependency, and defines a `hello` entrypoint for the application.

## Useful commands

To check that the environment is up-to-date after image builds:

```console
$ ./run.sh uv sync --frozen
Audited 2 packages ...
```

To enter a `bash` shell in the container:

```console
$ ./run.sh /bin/bash
```

To build the image without running anything:

```console
$ docker build .
```

To build the multistage image:

```console
$ docker build . --file multistage.Dockerfile
```