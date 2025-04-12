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
uv run ruff format --fix
```

To run the tests, run the following command:

```
uv run pytest tests.py

```

### Development Resources:
- How to initialise a FastAPI project with uv - [link](https://www.youtube.com/watch?v=igWlYl3asKw)
- Docker setup with FastAPI - [link](https://www.youtube.com/watch?v=DA6gywtTLL8)

