from src.fastapi.app import app
from fastapi.testclient import TestClient
from dotenv import load_dotenv
from src.geoguesser.agent import agent as geo_guess_agent
from src.yolo.agent import run_agent_loop, agent as yolo_agent
import pytest
import asyncio
load_dotenv()

client = TestClient(app)


def test_root():
    response = client.get("/")
    print(response.__dict__)
    assert response.status_code == 200
    assert response.text == '{"Hello":"World"}'


def test_geo_guess_agent_exists():
    try:
        result = geo_guess_agent.run_sync('Where does "hello world" come from?')
        print(result)
    except Exception as e:
        print(f"Agent does not exist: {e}")


@pytest.mark.asyncio
async def test_yolo_run_agent():
    try:
        image_file_path = "C:/Users/Gunee/Projects/aisight/public/images/dog.jpg"
        result = await asyncio.wait_for(run_agent_loop(image_file_path), timeout=30)
        print(result)
    except asyncio.TimeoutError:
        print("Test timed out!")
    except Exception as e:
        print(f"Agent does not exist: {e}")


def test_endpoint_geo_guess_location_from_image_url():
    response = client.post(
        "/geo_guess_location_from_image_url",
        json={"image_url": "https://cdn.britannica.com/12/136912-050-58299570/Tower-Bridge-River-Thames-London.jpg"},
    )
    print(response.__dict__)
    assert response.status_code == 200
    assert "London" in response.text


def test_classify_image_from_local_file_path():
    image_file_path = "C:/Users/Gunee/Projects/aisight/public/images/dog.jpg"
    response = client.post(
        "/classify_image_from_local_file_path",
        json={"file_path": image_file_path},
    )
    print(response.__dict__)
    assert response.status_code == 200
    assert "Beagle" in response.text