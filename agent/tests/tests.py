from main import app
from fastapi.testclient import TestClient
from dotenv import load_dotenv
from agent import agent
load_dotenv()

client = TestClient(app)


def test_root():
    response = client.get("/")
    print(response.__dict__)
    assert response.status_code == 200
    assert response.text == '{"Hello":"World"}'


def test_agent_exists():
    try:
        result = agent.run_sync('Where does "hello world" come from?')
        print(result)
    except Exception as e:
        print(f"Agent does not exist: {e}")


def test_geo_guess_location_from_image_url():
    response = client.post("/geo_guess_location_from_image_url", json={"image_url": "https://iili.io/3Hs4FMg.png"})
    print(response.__dict__)
    assert response.status_code == 200
    assert "Pydantic" in response.text

