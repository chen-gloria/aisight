from pydantic_ai import Agent, BinaryContent, UnexpectedModelBehavior
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.gemini import GeminiModelSettings
from dotenv import load_dotenv
from typing import Union
from pydantic_models import GeoGuessResult

load_dotenv()

model = GeminiModel(
    "gemini-2.0-flash",
)

agent = Agent(
    model,
    result_type=Union[GeoGuessResult, str],
    system_prompt="You are an OSINT investigator. Your job is to geolocate where the photos are taken. "
    "Provide the country, region, and city name of the location. Please pinpoint the exact location with latitude and longitude where the photo was taken. "
    "Could you always explain your methodology and how you concluded? Provide steps to verify your work. "
    "Also, mention the percentage of how sure you are of the place you have identified it to be and add a Google Maps link having google.com/maps to the exact location.",
)


async def geo_guess_location_from_image(image_bytes: bytes):
    try:
        return await agent.run(
            [
                BinaryContent(data=image_bytes, media_type="image/png"),
            ],
            model_settings=GeminiModelSettings(
                temperature=0.1,
                top_p=0.5,
                max_output_tokens=4096,
                response_modalities=["TEXT"],
                gemini_safety_settings=[
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "OFF",
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "OFF",
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "OFF",
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "OFF",
                    },
                ],
            ),
        )

    except UnexpectedModelBehavior as e:
        print(e)
        """
        Safety settings triggered, body:
        <safety settings details>
        """
