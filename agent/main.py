from __future__ import annotations as _annotations
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from agent import geo_guess_location_from_image
import httpx
from pydantic import BaseModel
load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"Hello": "World"}

class ImageUrlRequest(BaseModel):
    image_url: str

@app.post("/geo_guess_location_from_image_url")
async def geo_guess_location_from_image_url(request: ImageUrlRequest):
    image_url = request.image_url
    try:
        if not image_url.startswith(('http://', 'https://')):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Please enter a valid URL starting with http:// or https://"
            )

        image_response = httpx.get(image_url)
        image_response.raise_for_status()

        content_type = image_response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"The URL does not point to an image. Content type: {content_type}"
            )

        image_bytes_data = image_response.content
        result = await geo_guess_location_from_image(image_bytes_data)
        return result.data

    except httpx.RequestError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error fetching the image: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred: {str(e)}"
        )
