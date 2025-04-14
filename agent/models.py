from pydantic import BaseModel

class GeoGuessResult(BaseModel):
    result: str
