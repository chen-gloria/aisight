from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class GeoGuessResult(BaseModel):
    result: str


# Response Models
class DetectionBox(BaseModel):
    box: List[float] = Field(
        ..., description="Bounding box coordinates [x1, y1, x2, y2]"
    )
    confidence: float = Field(..., description="Detection confidence score")
    class_id: int = Field(..., description="Class ID")
    class_name: str = Field(..., description="Class name")


class DetectionResult(BaseModel):
    detections: List[DetectionBox] = Field(..., description="List of detected objects")
    image_shape: List[int] = Field(..., description="Image dimensions [width, height]")


class DetectionResponse(BaseModel):
    results: List[DetectionResult] = Field(
        ..., description="Detection results for each image"
    )
    model_used: str = Field(..., description="Model used for detection")
    total_detections: int = Field(..., description="Total number of detections")
    source: str = Field(..., description="Source of the image (path or base64)")
    error: Optional[str] = Field(None, description="Error message if detection failed")


class Classification(BaseModel):
    class_id: int = Field(..., description="Class ID")
    class_name: str = Field(..., description="Class name")
    probability: float = Field(..., description="Classification probability")


class ClassificationResult(BaseModel):
    classifications: List[Classification] = Field(
        ..., description="List of classifications"
    )
    image_shape: List[int] = Field(..., description="Image dimensions [width, height]")


class ClassificationResponse(BaseModel):
    source: str = Field(..., description="Source of the image (path or base64)")
    error: Optional[str] = Field(
        None, description="Error message if classification failed"
    )


class ModelInfo(BaseModel):
    model_name: str = Field(..., description="Model name")
    task: str = Field(
        ..., description="Model task (detect, segment, pose, classify, obb)"
    )
    yaml: str = Field(..., description="Model YAML configuration")
    pt_path: Optional[str] = Field(None, description="Path to model weights file")
    class_names: Dict[int, str] = Field(..., description="Class names dictionary")
    error: Optional[str] = Field(
        None, description="Error message if info retrieval failed"
    )


class ModelMetrics(BaseModel):
    precision: float = Field(..., description="Precision metric")
    recall: float = Field(..., description="Recall metric")
    mAP50: float = Field(..., description="mAP50 metric")
    mAP50_95: float = Field(..., description="mAP50-95 metric")


class TrainingResponse(BaseModel):
    status: str = Field(..., description="Training status")
    model_path: str = Field(..., description="Path to trained model")
    epochs_completed: int = Field(..., description="Number of completed epochs")
    final_metrics: ModelMetrics = Field(..., description="Final training metrics")
    error: Optional[str] = Field(None, description="Error message if training failed")


class ValidationResponse(BaseModel):
    status: str = Field(..., description="Validation status")
    metrics: ModelMetrics = Field(..., description="Validation metrics")
    error: Optional[str] = Field(None, description="Error message if validation failed")


class CameraStatus(BaseModel):
    status: str = Field(..., description="Camera status")
    message: str = Field(..., description="Status message")
    error: Optional[str] = Field(None, description="Error message if operation failed")
