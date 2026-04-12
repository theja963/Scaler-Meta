from pydantic import BaseModel
from typing import Optional, Dict


class AnomalyAction(BaseModel):
    label: str  # "normal" or "anomaly"
    severity: Optional[str] = None  # "low", "medium", "high"
    explanation: str


class AnomalyObservation(BaseModel):
    message: str
    image_path: Optional[str]
    description: Optional[str]
    reward: float
    done: bool
    metadata: Optional[Dict] = None