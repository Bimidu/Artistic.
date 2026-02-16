"""
Patient Report Models for MongoDB

This module defines the Report model and related schemas
for saving and retrieving patient analysis reports.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
import uuid


class PatientReport(BaseModel):
    """Patient report model as stored in MongoDB"""
    id: Optional[str] = Field(None, alias="_id")
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str  # User who created the report
    patient_name: str
    analysis_date: datetime = Field(default_factory=datetime.utcnow)
    prediction: str  # ASD or TD
    confidence: float
    probabilities: Dict[str, float]
    model_used: str
    input_type: str  # audio, text, or chat_file
    features_extracted: Optional[int] = None
    transcript: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "patient_name": "John Doe",
                "prediction": "ASD",
                "confidence": 0.85,
                "probabilities": {"ASD": 0.85, "TD": 0.15},
                "model_used": "pragmatic_conversational_xgboost",
                "input_type": "audio",
                "features_extracted": 42
            }
        }


class ReportCreate(BaseModel):
    """Schema for creating a new patient report"""
    patient_name: str = Field(..., min_length=1, description="Patient name")
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    model_used: str
    input_type: str
    features_extracted: Optional[int] = None
    transcript: Optional[str] = None


class ReportResponse(BaseModel):
    """Schema for report response"""
    report_id: str
    patient_name: str
    analysis_date: datetime
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    model_used: str
    input_type: str
    features_extracted: Optional[int] = None
    transcript: Optional[str] = None
    created_at: datetime

    class Config:
        populate_by_name = True


class PatientReportsGroup(BaseModel):
    """Schema for reports grouped by patient"""
    patient_name: str
    reports: list[ReportResponse]
    report_count: int
