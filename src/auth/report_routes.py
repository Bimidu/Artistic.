"""
Patient Report Routes

FastAPI routes for saving and retrieving patient analysis reports.
"""

from fastapi import APIRouter, HTTPException, status, Depends
from src.auth.report_models import (
    ReportCreate, ReportResponse, PatientReportsGroup
)
from src.auth.dependencies import get_current_active_user
from src.auth.models import UserResponse
from src.database import get_database
from datetime import datetime
from typing import List
import uuid

router = APIRouter(prefix="/api/reports", tags=["Reports"])


@router.post("/save", response_model=ReportResponse, status_code=status.HTTP_201_CREATED)
async def save_report(
    report_data: ReportCreate,
    current_user: UserResponse = Depends(get_current_active_user)
):
    """
    Save a new patient analysis report
    
    - **patient_name**: Name of the patient
    - **prediction**: Prediction result (ASD or TD)
    - **confidence**: Confidence score
    - **probabilities**: Class probabilities
    - **model_used**: Name of the model used
    - **input_type**: Type of input (audio, text, chat_file)
    """
    db = get_database()
    
    # Create report document
    report_id = str(uuid.uuid4())
    report_doc = {
        "_id": report_id,
        "report_id": report_id,
        "user_id": current_user.id,
        "patient_name": report_data.patient_name,
        "analysis_date": datetime.utcnow(),
        "prediction": report_data.prediction,
        "confidence": report_data.confidence,
        "probabilities": report_data.probabilities,
        "model_used": report_data.model_used,
        "input_type": report_data.input_type,
        "features_extracted": report_data.features_extracted,
        "transcript": report_data.transcript,
        "created_at": datetime.utcnow(),
    }
    
    # Insert into database
    await db.patient_reports.insert_one(report_doc)
    
    # Return report response
    return ReportResponse(**{
        "report_id": report_id,
        "patient_name": report_data.patient_name,
        "analysis_date": report_doc["analysis_date"],
        "prediction": report_data.prediction,
        "confidence": report_data.confidence,
        "probabilities": report_data.probabilities,
        "model_used": report_data.model_used,
        "input_type": report_data.input_type,
        "features_extracted": report_data.features_extracted,
        "transcript": report_data.transcript,
        "created_at": report_doc["created_at"],
    })


@router.get("/my-reports", response_model=List[ReportResponse])
async def get_my_reports(
    current_user: UserResponse = Depends(get_current_active_user)
):
    """
    Get all reports created by the current user
    """
    db = get_database()
    
    # Find all reports for this user
    cursor = db.patient_reports.find({"user_id": current_user.id})
    reports = await cursor.to_list(length=None)
    
    # Convert to response models
    return [
        ReportResponse(
            report_id=report["report_id"],
            patient_name=report["patient_name"],
            analysis_date=report["analysis_date"],
            prediction=report["prediction"],
            confidence=report["confidence"],
            probabilities=report["probabilities"],
            model_used=report["model_used"],
            input_type=report["input_type"],
            features_extracted=report.get("features_extracted"),
            transcript=report.get("transcript"),
            created_at=report["created_at"],
        )
        for report in reports
    ]


@router.get("/by-patient", response_model=List[PatientReportsGroup])
async def get_reports_by_patient(
    current_user: UserResponse = Depends(get_current_active_user)
):
    """
    Get reports grouped by patient name for the current user
    """
    db = get_database()
    
    # Find all reports for this user
    cursor = db.patient_reports.find({"user_id": current_user.id})
    reports = await cursor.to_list(length=None)
    
    # Group by patient name
    patient_groups = {}
    for report in reports:
        patient_name = report["patient_name"]
        if patient_name not in patient_groups:
            patient_groups[patient_name] = []
        
        patient_groups[patient_name].append(
            ReportResponse(
                report_id=report["report_id"],
                patient_name=report["patient_name"],
                analysis_date=report["analysis_date"],
                prediction=report["prediction"],
                confidence=report["confidence"],
                probabilities=report["probabilities"],
                model_used=report["model_used"],
                input_type=report["input_type"],
                features_extracted=report.get("features_extracted"),
                transcript=report.get("transcript"),
                created_at=report["created_at"],
            )
        )
    
    # Convert to PatientReportsGroup
    result = []
    for patient_name, patient_reports in patient_groups.items():
        # Sort reports by date (newest first)
        patient_reports.sort(key=lambda r: r.analysis_date, reverse=True)
        result.append(
            PatientReportsGroup(
                patient_name=patient_name,
                reports=patient_reports,
                report_count=len(patient_reports)
            )
        )
    
    # Sort patient groups alphabetically
    result.sort(key=lambda g: g.patient_name)
    
    return result
