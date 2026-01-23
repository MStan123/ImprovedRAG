from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Literal, Optional
from datetime import datetime
from feedback_manager import feedback_manager

router = APIRouter(prefix="/api/feedback", tags=["feedback"])


class SubmitFeedbackRequest(BaseModel):
    """Запрос на отправку оценки"""
    feedback_id: str
    rating: Literal["yes", "no"]


class AnalyticsRequest(BaseModel):
    """Запрос аналитики"""
    start_date: Optional[str] = None  # ISO format
    end_date: Optional[str] = None
    category: Optional[str] = None


@router.post("/submit")
async def submit_feedback(request: SubmitFeedbackRequest):
    """
    Endpoint для отправки оценки от пользователя

    Пример:
    POST /api/feedback/submit
    {
        "feedback_id": "abc123...",
        "rating": "yes"
    }
    """
    success = feedback_manager.submit_feedback(
        feedback_id=request.feedback_id,
        rating=request.rating
    )

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Feedback ID {request.feedback_id} not found"
        )

    return {
        "status": "success",
        "message": "Спасибо за оценку!" if request.rating == "yes" else "Спасибо за отзыв. Мы улучшимся!",
        "feedback_id": request.feedback_id,
        "rating": request.rating
    }


@router.get("/analytics")
async def get_analytics(
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        category: Optional[str] = None
):
    """
    Endpoint для получения аналитики

    Пример:
    GET /api/feedback/analytics?start_date=2024-01-01&end_date=2024-12-31
    """
    start = datetime.fromisoformat(start_date) if start_date else None
    end = datetime.fromisoformat(end_date) if end_date else None

    analytics = feedback_manager.get_analytics(
        start_date=start,
        end_date=end,
        category=category
    )

    return {
        "status": "success",
        "data": analytics
    }


@router.get("/health")
async def health_check():
    """Проверка работоспособности системы обратной связи"""
    pending_count = len(feedback_manager.pending_feedback)

    return {
        "status": "healthy",
        "pending_feedback_count": pending_count,
        "kb_version": feedback_manager.kb_version
    }

# Подключение к основному приложению FastAPI:
# from feedback_api import router as feedback_router
# app.include_router(feedback_router)