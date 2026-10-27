
from fastapi import APIRouter, HTTPException

from app.schemas import CustomerInput
from app.services.predictor import predict_customer

router = APIRouter(prefix="/api", tags=["Predictions"])


@router.get("/health")
def health():
    return {"status": "ok", "service": "ChurnGuard AI"}


@router.post("/predict")
def predict(customer: CustomerInput):
    try:
        return predict_customer(customer.model_dump())
    except (ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Unable to process customer data: {exc}",
        ) from exc
