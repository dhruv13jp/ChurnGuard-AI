
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.predictions import router as predictions_router

app = FastAPI(
    title="ChurnGuard AI",
    description="Telecom customer churn prediction API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.include_router(predictions_router)


@app.get("/")
def root():
    return {
        "message": "Welcome to ChurnGuard AI",
        "docs": "/docs",
    }
