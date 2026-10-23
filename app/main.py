from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.model import Predictor

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="Kalp Yetmezliği Tahmini")
predictor = Predictor()


class PatientData(BaseModel):
    Age: int = Field(ge=1, le=120)
    Sex: Literal["M", "F"]
    ChestPainType: Literal["ATA", "NAP", "ASY", "TA"]
    RestingBP: int = Field(ge=0, le=300)
    Cholesterol: int = Field(ge=0, le=700)
    FastingBS: Literal[0, 1]
    RestingECG: Literal["Normal", "ST", "LVH"]
    MaxHR: int = Field(ge=30, le=250)
    ExerciseAngina: Literal["Y", "N"]
    Oldpeak: float = Field(ge=-5, le=10)
    ST_Slope: Literal["Up", "Flat", "Down"]


@app.post("/api/predict")
def predict(data: PatientData):
    return predictor.predict(data.model_dump())


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
