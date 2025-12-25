from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from pydantic import BaseModel
from contextlib import asynccontextmanager
import pickle
import uvicorn
import os

from models.model import train_and_evaluate
from utils.preprocessing import preprocess_data

templates = Jinja2Templates(directory="templates")

model = None
scaler = None

def load_model():
    global model, scaler
    try:
        model = pickle.load(open("model.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
        print("Model loaded from disk")
    except FileNotFoundError:
        print("Training model...")
        model, scaler = train_and_evaluate()

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(lifespan=lifespan)

#data models
class PatientData(BaseModel):
    age: int
    sex: int
    cp: int            # chest pain
    trtbps: int        # resting blood pressure
    chol: int          # cholesterol
    fbs: int           # fasting blood sugar
    restecg: int       # resting electrocardiographic results
    thalachh: int      # max heart rate
    exng: int          # exercise induced angina
    oldpeak: float     # measure of ST depression
    slp: int           # slope
    caa: int           # number of vessels
    thall: int         # thalassemia

class PredictResponse(BaseModel):
    prediction: int

#routes
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_model=PredictResponse)
def predict(data: PatientData):

    if model is None or scaler is None:
        return {"prediction": -1}

    record = data.model_dump()
    X = preprocess_data(record).astype("float32")
    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]
    return {"prediction": int(prediction)}


#health check
class HealthCheck(BaseModel):
    status: str = "OK"


@app.get("/health", response_model=HealthCheck)
def health():
    return HealthCheck(status="OK")

#run
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
