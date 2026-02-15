from fastapi import FastAPI, UploadFile, File, Request
from pydantic import BaseModel
from app.model_loader import load_model
from app.inference import predict
from PIL import Image
import io
import logging
import time

# --------------------------------------------------
# Logging Configuration
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# --------------------------------------------------
# FastAPI App
# --------------------------------------------------

app = FastAPI(title="Cats vs Dogs Classifier")

model = load_model()

# --------------------------------------------------
# In-App Metrics
# --------------------------------------------------

request_count = 0
total_latency = 0.0
total_predictions = 0
correct_predictions = 0


# --------------------------------------------------
# Middleware for Logging & Latency Tracking
# --------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    global request_count, total_latency

    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    request_count += 1
    total_latency += process_time

    logger.info(
        f"Path={request.url.path} | "
        f"Method={request.method} | "
        f"Status={response.status_code} | "
        f"Latency={process_time:.4f}s | "
        f"TotalRequests={request_count}"
    )

    return response


# --------------------------------------------------
# Health Endpoint
# --------------------------------------------------

@app.get("/health")
def health():
    return {"status": "healthy"}


# --------------------------------------------------
# Prediction Endpoint
# --------------------------------------------------

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    prediction = predict(model, image)

    logger.info(
        f"Prediction={prediction['label']} | "
        f"Confidence={prediction['confidence']:.4f}"
    )

    return prediction


# --------------------------------------------------
# Feedback Endpoint (Model Performance Tracking)
# --------------------------------------------------

class Feedback(BaseModel):
    predicted_label: str
    true_label: str


@app.post("/feedback")
def feedback(data: Feedback):
    global total_predictions, correct_predictions

    total_predictions += 1

    if data.predicted_label == data.true_label:
        correct_predictions += 1

    logger.info(
        f"Feedback received | "
        f"Predicted={data.predicted_label} | "
        f"True={data.true_label}"
    )

    return {"message": "Feedback recorded"}


# --------------------------------------------------
# Metrics Endpoint
# --------------------------------------------------

@app.get("/metrics")
def metrics():
    avg_latency = total_latency / request_count if request_count > 0 else 0
    accuracy = (
        correct_predictions / total_predictions
        if total_predictions > 0 else 0
    )

    return {
        "total_requests": request_count,
        "average_latency": avg_latency,
        "total_predictions": total_predictions,
        "correct_predictions": correct_predictions,
        "accuracy": accuracy
    }
