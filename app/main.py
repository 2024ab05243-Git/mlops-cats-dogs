from fastapi import FastAPI, UploadFile, File, Request
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

# Simple in-app metric
request_count = 0


# --------------------------------------------------
# Middleware for Logging & Latency Tracking
# --------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    global request_count

    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    request_count += 1

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

    return prediction
