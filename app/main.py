from fastapi import FastAPI, UploadFile, File
from app.model_loader import load_model
from app.inference import predict
from PIL import Image
import io

app = FastAPI(title="Cats vs Dogs Classifier")

model = load_model()

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    prediction = predict(model, image)
    return prediction
