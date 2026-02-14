import torch
from PIL import Image
import numpy as np

from app.inference import predict


class DummyModel(torch.nn.Module):
    def forward(self, x):
        # Always return logits favoring class 1 ("dog")
        return torch.tensor([[0.1, 2.0]])


def test_predict_returns_label_and_confidence():
    # Create dummy RGB image (300x300)
    dummy_image = Image.fromarray(
        np.uint8(np.random.rand(300, 300, 3) * 255)
    )

    model = DummyModel()

    result = predict(model, dummy_image)

    assert "label" in result
    assert "confidence" in result
    assert result["label"] in ["cat", "dog"]
    assert isinstance(result["confidence"], float)
