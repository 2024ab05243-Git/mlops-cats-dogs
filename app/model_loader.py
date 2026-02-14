import torch
from src.models.model import get_model


def load_model():
    model = get_model()
    state_dict = torch.load("artifacts/model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model
