"""Model inference utilities."""
import io
import time
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from pathlib import Path
from typing import Dict, Tuple, Optional

from src.models.cnn import get_model

class ModelInference:
    """Handle model loading and inference."""
    
    def __init__(self, model_path: str = "artifacts/model.pt"):
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[torch.nn.Module] = None
        self.classes = ["cat", "dog"]
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self._load_model()
    
    def _load_model(self):
        """Load the trained model."""
        if self.model_path.exists():
            self.model = get_model("simple_cnn", num_classes=2)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded from {self.model_path}")
        else:
            print(f"Warning: Model not found at {self.model_path}")
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
    
    def predict(self, image_bytes: bytes) -> Tuple[str, float, Dict[str, float], float]:
        """
        Make prediction on image.
        
        Returns:
            Tuple of (predicted_class, confidence, probabilities_dict, inference_time_ms)
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        
        # Load and preprocess image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
        
        inference_time = (time.time() - start_time) * 1000
        
        # Get prediction
        pred_idx = probabilities.argmax().item()
        predicted_class = self.classes[pred_idx]
        confidence = probabilities[pred_idx].item()
        
        probs_dict = {cls: prob.item() for cls, prob in zip(self.classes, probabilities)}
        
        return predicted_class, confidence, probs_dict, inference_time

# Global instance
model_inference: Optional[ModelInference] = None

def get_inference_service() -> ModelInference:
    """Get or create inference service."""
    global model_inference
    if model_inference is None:
        model_inference = ModelInference()
    return model_inference
