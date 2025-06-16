from ts.torch_handler.base_handler import BaseHandler
import torch
from torchvision import transforms
from PIL import Image
import io
import torch.nn as nn
import torchvision.models as models
import os
import logging

logger = logging.getLogger(__name__)


class SkinLesionHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.667, 0.530, 0.552], std=[0.154, 0.178, 0.197]
                ),
            ]
        )
        self.labels = [
            "actinic keratosis",
            "basal cell carcinoma",
            "dermatofibroma",
            "melanoma",
            "nevus",
            "pigmented benign keratosis",
            "seborrheic keratosis",
            "squamous cell carcinoma",
            "vascular lesion",
        ]
        self.initialized = False

    def initialize(self, context):
        # Get model directory from context
        self.model_dir = context.system_properties.get("model_dir")
        logger.info(f"Model directory: {self.model_dir}")

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load model
        self.model = self._load_model()
        self.model = self.model.to(self.device)
        self.model.eval()

        # Warm up the model
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            self.model(dummy_input)

        self.initialized = True
        logger.info("Handler initialized successfully")

    def _load_model(self):
        logger.info("Loading model weights")
        NUM_CLASSES = 9

        # Create model architecture
        model = models.vit_b_16(weights=None)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, NUM_CLASSES)

        # Load weights
        model_path = os.path.join(self.model_dir, "best_model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}")

        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        logger.info("Model loaded successfully")
        return model

    def preprocess(self, data):
        logger.info("Preprocessing image")

        # Get image data
        if isinstance(data, list):
            image = data[0].get("body") or data[0].get("data")
        else:
            image = data

        if image is None:
            raise ValueError("No image data found in request")

        # Convert to PIL image
        image = Image.open(io.BytesIO(image)).convert("RGB")

        # Apply transformations
        image = self.transform(image)

        # Add batch dimension and move to device
        return image.unsqueeze(0).to(self.device)

    def inference(self, data):
        logger.info("Running inference")
        with torch.no_grad():
            return self.model(data)

    def postprocess(self, inference_output):
        logger.info("Postprocessing results")
        preds = inference_output[0]

        # Get predicted class
        pred_idx = torch.argmax(preds).item()
        confidence = torch.nn.functional.softmax(preds, dim=0)[pred_idx].item()
        pred_class = self.labels[pred_idx]

        return [
            {
                "class": pred_class,
                "label_index": pred_idx,
                "confidence": round(confidence, 4),
            }
        ]
