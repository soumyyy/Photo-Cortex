import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SceneClassifier:
    def __init__(self, model_path: str = "./models/weights/scene_classifier/resnet50_places365.pth.tar"):
        """Initialize Places365 model for scene classification."""
        try:
            # logger.info(f"Loading Places365 model from {model_path}")
            
            # Load the model architecture (ResNet50)
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            num_ftrs = self.model.fc.in_features
            self.model.fc = torch.nn.Linear(num_ftrs, 365)  # Places365 has 365 categories
            
            # Load Places365 weights
            # logger.info("Loading model weights...")
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
            self.model.load_state_dict(state_dict)
            self.model.eval()
            # logger.info("Model weights loaded successfully")

            # Define broader ranges for indoor/outdoor
            self.indoor_ranges = [(0, 150)]  # First 150 categories are typically indoor
            self.outdoor_ranges = [(250, 365)]  # Last 115 categories are typically outdoor

            # Define image preprocessing
            self.transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            logger.info("Places365 model initialized successfully")
        except Exception as e:
            logger.error(f"Error loading Places365 model: {str(e)}")
            self.model = None

    def predict_scene(self, image: np.ndarray) -> Dict[str, Any]:
        """Predict whether an image is indoor or outdoor."""
        if self.model is None:
            logger.warning("Scene classifier model not initialized")
            return {"scene_type": "unknown", "confidence": 0.0}

        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Prepare image for model
            input_tensor = self.transforms(image_rgb)
            input_batch = input_tensor.unsqueeze(0)
            
            with torch.no_grad():
                output = self.model(input_batch)
                probabilities = torch.nn.functional.softmax(output, dim=1)[0]

                # Calculate indoor/outdoor probabilities
                indoor_prob = 0.0
                outdoor_prob = 0.0
                total_weight = 0.0

                for idx, prob in enumerate(probabilities):
                    prob_value = prob.item()
                    if idx < 150:  # Indoor categories
                        weight = 1.0 if idx < 100 else 0.8
                        indoor_prob += prob_value * weight
                        total_weight += weight
                    elif idx > 250:  # Outdoor categories
                        weight = 1.0 if idx > 300 else 0.8
                        outdoor_prob += prob_value * weight
                        total_weight += weight

                # Normalize probabilities
                if total_weight > 0:
                    indoor_prob = (indoor_prob / total_weight) * 100
                    outdoor_prob = (outdoor_prob / total_weight) * 100

                # Determine scene type
                if indoor_prob > outdoor_prob:
                    scene_type = "indoor"
                    confidence = indoor_prob
                else:
                    scene_type = "outdoor"
                    confidence = outdoor_prob

                # logger.info(f"Scene prediction: {scene_type} (confidence: {confidence:.2f}%)")
                # logger.debug(f"Indoor prob: {indoor_prob:.2f}%, Outdoor prob: {outdoor_prob:.2f}%")

                return {
                    "scene_type": scene_type,
                    "confidence": float(confidence / 100.0),  # Convert to 0-1 range
                    "probabilities": {
                        "indoor": float(indoor_prob / 100.0),
                        "outdoor": float(outdoor_prob / 100.0)
                    }
                }

        except Exception as e:
            logger.error(f"Error in scene prediction: {str(e)}")
            return {"scene_type": "unknown", "confidence": 0.0}