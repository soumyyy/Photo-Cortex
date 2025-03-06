import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image, ImageEnhance
import numpy as np
from typing import Dict, Any, List, Callable
import logging
import cv2
from pathlib import Path
import os
import warnings
import urllib.request

# For Places365 integration
from torchvision import models, transforms

# Filter out PyTorch TypedStorage deprecation warning
warnings.filterwarnings('ignore', message='TypedStorage is deprecated')

logger = logging.getLogger(__name__)

class SceneClassifier:
    """Scene classifier using CLIP ViT-B/16 for zero-shot classification, with optional Places365 integration."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SceneClassifier, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the CLIP model (and Places365 model) for scene classification."""
        if self._initialized:
            return
            
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Define scene categories and their descriptions
            self.scene_categories = {
                "Nature": [
                    "a photo of nature, showing beaches, forests, mountains, or natural landscapes",
                    "a scenic natural landscape with trees, water, or mountains",
                    "an outdoor nature scene with natural elements like sky, water, or vegetation",
                    "a peaceful nature setting with natural lighting and organic elements",
                    "a photograph taken outdoors in nature, showing the beauty of the natural world",
                    "NOT a photo of urban areas or man-made structures"
                ],
                "City": [
                    "an urban city scene with buildings, streets, and architecture",
                    "a cityscape showing modern buildings and urban infrastructure",
                    "a photo of city streets, buildings, or urban areas",
                    "a bustling metropolitan area with skyscrapers and urban development",
                    "a photograph of city life showing urban planning and architecture",
                    "NOT a photo of natural landscapes or rural areas"
                ],
                "Event": [
                    "a photo of an organized event like a wedding, concert, or festival",
                    "a large gathering or celebration at a formal event",
                    "a public event with many people attending",
                    "a professionally organized celebration or ceremony",
                    "a formal gathering with decorations and planned activities",
                    "NOT a casual house party or informal gathering"
                ],
                "Party": [
                    "a casual party or celebration with people having fun",
                    "an indoor party scene with people celebrating",
                    "a social gathering or house party with decorations",
                    "an informal celebration with friends and family",
                    "a birthday party or casual get-together at home",
                    "NOT a formal event or professional ceremony"
                ],
                "Food": [
                    "a photo of food, meals, or prepared dishes",
                    "food items or dishes served at a restaurant or home",
                    "culinary dishes or prepared meals on plates",
                    "appetizing food photography showing texture and presentation",
                    "close-up shots of delicious meals and food items",
                    "NOT documents or non-food items"
                ],
                "Documents": [
                    "a photo of documents, papers, or printed text",
                    "scanned documents or official papers with text",
                    "business documents, forms, or printed materials",
                    "legal documents or paperwork with clear text",
                    "professional documents with letterhead or official formatting",
                    "NOT handwritten notes or casual papers"
                ],
                "Receipts": [
                    "a photo of a receipt, bill, or transaction record",
                    "store receipts or financial documents with prices",
                    "paper receipts from stores or businesses",
                    "detailed transaction receipts with itemized lists",
                    "proof of purchase or payment documentation",
                    "NOT general documents or handwritten notes"
                ]
            }
            
            # Load CLIP model and processor
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
            
            # Move CLIP model to device and set to eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Cache text features for efficiency
            logger.info("Computing text features...")
            self.text_features = self._compute_text_features()
            
            # Define confidence thresholds for validation
            self.confidence_thresholds = {
                "Nature": 0.25,
                "City": 0.22,
                "Event": 0.30,
                "Party": 0.30,
                "Food": 0.25,
                "Documents": 0.35,
                "Receipts": 0.35
            }
            
            # Temperature parameter for confidence calibration
            self.temperature = 0.8
            
            # Load Places365 model and define its preprocessing
            self.places_model = self._load_places365_model()
            self.places_preprocess = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            
            self._initialized = True
            logger.info("Scene classifier (CLIP + Places365) initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize scene classifier: {e}")
            self.model = None
            self.places_model = None
            
    def _compute_text_features(self) -> Dict[str, torch.Tensor]:
        """Pre-compute text features for all scene descriptions."""
        text_features = {}
        with torch.no_grad():
            for category, descriptions in self.scene_categories.items():
                inputs = self.processor(
                    text=descriptions,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                category_features = self.model.get_text_features(**inputs)
                text_features[category] = category_features.mean(dim=0)
        return text_features
    
    def _load_places365_model(self):
        """Load the pre-trained Places365 model (ResNet50) with Places365 weights."""
        try:
            model = models.resnet50(num_classes=365)
            weights_dir = Path("./models/weights/scene_classifier")
            checkpoint_path = weights_dir / "resnet50_places365.pth.tar"
            weights_dir.mkdir(parents=True, exist_ok=True)
            
            if not checkpoint_path.exists():
                logger.info("Downloading Places365 weights...")
                url = "https://dl.fbaipublicfiles.com/places365/resnet50_places365.pth.tar"
                try:
                    urllib.request.urlretrieve(url, checkpoint_path)
                    logger.info("Successfully downloaded Places365 weights")
                except Exception as e:
                    logger.error(f"Failed to download Places365 weights: {e}")
                    return None
            
            if not checkpoint_path.exists():
                logger.error("Places365 checkpoint not found and could not be downloaded")
                return None
                
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
            model.load_state_dict(state_dict)
            model = model.to(self.device)
            model.eval()
            logger.info("Successfully loaded Places365 model")
            return model
            
        except Exception as e:
            logger.error(f"Error loading Places365 model: {e}")
            return None

    def validate_scene(self, scene_type: str, confidence: float) -> str:
        """Validate scene classification based on confidence thresholds."""
        if confidence > 0.8:
            return scene_type
        if confidence < self.confidence_thresholds.get(scene_type, 0.3):
            return "Other"
        return scene_type
    
    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE to the image in LAB color space for contrast enhancement."""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return enhanced
    
    def _generate_augmentations(self, image_pil: Image.Image) -> List[Image.Image]:
        """Generate a list of augmented images for test-time augmentation."""
        augmentations: List[Callable[[Image.Image], Image.Image]] = []
        # Identity (no change)
        augmentations.append(lambda img: img)
        # Horizontal flip
        augmentations.append(lambda img: img.transpose(Image.FLIP_LEFT_RIGHT))
        # Slight rotation +10
        augmentations.append(lambda img: img.rotate(10, expand=True))
        # Slight rotation -10
        augmentations.append(lambda img: img.rotate(-10, expand=True))
        # Slight zoom in
        augmentations.append(lambda img: img.resize(
            (int(img.width * 1.1), int(img.height * 1.1)), Image.Resampling.LANCZOS))
        # Slight zoom out
        augmentations.append(lambda img: img.resize(
            (int(img.width * 0.9), int(img.height * 0.9)), Image.Resampling.LANCZOS))
        # Brightness increase
        augmentations.append(lambda img: ImageEnhance.Brightness(img).enhance(1.2))
        # Contrast increase
        augmentations.append(lambda img: ImageEnhance.Contrast(img).enhance(1.2))
        
        augmented_images = [fn(image_pil) for fn in augmentations]
        return augmented_images
    
    def classify_scene(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Classify an image using CLIP's zero-shot classification with enhanced augmentations.
        Args:
            image (np.ndarray): Input image in BGR format (OpenCV style)
        Returns:
            Dict[str, Any]: Scene classification results with confidence scores
        """
        if self.model is None:
            return {
                "scene_type": "Unknown",
                "confidence": 0.0,
                "all_scene_scores": {category: 0.0 for category in self.scene_categories}
            }
        
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            mean_brightness = np.mean(gray)
            # If image is too dark, apply CLAHE for contrast enhancement
            if mean_brightness < 50:
                image_rgb = self._apply_clahe(image_rgb)
            
            image_pil = Image.fromarray(image_rgb)
            
            # Generate augmented images
            augmented_images = self._generate_augmentations(image_pil)
            
            # Store predictions from each augmentation
            predictions = []
            for aug_image in augmented_images:
                inputs = self.processor(
                    images=aug_image,
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    image_features = self.model.get_image_features(**inputs)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    similarities = {}
                    for category, text_feature in self.text_features.items():
                        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
                        similarity = (100.0 * image_features @ text_feature.unsqueeze(-1)).squeeze()
                        similarities[category] = float(similarity)
                    predictions.append(similarities)
            
            # Aggregate predictions using median for robustness
            aggregated_scores = {}
            for category in self.scene_categories.keys():
                cat_scores = [pred[category] for pred in predictions]
                aggregated_scores[category] = np.median(cat_scores)
            
            # Apply temperature scaling before softmax
            scores_tensor = torch.tensor(list(aggregated_scores.values()))
            probabilities = torch.nn.functional.softmax(scores_tensor / self.temperature, dim=0)
            
            all_scene_scores = {
                category: float(prob)
                for category, prob in zip(aggregated_scores.keys(), probabilities)
            }
            
            top_scene = max(all_scene_scores.items(), key=lambda x: x[1])
            scene_type = top_scene[0]
            confidence = top_scene[1]
            validated_scene = self.validate_scene(scene_type, confidence)
            
            result = {
                "scene_type": validated_scene,
                "confidence": confidence if validated_scene == scene_type else 0.0,
                "all_scene_scores": all_scene_scores
            }
            logger.debug(f"CLIP scene classification result: {result}")
            return result
        
        except Exception as e:
            logger.error(f"Error in scene classification: {e}")
            return {
                "scene_type": "Unknown",
                "confidence": 0.0,
                "all_scene_scores": {category: 0.0 for category in self.scene_categories}
            }
    
    def classify_places(self, image: np.ndarray) -> Dict[str, float]:
        """
        Classify an image using the Places365 model and map its predictions
        to a subset of custom scene categories.
        Args:
            image (np.ndarray): Input image in BGR format (OpenCV style)
        Returns:
            Dict[str, float]: Mapped category scores from Places365
        """
        if self.places_model is None:
            return {category: 0.0 for category in self.scene_categories}
        
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            input_tensor = self.places_preprocess(image_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = self.places_model(input_tensor)
                probs = torch.nn.functional.softmax(logits, dim=1)[0]
            
            # Dummy mapping from Places365 class indices to custom categories.
            mapping = {
                "Nature": [10, 50, 80],
                "City": [20, 60, 100],
                "Event": [30, 70, 110],
                "Party": [40, 90, 130],
                "Food": [5, 15, 25]
            }
            custom_scores = {category: 0.0 for category in self.scene_categories}
            for category, indices in mapping.items():
                valid_indices = [i for i in indices if i < probs.shape[0]]
                if valid_indices:
                    score = probs[valid_indices].sum().item()
                    custom_scores[category] = score
            return custom_scores
        
        except Exception as e:
            logger.error(f"Error in Places365 classification: {e}")
            return {category: 0.0 for category in self.scene_categories}
    
    def classify_scene_combined(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Classify an image using a combination of CLIP and Places365 predictions.
        The results are combined (averaged) for overlapping categories.
        Args:
            image (np.ndarray): Input image in BGR format (OpenCV style)
        Returns:
            Dict[str, Any]: Combined classification results
        """
        clip_result = self.classify_scene(image)
        places_scores = self.classify_places(image)
        
        combined_scores = {}
        for category in self.scene_categories:
            clip_score = clip_result["all_scene_scores"].get(category, 0.0)
            places_score = places_scores.get(category, 0.0)
            # For categories without Places365 mapping (Documents, Receipts), use only CLIP.
            if category in ["Documents", "Receipts"]:
                combined_scores[category] = clip_score
            else:
                combined_scores[category] = (clip_score + places_score) / 2
        
        top_category, top_confidence = max(combined_scores.items(), key=lambda x: x[1])
        validated_scene = self.validate_scene(top_category, top_confidence)
        final_confidence = top_confidence if validated_scene == top_category else 0.0
        
        return {
            "scene_type": validated_scene,
            "confidence": final_confidence,
            "all_scene_scores": combined_scores,
            "clip_scores": clip_result["all_scene_scores"],
            "places_scores": places_scores
        }
    
    # Maintain backward compatibility
    predict_scene = classify_scene

# Example usage:
# classifier = SceneClassifier()
# image = cv2.imread("backend/image/IMG_9407.jpg")
# result = classifier.classify_scene_combined(image)
# print(result)