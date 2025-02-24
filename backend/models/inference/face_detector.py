import numpy as np
from typing import List, Dict, Any
import logging
import cv2
from insightface.app import FaceAnalysis
import os
from pathlib import Path
import warnings
import io
from contextlib import redirect_stdout, redirect_stderr

# Filter the specific warning from insightface
warnings.filterwarnings('ignore', category=FutureWarning, module='insightface.utils.transform')

logger = logging.getLogger(__name__)

class FaceDetector:
    def __init__(self, model_path: str = "./models/weights/insightface"):
        """Initialize InsightFace model."""
        try:
            # Initialize face analysis model with suppressed output
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.app = FaceAnalysis(
                    name="buffalo_l",
                    root=model_path,
                    providers=['CPUExecutionProvider']
                )
                self.app.prepare(ctx_id=-1, det_size=(640, 640))
            
            # Dictionary to store face embeddings and their associated images
            self.face_db = {}
            self.similarity_threshold = 0.6
            
            # Create faces directory
            self.faces_dir = Path("./images/faces")
            self.faces_dir.mkdir(exist_ok=True, parents=True)
            
            logger.info("Face detector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize face detector: {str(e)}")
            self.app = None

    def detect_faces(self, image: np.ndarray, image_name: str = "") -> Dict[str, Any]:
        """
        Detect faces in an image and extract embeddings.
        Args:
            image: numpy array of shape (H, W, 3) in BGR format
            image_name: name of the image file
        Returns:
            Dictionary containing face detections and embeddings
        """
        if self.app is None:
            return {"faces": [], "embeddings": []}

        try:
            # Basic image validation
            if image is None or not isinstance(image, np.ndarray):
                return {"faces": [], "embeddings": []}

            if image.size == 0:
                return {"faces": [], "embeddings": []}

            # Ensure image is in BGR format (OpenCV default)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

            # Get face detections with embeddings
            faces = self.app.get(image)
            
            if faces is None or len(faces) == 0:
                return {"faces": [], "embeddings": []}

            # Process each face
            detections = []
            embeddings = []
            
            for idx, face in enumerate(faces):
                try:
                    # Only include faces with high detection confidence
                    if face.det_score < 0.5:
                        continue
                        
                    bbox = face.bbox.astype(int).tolist()
                    embedding = face.embedding
                    
                    # Skip if embedding is None or has wrong dimensions
                    if embedding is None or embedding.shape[0] != 512:
                        continue
                    
                    # Extract and save face cutout
                    x1, y1, x2, y2 = [int(coord) for coord in bbox]
                    # Add padding
                    padding = 30
                    h, w = image.shape[:2]
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(w, x2 + padding)
                    y2 = min(h, y2 + padding)
                    
                    face_img = image[y1:y2, x1:x2]
                    face_filename = f"face_{Path(image_name).stem}_{idx}.jpg"
                    face_path = self.faces_dir / face_filename
                    cv2.imwrite(str(face_path), face_img)
                    
                    detection = {
                        'bbox': bbox,
                        'confidence': float(face.det_score),
                        'landmarks': face.landmark.tolist() if face.landmark is not None else None,
                        'face_image': f"faces/{face_filename}"
                    }
                    
                    detections.append(detection)
                    embeddings.append(embedding.tolist())
                    
                    # Update face database
                    if image_name:
                        self._update_face_db(embedding, image_name, f"faces/{face_filename}")
                except Exception:
                    continue
            
            return {
                "faces": detections,
                "embeddings": embeddings
            }

        except Exception as e:
            if "No face detected" not in str(e):
                logger.error(f"Error in face detection for {image_name}: {str(e)}")
            return {"faces": [], "embeddings": []}

    def _update_face_db(self, new_embedding: np.ndarray, image_name: str, face_image: str):
        """Update face database with new embedding."""
        try:
            if new_embedding.shape[0] != 512:
                return
                
            new_embedding_bytes = new_embedding.tobytes()
            found_match = False
            best_similarity = 0
            best_embedding = None
            
            for existing_embedding, data in self.face_db.items():
                try:
                    existing_embedding_array = np.frombuffer(existing_embedding, dtype=np.float32).reshape(-1)
                    
                    if existing_embedding_array.shape[0] != 512:
                        continue
                        
                    similarity = self._compute_similarity(existing_embedding_array, new_embedding)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_embedding = existing_embedding
                except Exception:
                    continue
            
            if best_similarity > self.similarity_threshold:
                if 'images' not in self.face_db[best_embedding]:
                    self.face_db[best_embedding]['images'] = set()
                if 'face_images' not in self.face_db[best_embedding]:
                    self.face_db[best_embedding]['face_images'] = set()
                self.face_db[best_embedding]['images'].add(image_name)
                self.face_db[best_embedding]['face_images'].add(face_image)
                found_match = True
            
            if not found_match:
                self.face_db[new_embedding_bytes] = {
                    'images': {image_name},
                    'face_images': {face_image}
                }
                
        except Exception:
            pass

    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        try:
            if embedding1.shape[0] != 512 or embedding2.shape[0] != 512:
                return 0.0
                
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
        except Exception:
            return 0.0

    def get_unique_faces(self) -> List[Dict[str, Any]]:
        """Get list of unique faces and their associated images."""
        try:
            unique_faces = []
            
            sorted_groups = sorted(
                self.face_db.items(),
                key=lambda x: len(x[1]['images']),
                reverse=True
            )
            
            for idx, (_, data) in enumerate(sorted_groups):
                unique_faces.append({
                    "id": idx,
                    "images": sorted(list(data['images'])),
                    "face_images": sorted(list(data['face_images']))
                })
            
            return unique_faces
        except Exception as e:
            logger.error(f"Error getting unique faces: {str(e)}")
            return []