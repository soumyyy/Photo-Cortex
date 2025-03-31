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
            self.similarity_threshold = 0.5
            
            # Create faces directory
            self.faces_dir = Path("./image/faces")
            self.faces_dir.mkdir(exist_ok=True, parents=True)
            
            logger.info("Face detector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize face detector: {str(e)}")
            self.app = None

    def calculate_smile_intensity(self, landmarks) -> float:
        """Calculate smile intensity based on facial landmarks."""
        try:
            if landmarks.shape[0] != 5:
                logger.warning(f"Unexpected landmark format: {landmarks.shape}")
                return 0.0
                
            # For 5 keypoints: [left_eye, right_eye, nose, left_mouth, right_mouth]
            left_mouth = landmarks[3]
            right_mouth = landmarks[4]
            mouth_width = np.linalg.norm(right_mouth - left_mouth)
            
            # Use the eye distance as a reference for scale
            left_eye = landmarks[0]
            right_eye = landmarks[1]
            eye_distance = np.linalg.norm(right_eye - left_eye)
            
            # Calculate ratio of mouth width to eye distance
            ratio = mouth_width / (eye_distance + 1e-6)
            norm_ratio = (ratio - 0.8) / 0.4  # Normalize to 0-1 range
            
            # Calculate mouth corner elevation relative to nose
            nose = landmarks[2]
            mouth_center_y = (left_mouth[1] + right_mouth[1]) / 2
            elevation = (mouth_center_y - nose[1]) / eye_distance
            norm_elevation = (elevation - 0.3) / 0.3
            
            # Combine metrics with more weight on elevation
            intensity = 0.4 * norm_ratio + 0.6 * norm_elevation
            
            return float(np.clip(intensity, 0, 1))
            
        except Exception as e:
            logger.error(f"Error in smile calculation: {str(e)}")
            return 0.0

    def calculate_eye_status(self, landmarks) -> dict:
        """
        Calculate eye status (open/closed) using eye landmarks.
        For 5-point landmarks: [left_eye, right_eye, nose, left_mouth, right_mouth]
        """
        try:
            if landmarks is None or landmarks.shape[0] != 5:
                return {"status": "unknown", "left_ear": 0.0, "right_ear": 0.0}

            # Get eye points
            left_eye = landmarks[0]  # Left eye center
            right_eye = landmarks[1]  # Right eye center
            nose = landmarks[2]  # Nose tip
            
            # Calculate vertical distances (from eyes to nose)
            left_eye_nose_dist = np.linalg.norm(left_eye - nose)
            right_eye_nose_dist = np.linalg.norm(right_eye - nose)
            
            # Calculate horizontal distance between eyes
            eye_distance = np.linalg.norm(right_eye - left_eye)
            
            # Calculate eye aspect ratios
            left_ear = left_eye_nose_dist / (eye_distance + 1e-6)
            right_ear = right_eye_nose_dist / (eye_distance + 1e-6)
            
            # Average eye ratio
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Determine eye status based on the ratio
            # These thresholds might need adjustment based on your specific use case
            if avg_ear > 0.4:  # Higher ratio indicates more open eyes
                status = "open"
            elif avg_ear > 0.25:  # Middle range might indicate partially open
                status = "partially open"
            else:  # Lower ratio indicates closed eyes
                status = "closed"
            
            logger.debug(f"Eye metrics - Left: {left_ear:.3f}, Right: {right_ear:.3f}, Avg: {avg_ear:.3f}, Status: {status}")
            
            return {
                "status": status,
                "left_ear": float(left_ear),
                "right_ear": float(right_ear)
            }
            
        except Exception as e:
            logger.error(f"Error in eye status calculation: {str(e)}")
            return {"status": "unknown", "left_ear": 0.0, "right_ear": 0.0}

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

                    # Calculate facial attributes
                    smile_score = self.calculate_smile_intensity(face.kps) if hasattr(face, 'kps') else 0.0
                    
                    # Classify age as young/old
                    age_value = int(face.age) if hasattr(face, 'age') and face.age is not None else None
                    age_category = "unknown"
                    if age_value is not None:
                        age_category = "young" if age_value < 40 else "old"
                    
                    gender = "male" if hasattr(face, 'gender') and face.gender == 1 else "female"
                    eye_status = self.calculate_eye_status(face.kps) if hasattr(face, 'kps') else {"status": "unknown", "left_ear": 0.0, "right_ear": 0.0}
                    
                    # Update face path to use image/faces
                    face_relative_path = f"image/faces/{face_filename}"
                    
                    detection = {
                        'bbox': bbox,
                        'score': float(face.det_score),
                        'landmarks': face.kps.tolist() if hasattr(face, 'kps') else None,
                        'face_image': face_relative_path,
                        'attributes': {
                            'age': age_category,
                            'gender': gender,
                            'smile_intensity': smile_score,
                            'eye_status': eye_status["status"],
                            'eye_metrics': {
                                "left_ear": eye_status["left_ear"],
                                "right_ear": eye_status["right_ear"]
                            }
                        }
                    }
                    
                    detections.append(detection)
                    embeddings.append(embedding.tolist())
                    
                    # Update face database with correct path
                    if image_name:
                        self._update_face_db(embedding, image_name, face_relative_path)
                except Exception as e:
                    logger.error(f"Error processing face {idx}: {str(e)}")
            return {
                "faces": detections,
                "embeddings": embeddings
            }

        except Exception as e:
            if "No face detected" not in str(e):
                logger.error(f"Error in face detection for {image_name}: {str(e)}")
            return {"faces": [], "embeddings": []}

    def _save_face_crop(self, image: np.ndarray, bbox: np.ndarray, image_name: str, face_idx: int) -> str:
        """Save a cropped face image and return its ID.
        
        Args:
            image: Full image array
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            image_name: Original image filename
            face_idx: Index of the face in the image
            
        Returns:
            Face ID string
        """
        try:
            # Add padding to the bounding box
            padding = 30
            h, w = image.shape[:2]
            x1, y1, x2, y2 = bbox
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            # Crop and save the face image
            face_img = image[int(y1):int(y2), int(x1):int(x2)]
            face_id = f"face_{Path(image_name).stem}_{face_idx}"
            face_filename = f"{face_id}.jpg"
            face_path = self.faces_dir / face_filename
            
            cv2.imwrite(str(face_path), face_img)
            return face_id
            
        except Exception as e:
            logger.error(f"Error saving face crop: {str(e)}")
            return None

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

    def get_unique_faces(self) -> list:
        """Get list of unique faces with their associated images.
        
        Returns:
            List of dictionaries containing:
            - id: unique identifier for the face group
            - images: list of image filenames containing this face
            - face_images: list of cropped face image filenames
        """
        try:
            unique_faces = []
            for idx, (_, data) in enumerate(self.face_db.items()):
                unique_faces.append({
                    'id': idx,
                    'images': list(data['images']),
                    'face_images': list(data['face_images'])
                })
            return unique_faces
        except Exception as e:
            logger.error(f"Error getting unique faces: {str(e)}")
            return []