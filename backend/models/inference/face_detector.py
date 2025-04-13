import numpy as np
from typing import List, Dict, Any, Tuple
import logging
import cv2
from insightface.app import FaceAnalysis
import os
from pathlib import Path
import warnings
import io
from contextlib import redirect_stdout, redirect_stderr
import asyncio
from sqlalchemy import select, func
from database.models import FaceIdentity, FaceDetection, Image

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
            
            # Increase similarity threshold for stricter face matching
            self.similarity_threshold = 0.5
            
            # Create faces directory with absolute path
            backend_dir = Path(__file__).resolve().parent.parent.parent
            self.faces_dir = backend_dir / "image" / "faces"
            self.faces_dir.mkdir(exist_ok=True, parents=True)
            
            logger.info(f"Face detector initialized with faces directory: {self.faces_dir}")
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

    async def _update_face_db(self, new_embedding: np.ndarray, image_id: int, bbox: np.ndarray, landmarks: np.ndarray, confidence: float, db_session=None) -> Tuple[int, int]:
        """Update face database with new embedding and persist to SQL database.
        Returns a tuple of (identity_id, detection_id) if a match is found or a new identity is created."""
        try:
            if db_session is None:
                logger.error("No database session provided")
                return None, None

            # Query existing face identities
            query = select(FaceIdentity)
            result = await db_session.execute(query)
            identities = result.scalars().all()

            best_match = None
            best_similarity = 0.0

            # Compare with existing identities
            for identity in identities:
                identity_embedding = np.array(identity.reference_embedding)
                similarity = self._compute_similarity(new_embedding, identity_embedding)
                
                if similarity > self.similarity_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = identity
                    logger.debug(f"Found match with similarity {similarity:.3f} for face in image {image_id}")

            if best_match:
                # Use existing identity
                identity_id = best_match.id
                logger.info(f"Matched face in image {image_id} to existing identity {identity_id} with similarity {best_similarity:.3f}")
            else:
                # Count existing identities for new label
                count_query = select(func.count(FaceIdentity.id))
                count_result = await db_session.execute(count_query)
                existing_count = count_result.scalar() or 0

                # Create new identity
                new_identity = FaceIdentity(
                    label=f"Person_{existing_count + 1}",
                    reference_embedding=new_embedding.tolist()
                )
                db_session.add(new_identity)
                await db_session.flush()
                identity_id = new_identity.id
                logger.info(f"Created new identity {identity_id} for face in image {image_id}")

            # Create face detection with all required fields
            detection = FaceDetection(
                image_id=image_id,
                identity_id=identity_id,
                embedding=new_embedding.tolist(),
                confidence=confidence,
                bounding_box=bbox.tolist(),
                landmarks=landmarks.tolist() if landmarks is not None else None
            )
            db_session.add(detection)
            await db_session.flush()
            
            return identity_id, detection.id

        except Exception as e:
            logger.error(f"Error updating face database: {e}")
            return None, None

    async def detect_faces(self, image: np.ndarray, image_id: int, db_session=None) -> Dict:
        """Detect faces in an image and extract embeddings."""
        try:
            if self.app is None:
                raise ValueError("Face detector not properly initialized")
                
            # Convert image to RGB if needed
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[2] == 3 and image.dtype == np.uint8:
                pass  # Already RGB
            else:
                raise ValueError(f"Unsupported image format: {image.shape}, {image.dtype}")

            # Detect faces
            faces = self.app.get(image)
            results = []
            
            for face in faces:
                # Calculate smile intensity and eye status
                smile_intensity = self.calculate_smile_intensity(face.kps)
                eye_status = self.calculate_eye_status(face.kps)
                
                # Get embedding for face recognition
                embedding = face.embedding
                
                # Update face database
                identity_id, detection_id = await self._update_face_db(
                    embedding,
                    image_id,
                    face.bbox,
                    face.kps,
                    float(face.det_score),
                    db_session
                )
                
                if detection_id is None:
                    continue
                    
                # Save face crop only after we have the detection_id
                face_id = self._save_face_crop(image, face.bbox, image_id, detection_id)
                
                results.append({
                    "confidence": float(face.det_score),
                    "embedding": face.embedding.tolist(),
                    "bounding_box": face.bbox.tolist(),
                    "landmarks": face.kps.tolist() if face.kps is not None else None,
                    "identity_id": identity_id,
                    "face_image": face_id,
                    "attributes": {
                        "smile_intensity": smile_intensity,
                        "eye_status": eye_status
                    }
                })

            return {
                "faces_detected": True,
                "face_count": len(results),
                "faces": results
            }

        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return {
                "error": str(e),
                "faces_detected": False,
                "face_count": 0,
                "faces": []
            }

    def _save_face_crop(self, image: np.ndarray, bbox: np.ndarray, image_id: int, detection_id: int) -> str:
        """Save a cropped face image and return its ID."""
        try:
            # Add padding to the bounding box
            padding = 30
            h, w = image.shape[:2]
            x1, y1, x2, y2 = bbox
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            # Ensure valid coordinates
            if x1 >= x2 or y1 >= y2:
                logger.error(f"Invalid bounding box after padding: [{x1}, {y1}, {x2}, {y2}]")
                return None
            
            # Crop and save face
            face_img = image[int(y1):int(y2), int(x1):int(x2)]
            if face_img.size == 0:
                logger.error("Empty face crop")
                return None
            
            # Convert RGB to BGR for OpenCV
            face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
            
            # Construct face image path
            face_id = f"face_{image_id}_{detection_id}"
            face_path = self.faces_dir / f"{face_id}.jpg"
            
            # Save the image with error checking
            success = cv2.imwrite(str(face_path), face_img)
            if not success:
                logger.error(f"Failed to write face crop to {face_path}")
                return None
            
            # Verify file was created
            if not face_path.exists():
                logger.error(f"Face crop file not found after saving: {face_path}")
                return None
            
            logger.info(f"Successfully saved face crop to {face_path}")
            return face_id
            
        except Exception as e:
            logger.error(f"Error saving face crop: {str(e)}")
            return None

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