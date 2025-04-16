import cv2
import numpy as np
from pathlib import Path
import logging
from .text_recognizer import TextRecognizer
from .face_detector import FaceDetector
from .object_detector import ObjectDetector
from .scene_classifier import SceneClassifier
# from .clip_encoder import ClipEncoder
from typing import Dict, Any, List
from sqlalchemy.ext.asyncio import AsyncSession, async_session
from sqlalchemy.orm import declarative_base, scoped_session
from sqlalchemy.orm.session import sessionmaker
from database.models import Image, TextDetection, SceneClassification, ObjectDetection, FaceDetection, FaceIdentity, ExifMetadata
from database.config import get_db
import datetime
from geoalchemy2.shape import WKTElement
from sqlalchemy import select, delete
from sqlalchemy.future import select
import json

logger = logging.getLogger(__name__)

class ImageAnalyzer:
    def __init__(self):
        """Initialize the image analyzer with various models."""
        self.text_recognizer = TextRecognizer()
        self.face_detector = FaceDetector()
        self.object_detector = ObjectDetector()
        self.scene_classifier = SceneClassifier()
        # self.clip_encoder = ClipEncoder()
        logger.info("ImageAnalyzer initialized successfully")

    async def analyze_image_with_session(self, image_path: Path, session: AsyncSession) -> int:
        """
        Analyze an image using various models and save results using provided session.
        Handles existing images by retrieving their ID instead of inserting duplicates.
        """
        image_id = None # Initialize image_id
        image_record = None
        image = None # Initialize image variable

        try:
            # Ensure image path is absolute
            image_path = Path(image_path).resolve()
            if not image_path.exists():
                error_msg = f"Image file not found: {image_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            # --- Check if image already exists in DB ---
            stmt = select(Image).where(Image.filename == image_path.name)
            result = await session.execute(stmt)
            image_record = result.scalar_one_or_none()

            if image_record:
                logger.info(f"Image {image_path.name} already exists in DB with ID: {image_record.id}. Cleaning up old analysis and cutouts.")
                image_id = image_record.id

                # --- Clean up old face cutout files first ---
                faces_dir = self.face_detector.faces_dir
                if faces_dir.exists():
                    for face_file in faces_dir.glob(f"face_{image_id}_*.jpg"):
                        try:
                            face_file.unlink()
                            logger.debug(f"Deleted old face cutout: {face_file}")
                        except Exception as e:
                            logger.warning(f"Failed to delete face cutout {face_file}: {e}")

                # --- Clean up old analysis records ---
                try:
                    # Delete EXIF metadata first (due to foreign key constraints)
                    await session.execute(delete(ExifMetadata).where(ExifMetadata.image_id == image_id))
                    await session.flush()  # Ensure EXIF deletion is complete
                    
                    # Delete other analysis records
                    await session.execute(delete(FaceDetection).where(FaceDetection.image_id == image_id))
                    await session.execute(delete(ObjectDetection).where(ObjectDetection.image_id == image_id))
                    await session.execute(delete(TextDetection).where(TextDetection.image_id == image_id))
                    await session.execute(delete(SceneClassification).where(SceneClassification.image_id == image_id))
                    
                    await session.commit()  # Commit the deletions
                    logger.debug(f"Successfully cleaned up old analysis records for image ID {image_id}")
                except Exception as e:
                    logger.error(f"Error cleaning up old records: {e}")
                    await session.rollback()
                    raise

                # --- Read image for re-analysis ---
                image = cv2.imread(str(image_path))
                if image is None:
                    logger.error(f"Failed to read existing image file: {image_path}")
                    raise ValueError(f"Failed to read existing image file: {image_path}")
                logger.info(f"Successfully read existing image for re-analysis: {image_path}")

            else:
                logger.info(f"Image {image_path.name} not found in DB. Creating new record.")
                # --- Create new image record ---
                # Load image bytes first to get metadata and ensure readability
                image = cv2.imread(str(image_path))
                if image is None:
                     logger.error(f"Failed to read new image file: {image_path}")
                     raise ValueError(f"Failed to read new image file: {image_path}")

                height, width = image.shape[:2]
                file_size = image_path.stat().st_size
                exif_data = self._extract_metadata(image_path)
                created_at = datetime.datetime.now()
                date_taken = exif_data.get("date_taken")
                gps_info = exif_data.get("gps")
                latitude = gps_info[0] if gps_info else None
                longitude = gps_info[1] if gps_info else None
                location = f"POINT({longitude} {latitude})" if latitude and longitude else None

                image_record = Image(
                    filename=image_path.name,
                    created_at=created_at,
                    dimensions=f"{width}x{height}",
                    format=image_path.suffix.lower().strip('.'),
                    file_size=file_size,
                    date_taken=date_taken,
                    latitude=latitude,
                    longitude=longitude,
                    location=WKTElement(location, srid=4326) if location else None,
                    embedding=None
                )
                session.add(image_record)
                try:
                    await session.flush()  # Try to save and get the ID
                    image_id = image_record.id
                    logger.info(f"Successfully added new image record for {image_path.name} with ID: {image_id}")
                except Exception as e:
                    logger.error(f"Error saving new image record: {e}")
                    await session.rollback()
                    raise

            # --- Run face detection ---
            try:
                face_results = await self.face_detector.detect_faces(image, image_id, session)
                if face_results["faces_detected"]:
                    logger.info(f"Detected {face_results['face_count']} faces in {image_path.name}")
            except Exception as e:
                logger.error(f"Error in face detection: {e}")
                face_results = {"faces_detected": False, "face_count": 0, "faces": []}

            # --- Run object detection ---
            try:
                objects = self.object_detector.detect_objects(image)
                if objects:
                    for obj in objects:
                        obj_detection = ObjectDetection(
                            image_id=image_id,
                            label=obj["label"],
                            confidence=obj["confidence"],
                            bounding_box=obj["bbox"]
                        )
                        session.add(obj_detection)
                    logger.info(f"Detected {len(objects)} objects in {image_path.name}")
            except Exception as e:
                logger.error(f"Error in object detection: {e}")

            # --- Run text recognition ---
            try:
                text_blocks = self.text_recognizer.detect_text(image)
                if text_blocks:
                    for block in text_blocks:
                        text_detection = TextDetection(
                            image_id=image_id,
                            text=block["text"],
                            confidence=block["confidence"],
                            bounding_box=block["bbox"]
                        )
                        session.add(text_detection)
                    logger.info(f"Detected {len(text_blocks)} text blocks in {image_path.name}")
            except Exception as e:
                logger.error(f"Error in text recognition: {e}")

            # --- Run scene classification ---
            try:
                scene_type, confidence = self.scene_classifier.classify_scene(image)
                if scene_type:
                    scene_classification = SceneClassification(
                        image_id=image_id,
                        scene_type=scene_type,
                        confidence=confidence
                    )
                    session.add(scene_classification)
                    logger.info(f"Classified scene as {scene_type} with confidence {confidence:.2f}")
            except Exception as e:
                logger.error(f"Error in scene classification: {e}")

            # --- Save EXIF metadata ---
            try:
                exif_data = self._extract_metadata(image_path).get("exif", {})
                if any(exif_data.values()):
                    exif_metadata = ExifMetadata(
                        image_id=image_id,
                        camera_make=exif_data.get("camera_make"),
                        camera_model=exif_data.get("camera_model"),
                        focal_length=exif_data.get("focal_length"),
                        exposure_time=exif_data.get("exposure_time"),
                        f_number=exif_data.get("f_number"),
                        iso=exif_data.get("iso")
                    )
                    session.add(exif_metadata)
                    logger.info(f"Added EXIF metadata for {image_path.name}")
            except Exception as e:
                logger.error(f"Error saving EXIF metadata: {e}")

            # --- Final commit ---
            await session.commit()
            logger.info(f"Successfully completed analysis of {image_path.name}")
            return image_id

        except Exception as e:
            logger.error(f"Error in analyze_image_with_session: {e}")
            await session.rollback()
            raise

    async def analyze_image(self, image_path: Path) -> Dict[str, Any]:
        """
        Legacy method that creates its own session.
        Use analyze_image_with_session for FastAPI endpoints.
        """
        try:
            # Ensure image path is absolute
            image_path = Path(image_path).resolve()
            if not image_path.exists():
                raise ValueError(f"Image file not found: {image_path}")
            
            # Create a new session
            async with get_db() as session:
                # Use the new method with session handling
                image_id = await self.analyze_image_with_session(image_path, session)
                
                # Query the database to get the full analysis
                query = select(Image).options(
                    joinedload(Image.face_detections),
                    joinedload(Image.object_detections),
                    joinedload(Image.text_detections),
                    joinedload(Image.scene_classifications),
                    joinedload(Image.exif_metadata)
                ).where(Image.id == image_id)
                
                result = await session.execute(query)
                image = result.unique().scalar_one_or_none()
                
                if not image:
                    logger.error(f"Failed to retrieve image with ID {image_id}")
                    return {
                        "faces": [],
                        "embeddings": [],
                        "objects": [],
                        "text_recognition": {"text_detected": False, "text_blocks": []},
                        "scene_classification": {"scene_type": None, "confidence": 0}
                    }
                
                # Format the results
                faces_data = []
                embeddings_data = []
                
                for face in image.face_detections:
                    face_data = {
                        "bbox": face.bounding_box,
                        "score": face.confidence,
                        "landmarks": face.landmarks
                    }
                    faces_data.append(face_data)
                    
                    if face.embedding is not None:
                        embeddings_data.append(face.embedding)
                
                # Build metadata dictionary
                metadata = {
                    "dimensions": image.dimensions,
                    "format": image.format,
                    "file_size": str(image.file_size),
                    "date_taken": image.date_taken.isoformat() if image.date_taken else None,
                    "gps": {
                        "latitude": float(image.latitude),
                        "longitude": float(image.longitude)
                    } if image.latitude is not None and image.longitude is not None else None
                }
                
                # Keep EXIF data separate from metadata
                exif_data = {}
                if image.exif_metadata:
                    exif_data = {
                        "camera_make": image.exif_metadata.camera_make,
                        "camera_model": image.exif_metadata.camera_model,
                        "focal_length": image.exif_metadata.focal_length,
                        "exposure_time": image.exif_metadata.exposure_time,
                        "f_number": image.exif_metadata.f_number,
                        "iso": image.exif_metadata.iso
                    }
                
                # Build the response
                return {
                    "filename": image.filename,
                    "metadata": metadata,
                    "exif": exif_data,
                    "faces": faces_data,
                    "embeddings": embeddings_data,
                    "objects": [obj.label for obj in image.object_detections],
                    "text_recognition": {
                        "text_detected": len(image.text_detections) > 0,
                        "text_blocks": [
                            {
                                "text": text.text,
                                "confidence": text.confidence,
                                "bbox": text.bounding_box
                            } for text in image.text_detections
                        ]
                    },
                    "scene_classification": next(
                        (
                            {
                                "scene_type": scene.scene_type,
                                "confidence": scene.confidence
                            } for scene in image.scene_classifications
                        ),
                        {"scene_type": None, "confidence": 0}
                    )
                }
                
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return {
                "faces": [],
                "embeddings": [],
                "objects": [],
                "text_recognition": {"text_detected": False, "text_blocks": []},
                "scene_classification": {"scene_type": None, "confidence": 0},
                "metadata": {},
                "exif": {}
            }

    def _extract_metadata(self, image_path: Path) -> Dict[str, Any]:
        """Extract EXIF metadata from image."""
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS
            
            metadata = {
                "dimensions": None,
                "format": None,
                "file_size": None,
                "date_taken": None,
                "gps": None
            }
            
            exif_metadata = {
                "camera_make": None,
                "camera_model": None,
                "focal_length": None,
                "exposure_time": None,
                "f_number": None,
                "iso": None
            }
            
            with Image.open(image_path) as img:
                # Set basic metadata
                metadata["dimensions"] = f"{img.width}x{img.height}"
                metadata["format"] = image_path.suffix[1:].lower()  # Remove leading dot
                metadata["file_size"] = image_path.stat().st_size / 1024  # KB
                
                try:
                    exif = img._getexif()
                    if exif:
                        raw_exif = {}
                        for tag_id, value in exif.items():
                            tag = TAGS.get(tag_id, tag_id)
                            raw_exif[tag] = value

                        # Extract date taken
                        if "DateTimeOriginal" in raw_exif:
                            metadata["date_taken"] = datetime.datetime.strptime(
                                raw_exif["DateTimeOriginal"], "%Y:%m:%d %H:%M:%S"
                            )
                            
                        # Extract EXIF data (separate from basic metadata)
                        exif_metadata["camera_make"] = raw_exif.get("Make")
                        exif_metadata["camera_model"] = raw_exif.get("Model")
                        
                        if "FocalLength" in raw_exif:
                            exif_metadata["focal_length"] = float(str(raw_exif["FocalLength"]).split()[0])
                            
                        exif_metadata["exposure_time"] = str(raw_exif.get("ExposureTime", ""))
                        
                        if "FNumber" in raw_exif:
                            exif_metadata["f_number"] = float(str(raw_exif["FNumber"]).split()[0])
                            
                        exif_metadata["iso"] = raw_exif.get("ISOSpeedRatings")

                        # Extract GPS data
                        if "GPSInfo" in raw_exif:
                            gps = raw_exif["GPSInfo"]
                            if all(i in gps for i in (1, 2, 3, 4)):
                                lat = self._convert_to_degrees(gps[2], gps[1])
                                lon = self._convert_to_degrees(gps[4], gps[3])
                                metadata["gps"] = (lat, lon)

                except Exception as e:
                    logger.error(f"Error extracting EXIF: {str(e)}")

            # Combine metadata and exif_metadata
            result = {**metadata}
            result["exif"] = exif_metadata
            
            return result
            
        except Exception as e:
            logger.error(f"Error opening image for metadata: {str(e)}")
            return {"exif": {}}

    def _convert_to_degrees(self, value, ref) -> float:
        """Convert GPS coordinates to degrees."""
        d = float(value[0])
        m = float(value[1])
        s = float(value[2])
        
        degrees = d + (m / 60.0) + (s / 3600.0)
        
        if ref in ['S', 'W']:
            degrees = -degrees
            
        return degrees