import cv2
import numpy as np
from pathlib import Path
import logging
from .text_recognizer import TextRecognizer
from typing import Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from database.models import Image, TextDetection, SceneClassification
from database.config import get_async_session
import datetime
from geoalchemy2.shape import WKTElement
from sqlalchemy import select

logger = logging.getLogger(__name__)

class ImageAnalyzer:
    def __init__(self):
        """Initialize the image analyzer with various models."""
        self.text_recognizer = TextRecognizer()
        logger.info("ImageAnalyzer initialized successfully")

    async def analyze_image(self, image_path: Path) -> Dict[str, Any]:
        """
        Analyze an image using various models for different features and save results to database.
        """
        try:
            # Ensure image path is absolute
            image_path = Path(image_path).resolve()
            if not image_path.exists():
                raise ValueError(f"Image file not found: {image_path}")
                
            # Read the image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not read image at {image_path}")
                
            # Validate image dimensions
            if image.shape[0] == 0 or image.shape[1] == 0:
                raise ValueError("Invalid image dimensions")

            # Extract metadata
            metadata = self._extract_metadata(image_path)

            # Analyze text
            text_analysis = await self.text_recognizer.analyze_image(image)

            # Get database session
            async with get_async_session() as session:
                # Create Image record
                db_image = Image(
                    filename=image_path.name,
                    dimensions=f"{image.shape[1]}x{image.shape[0]}",
                    format=image_path.suffix[1:],  # Remove leading dot
                    file_size=image_path.stat().st_size / 1024,  # Convert to KB
                    date_taken=metadata.get("date_taken"),
                    camera_make=metadata.get("camera_make"),
                    camera_model=metadata.get("camera_model"),
                    focal_length=metadata.get("focal_length"),
                    exposure_time=metadata.get("exposure_time"),
                    f_number=metadata.get("f_number"),
                    iso=metadata.get("iso")
                )

                # Add GPS location if available
                if metadata.get("gps"):
                    lat, lon = metadata["gps"]
                    db_image.location = WKTElement(f'POINT({lon} {lat})', srid=4326)

                session.add(db_image)
                await session.flush()  # Get the image ID

                # Save text detections
                if text_analysis["text_detected"]:
                    for block in text_analysis["text_blocks"]:
                        text_detection = TextDetection(
                            image_id=db_image.id,
                            text=block["text"],
                            confidence=block["confidence"],
                            bounding_box=block["bbox"]
                        )
                        session.add(text_detection)

                await session.commit()

            return {
                "filename": image_path.name,
                "metadata": metadata,
                "text_recognition": {
                    "text_detected": text_analysis["text_detected"],
                    "text_blocks": text_analysis["text_blocks"],
                    "total_confidence": text_analysis["total_confidence"],
                    "categories": text_analysis["categories"],
                    "raw_text": text_analysis["raw_text"]
                }
            }

        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {str(e)}")
            return {
                "filename": image_path.name,
                "error": str(e),
                "text_recognition": {
                    "text_detected": False,
                    "text_blocks": [],
                    "total_confidence": 0.0,
                    "categories": [],
                    "raw_text": ""
                }
            }

    def _extract_metadata(self, image_path: Path) -> Dict[str, Any]:
        """Extract EXIF metadata from image."""
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS
            
            metadata = {}
            with Image.open(image_path) as img:
                try:
                    exif = img._getexif()
                    if exif:
                        for tag_id, value in exif.items():
                            tag = TAGS.get(tag_id, tag_id)
                            metadata[tag] = value

                        # Extract common EXIF fields
                        if "DateTimeOriginal" in metadata:
                            metadata["date_taken"] = datetime.datetime.strptime(
                                metadata["DateTimeOriginal"], "%Y:%m:%d %H:%M:%S"
                            )
                        metadata["camera_make"] = metadata.get("Make")
                        metadata["camera_model"] = metadata.get("Model")
                        metadata["focal_length"] = float(str(metadata.get("FocalLength", "0")).split()[0])
                        metadata["exposure_time"] = str(metadata.get("ExposureTime", ""))
                        metadata["f_number"] = float(str(metadata.get("FNumber", "0")).split()[0])
                        metadata["iso"] = metadata.get("ISOSpeedRatings")

                        # Extract GPS data
                        if "GPSInfo" in metadata:
                            gps = metadata["GPSInfo"]
                            if all(i in gps for i in (1, 2, 3, 4)):
                                lat = self._convert_to_degrees(gps[2], gps[1])
                                lon = self._convert_to_degrees(gps[4], gps[3])
                                metadata["gps"] = (lat, lon)

                except Exception as e:
                    logger.error(f"Error extracting EXIF: {str(e)}")

            return metadata

        except Exception as e:
            logger.error(f"Error opening image for metadata: {str(e)}")
            return {}

    def _convert_to_degrees(self, value, ref) -> float:
        """Convert GPS coordinates to degrees."""
        d = float(value[0])
        m = float(value[1])
        s = float(value[2])
        
        degrees = d + (m / 60.0) + (s / 3600.0)
        
        if ref in ['S', 'W']:
            degrees = -degrees
            
        return degrees