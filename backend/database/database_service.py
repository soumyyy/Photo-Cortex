from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, delete, update
from sqlalchemy.orm import joinedload
from typing import Optional, List, Dict, Any
import logging
from .models import Image, FaceDetection, ObjectDetection, TextDetection, SceneClassification, ExifMetadata, FaceIdentity
from geoalchemy2.functions import ST_GeogFromText
from datetime import datetime
import pathlib

logger = logging.getLogger(__name__)

class DatabaseService:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    async def get_session(self) -> AsyncSession:
        """Get a new database session."""
        return self.session_factory()

    async def get_image_by_filename(self, filename: str) -> Optional[Image]:
        """Get image by filename."""
        async with self.session_factory() as session:
            async with session.begin():
                query = select(Image).where(Image.filename == filename)
                result = await session.execute(query)
                return result.scalar_one_or_none()

    async def save_image_analysis(self, image_data: Dict[str, Any], exif_data: Dict[str, Any]) -> Optional[Image]:
        """Save image analysis results to database."""
        try:
            # Validate required fields
            required_fields = ['filename', 'dimensions', 'format', 'file_size']
            missing_fields = [field for field in required_fields if not image_data.get(field)]
            if missing_fields:
                logger.error(f"Missing required fields for image: {missing_fields}")
                return None

            # Filter valid EXIF data
            valid_exif_keys = {'camera_make', 'camera_model', 'focal_length', 'exposure_time', 'f_number', 'iso'}
            filtered_exif = {k: v for k, v in exif_data.items() if k in valid_exif_keys and v is not None}

            async with self.session_factory() as session:
                # Check if image already exists
                existing_image = await self.get_image_by_filename(image_data['filename'])
                
                if existing_image:
                    # Update existing image instead of creating new one
                    for key, value in image_data.items():
                        if key not in ['filename', 'faces', 'objects', 'text_recognition', 'scene_classification'] and hasattr(existing_image, key):
                            setattr(existing_image, key, value)
                    
                    # Update or create EXIF metadata
                    if filtered_exif:
                        if existing_image.exif_metadata:
                            for key, value in filtered_exif.items():
                                if value is not None:
                                    setattr(existing_image.exif_metadata, key, value)
                        else:
                            exif = ExifMetadata(
                                image_id=existing_image.id,
                                **filtered_exif
                            )
                            session.add(exif)
                    
                    # Delete existing analysis data
                    await session.execute(delete(FaceDetection).where(FaceDetection.image_id == existing_image.id))
                    await session.execute(delete(ObjectDetection).where(ObjectDetection.image_id == existing_image.id))
                    await session.execute(delete(TextDetection).where(TextDetection.image_id == existing_image.id))
                    await session.execute(delete(SceneClassification).where(SceneClassification.image_id == existing_image.id))
                    
                    image = existing_image
                else:
                    # Create new image record with required fields
                    # Filter out analysis data that goes into separate tables
                    # Also filter out EXIF data that should go into ExifMetadata
                    valid_image_fields = {
                        'filename', 'dimensions', 'format', 'file_size', 'date_taken',
                        'latitude', 'longitude', 'embedding'
                    }
                    base_image_data = {k: v for k, v in image_data.items() 
                                      if k in valid_image_fields}
                    
                    image = Image(**base_image_data)
                    session.add(image)
                    await session.flush()  # Get the image ID

                    # Create EXIF metadata
                    if filtered_exif:
                        exif = ExifMetadata(
                            image_id=image.id,
                            **filtered_exif
                        )
                        session.add(exif)

                # Save face detections if any
                if image_data.get('faces'):
                    for face in image_data['faces']:
                        if not face:  # Skip None or empty faces
                            continue
                        try:
                            face_detection = FaceDetection(
                                image_id=image.id,
                                embedding=face.get('embedding'),
                                bounding_box=face.get('bbox'),
                                landmarks=face.get('landmarks'),
                                similarity_score=face.get('similarity_score', 0.0)
                            )
                            session.add(face_detection)
                        except Exception as e:
                            logger.warning(f"Failed to save face detection: {e}")
                            continue

                # Save object detections if any
                if image_data.get('objects'):
                    for obj in image_data['objects']:
                        if not obj:  # Skip None or empty objects
                            continue
                        try:
                            object_detection = ObjectDetection(
                                image_id=image.id,
                                label=obj
                            )
                            session.add(object_detection)
                        except Exception as e:
                            logger.warning(f"Failed to save object detection: {e}")
                            continue

                # Save text detections if any
                if image_data.get('text_recognition') and image_data['text_recognition'].get('text_blocks'):
                    for text_block in image_data['text_recognition']['text_blocks']:
                        if not text_block:  # Skip None or empty text blocks
                            continue
                        try:
                            text_detection = TextDetection(
                                image_id=image.id,
                                text=text_block.get('text'),
                                confidence=text_block.get('confidence', 0.0),
                                bounding_box=text_block.get('bbox')
                            )
                            session.add(text_detection)
                        except Exception as e:
                            logger.warning(f"Failed to save text detection: {e}")
                            continue

                # Save scene classification if any
                if image_data.get('scene_classification'):
                    scene = image_data['scene_classification']
                    if scene:
                        try:
                            scene_classification = SceneClassification(
                                image_id=image.id,
                                scene_type=scene.get('scene_type'),
                                confidence=scene.get('confidence', 0.0)
                            )
                            session.add(scene_classification)
                        except Exception as e:
                            logger.warning(f"Failed to save scene classification: {e}")

                # Commit all changes
                await session.commit()
                return image

        except Exception as e:
            logger.exception(f"Error saving image analysis: {str(e)}")
            print("❌ ERROR saving image analysis!")
            print("Image Data:", image_data)
            print("EXIF Data:", exif_data)
            import traceback
            traceback.print_exc()
            return None

    async def get_image_analysis(self, image_id: int) -> Optional[Dict[str, Any]]:
        """Get complete image analysis results."""
        try:
            async with self.session_factory() as session:
                async with session.begin():
                    # Load image with EXIF metadata
                    query = select(Image).options(joinedload(Image.exif_metadata)).where(Image.id == image_id)
                    result = await session.execute(query)
                    image = result.unique().scalar_one_or_none()

                    if not image:
                        logger.info(f"No image found with ID {image_id}")
                        return None

                    # Get all related data
                    face_query = select(FaceDetection).where(FaceDetection.image_id == image_id)
                    object_query = select(ObjectDetection).where(ObjectDetection.image_id == image_id)
                    text_query = select(TextDetection).where(TextDetection.image_id == image_id)
                    scene_query = select(SceneClassification).where(SceneClassification.image_id == image_id)

                    faces = await session.execute(face_query)
                    objects = await session.execute(object_query)
                    texts = await session.execute(text_query)
                    scenes = await session.execute(scene_query)

                    # Convert results to lists for checking if empty
                    text_blocks = list(texts.scalars().all())
                    face_detections = list(faces.scalars().all())
                    object_detections = list(objects.scalars().all())
                    scene_classifications = list(scenes.scalars().all())
                    
                    logger.info(f"Retrieved analysis for image '{image.filename}': {len(face_detections)} faces, {len(object_detections)} objects, {len(text_blocks)} text blocks")

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

                    # Format face detections to match the structure expected by the frontend
                    faces_data = []
                    embeddings_data = []
                    
                    for face in face_detections:
                        # Create the face data structure
                        face_data = {
                            "bbox": face.bounding_box,
                            "score": face.confidence,  # Renamed from confidence to score for frontend consistency
                            "attributes": {
                                "landmarks": face.landmarks
                            }
                        }
                        
                        # Add face to the faces array
                        faces_data.append(face_data)
                        
                        # Add embedding to the embeddings array if it exists
                        if hasattr(face, 'embedding') and face.embedding is not None:
                            embeddings_data.append(face.embedding)

                    # Format text blocks to match the frontend's expected structure
                    text_blocks_data = []
                    raw_text = ""
                    total_confidence = 0
                    
                    for text in text_blocks:
                        # Convert bounding box from array to object format
                        bbox_obj = {
                            "x_min": text.bounding_box[0],
                            "y_min": text.bounding_box[1],
                            "x_max": text.bounding_box[2],
                            "y_max": text.bounding_box[3]
                        } if len(text.bounding_box) >= 4 else {
                            "x_min": 0, "y_min": 0, "x_max": 0, "y_max": 0
                        }
                        
                        text_blocks_data.append({
                            "text": text.text,
                            "confidence": text.confidence,
                            "bbox": bbox_obj
                        })
                        
                        # Accumulate raw text and confidence
                        raw_text += text.text + " "
                        total_confidence += text.confidence
                    
                    # Calculate average confidence if there are text blocks
                    if text_blocks:
                        total_confidence /= len(text_blocks)
                    
                    # Text recognition data with all required fields
                    text_recognition_data = {
                        "text_detected": len(text_blocks) > 0,
                        "text_blocks": text_blocks_data,
                        "total_confidence": total_confidence,
                        "categories": [],  # Default empty categories
                        "raw_text": raw_text.strip(),
                        "language": "en"  # Default language
                    } if text_blocks else {
                        "text_detected": False,
                        "text_blocks": [],
                        "total_confidence": 0,
                        "categories": [],
                        "raw_text": "",
                        "language": ""
                    }

                    return {
                        "filename": image.filename,
                        "metadata": metadata,
                        "exif": exif_data,  
                        "faces": faces_data,
                        "embeddings": embeddings_data,
                        "objects": [obj.label for obj in object_detections],
                        "text_recognition": text_recognition_data,
                        "scene_classification": next(
                            (
                                {
                                    "scene_type": scene.scene_type,
                                    "confidence": scene.confidence
                                } for scene in scene_classifications
                            ),
                            None
                        )
                    }

        except Exception as e:
            logger.exception(f"Error getting image analysis: {str(e)}")
            return None

    async def save_face_detections(self, image_id: int, faces_data: List[Dict[str, Any]]) -> List[FaceDetection]:
        """Save face detection results to the database."""
        try:
            async with self.session_factory() as session:
                async with session.begin():
                    detections = []
                    for face_data in faces_data:
                        # Ensure we have a bounding box
                        if "bbox" in face_data and "bounding_box" not in face_data:
                            face_data["bounding_box"] = face_data.pop("bbox")
                        
                        if "bounding_box" not in face_data:
                            logger.error(f"Face detection missing bounding box: {face_data}")
                            continue
                        
                        # Create detection
                        face = FaceDetection(
                            image_id=image_id,
                            bounding_box=face_data["bounding_box"],
                            landmarks=face_data.get("landmarks"),
                            embedding=face_data.get("embedding"),
                            identity_id=face_data.get("identity_id"),
                            face_id=face_data.get("face_id"),
                            score=face_data.get("score", 0.0),  # Add detection score
                            similarity_score=face_data.get("similarity_score", 0.0),  # Keep similarity score
                            attributes=face_data.get("attributes", {})
                        )
                        session.add(face)
                        detections.append(face)
                    
                    await session.commit()
                    return detections
        except Exception as e:
            logger.error(f"Error saving face detections: {e}")
            return []

    async def save_object_detections(self, image_id: int, objects_data: List[Dict[str, Any]]) -> List[ObjectDetection]:
        """Save object detection results."""
        async with self.session_factory() as session:
            async with session.begin():
                objects = []
                
                for obj_data in objects_data:
                    obj = ObjectDetection(
                        image_id=image_id,
                        **obj_data
                    )
                    session.add(obj)
                    objects.append(obj)
                
                await session.commit()
                return objects

    async def save_text_detections(self, image_id: int, text_data: List[Dict[str, Any]]) -> List[TextDetection]:
        """Save text detection results."""
        async with self.session_factory() as session:
            async with session.begin():
                texts = []
                
                for text_item in text_data:
                    text = TextDetection(
                        image_id=image_id,
                        **text_item
                    )
                    session.add(text)
                    texts.append(text)
                
                await session.commit()
                return texts

    async def save_scene_classification(self, image_id: int, scene_data: Dict[str, Any]) -> SceneClassification:
        """Save scene classification result."""
        async with self.session_factory() as session:
            async with session.begin():
                scene = SceneClassification(
                    image_id=image_id,
                    **scene_data
                )
                session.add(scene)
                await session.commit()
                return scene

    async def clear_face_data(self) -> bool:
        """Clear all face detections and identities from the database."""
        try:
            async with self.session_factory() as session:
                async with session.begin():
                    # Delete all face detections first (due to foreign key constraint)
                    await session.execute(delete(FaceDetection))
                    # Then delete all face identities
                    await session.execute(delete(FaceIdentity))
                    await session.commit()
                    return True
        except Exception as e:
            logger.error(f"Error clearing face data: {e}")
            return False

    async def get_all_images(self) -> List[Image]:
        """Get all images from the database."""
        try:
            async with self.session_factory() as session:
                async with session.begin():
                    query = select(Image)
                    result = await session.execute(query)
                    return result.scalars().all()
        except Exception as e:
            logger.error(f"Error getting all images: {e}")
            return []

    async def get_face_identities(self) -> List[Dict[str, Any]]:
        """Get all face identities with their associated detections."""
        try:
            async with self.session_factory() as session:
                async with session.begin():
                    # Query all face identities
                    query = select(FaceIdentity).options(
                        joinedload(FaceIdentity.detections).joinedload(FaceDetection.image)
                    )
                    result = await session.execute(query)
                    identities = result.unique().scalars().all()

                    # Format response
                    formatted_identities = []
                    for identity in identities:
                        detections = []
                        for detection in identity.detections:
                            # Get the image filename for this detection
                            image_stem = pathlib.Path(detection.image.filename).stem
                            # Format face ID consistently with how they're saved
                            face_id = f"face_{image_stem}_{detection.id}"
                            face_path = f"image/faces/{face_id}.jpg"
                            
                            detections.append({
                                'image_id': detection.image_id,
                                'image_path': face_path,
                                'face_id': face_id,
                                'bounding_box': detection.bounding_box
                            })
                        
                        formatted_identities.append({
                            'id': identity.id,
                            'label': identity.label,
                            'reference_embedding': identity.reference_embedding,
                            'detections': detections
                        })
                    
                    return formatted_identities
        except Exception as e:
            logger.error(f"Error getting face identities: {e}")
            return []

    async def update_face_identity_label(self, identity_id: int, new_label: str) -> bool:
        """Update the label of a face identity."""
        try:
            async with self.session_factory() as session:
                async with session.begin():
                    # Get the identity
                    query = select(FaceIdentity).where(FaceIdentity.id == identity_id)
                    result = await session.execute(query)
                    identity = result.scalar_one_or_none()

                    if not identity:
                        logger.error(f"Face identity {identity_id} not found")
                        return False

                    # Update the label
                    identity.label = new_label
                    await session.commit()
                    return True
        except Exception as e:
            logger.error(f"Error updating face identity label: {e}")
            return False

    async def merge_face_identities(self, source_id: int, target_id: int) -> bool:
        """Merge two face identities, moving all detections from source to target."""
        try:
            async with self.session_factory() as session:
                async with session.begin():
                    # Get both identities
                    source = await session.get(FaceIdentity, source_id)
                    target = await session.get(FaceIdentity, target_id)

                    if not source or not target:
                        logger.error(f"One or both face identities not found: {source_id}, {target_id}")
                        return False

                    # Update all detections from source to point to target
                    await session.execute(
                        update(FaceDetection)
                        .where(FaceDetection.identity_id == source_id)
                        .values(identity_id=target_id)
                    )

                    # Delete the source identity
                    await session.delete(source)
                    await session.commit()
                    return True
        except Exception as e:
            logger.error(f"Error merging face identities: {e}")
            return False

    async def cleanup(self):
        """Close the database session."""
        pass