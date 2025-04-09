import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import logging
import json
from typing import List, Dict, Any, Optional, AsyncGenerator
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from database.database_service import DatabaseService
from database.config import get_async_session, get_db
from pydantic import BaseModel
import cv2
import numpy as np
import mimetypes
from PIL import Image as PILImage
from PIL.ExifTags import TAGS
from datetime import datetime
import pytz
from fractions import Fraction
import time
from sqlalchemy import select, func, distinct, delete
from database.models import Image as DBImage, TextDetection, ObjectDetection, SceneClassification, FaceIdentity, FaceDetection
from fastapi.responses import JSONResponse
from fastapi import status
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import shutil
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',  # Just the message, no level or timestamps
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('app.log')  # File output with full details
    ]
)

# Set more detailed format for file handler only
for handler in logging.root.handlers:
    if isinstance(handler, logging.FileHandler):
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Completely silence SQLAlchemy logging for console output
logging.getLogger('sqlalchemy').setLevel(logging.CRITICAL)  # Most aggressive setting
logging.getLogger('sqlalchemy.engine').setLevel(logging.CRITICAL)
logging.getLogger('sqlalchemy.pool').setLevel(logging.CRITICAL)
logging.getLogger('sqlalchemy.dialects').setLevel(logging.CRITICAL)
logging.getLogger('sqlalchemy.orm').setLevel(logging.CRITICAL)

# Silence other verbose loggers
logging.getLogger('uvicorn').setLevel(logging.ERROR)
logging.getLogger('fastapi').setLevel(logging.ERROR)
logging.getLogger('models').setLevel(logging.WARNING)

# Create a custom filter for our application logs
class HumanReadableFilter(logging.Filter):
    def filter(self, record):
        # Block all SQLAlchemy logs
        if record.name.startswith('sqlalchemy'):
            return False
        
        # Allow only specific log messages that are human-readable summaries
        if hasattr(record, 'human_readable') and record.human_readable:
            return True
            
        # Block most other logs
        if record.levelno < logging.WARNING:
            return False
            
        return True

# Apply the filter to the console handler
for handler in logging.root.handlers:
    if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
        handler.addFilter(HumanReadableFilter())

logger = logging.getLogger(__name__)

# Helper function for human-readable logs
def log_human_readable(message):
    """Log a human-readable message that won't be filtered"""
    extra = {'human_readable': True}
    logger.info(message, extra=extra)

def convert_to_serializable(value):
    """Convert EXIF values to JSON serializable format."""
    if isinstance(value, Fraction):
        return float(value)
    elif isinstance(value, tuple):
        return tuple(convert_to_serializable(x) for x in value)
    elif isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        return float(value)
    elif isinstance(value, np.ndarray):
        return obj.tolist()
    elif isinstance(value, datetime):
        return value.isoformat() if value else None
    elif isinstance(value, bytes):
        return value.decode('utf-8', errors='ignore')
    return value

def extract_image_metadata(image_path: Path) -> Dict[str, Any]:
    """Extract metadata from an image file."""
    try:
        with PILImage.open(image_path) as img:
            # Get basic image info
            width, height = img.size
            format_type = img.format
            
            # Initialize metadata dict
            metadata = {
                "dimensions": f"{width}x{height}",
                "format": format_type.lower() if format_type else "unknown",
                "file_size": image_path.stat().st_size / 1024.0,  # Convert to KB
                "date_taken": None,
                "camera_make": None,
                "camera_model": None,
                "focal_length": None,
                "exposure_time": None,
                "f_number": None,
                "iso": None,
                "location": None
            }
            
            # Extract EXIF data if available
            if hasattr(img, '_getexif') and img._getexif():
                exif = img._getexif()
                if exif:
                    for tag_id, value in exif.items():
                        tag = TAGS.get(tag_id, tag_id)
                        
                        if tag == 'DateTimeOriginal':
                            try:
                                metadata['date_taken'] = datetime.strptime(value, '%Y:%m:%d %H:%M:%S')
                            except:
                                pass
                        elif tag == 'Make':
                            metadata['camera_make'] = value
                        elif tag == 'Model':
                            metadata['camera_model'] = value
                        elif tag == 'FocalLength':
                            if isinstance(value, tuple):
                                metadata['focal_length'] = float(value[0]) / float(value[1])
                            else:
                                metadata['focal_length'] = float(value)
                        elif tag == 'ExposureTime':
                            if isinstance(value, tuple):
                                metadata['exposure_time'] = f"{value[0]}/{value[1]}"
                            else:
                                metadata['exposure_time'] = str(value)
                        elif tag == 'FNumber':
                            if isinstance(value, tuple):
                                metadata['f_number'] = float(value[0]) / float(value[1])
                            else:
                                metadata['f_number'] = float(value)
                        elif tag == 'ISOSpeedRatings':
                            metadata['iso'] = int(value) if isinstance(value, (int, str)) else value[0]
                        
            return metadata
    except Exception as e:
        logger.error(f"Failed to extract metadata from {image_path.name}: {e}")
        return {
            "dimensions": "unknown",
            "format": image_path.suffix.lower()[1:],
            "file_size": image_path.stat().st_size / 1024.0,
            "date_taken": None,
            "camera_make": None,
            "camera_model": None,
            "focal_length": 0.0,
            "exposure_time": "",
            "f_number": 0.0,
            "iso": None,
            "location": None
        }

class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types and PIL EXIF types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Fraction):
            return float(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat() if obj else None
        elif isinstance(obj, bytes):
            return obj.decode('utf-8', errors='ignore')
        return super().default(obj)

# Initialize the analyzer only once
from models.inference import ImageAnalyzer
analyzer = ImageAnalyzer()
log_human_readable("PhotoCortex models loaded and ready")

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global configuration
APP_CONFIG = {
    "IMAGES_DIR": "./image",
    "FACES_DIR": "./image/faces",
    "API_BASE_URL": "http://localhost:8000"
}

# Ensure directories exist
IMAGES_DIR = Path(APP_CONFIG["IMAGES_DIR"]).resolve()
FACES_DIR = Path(APP_CONFIG["FACES_DIR"]).resolve()
os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

@app.get("/config")
async def get_config():
    """Get application configuration."""
    return JSONResponse(content=APP_CONFIG)

@app.get("/images/faces/{face_identifier}")
async def get_face_image(face_identifier: str):
    """Serve face images with proper headers."""
    # Try new format first (face_{id}.jpg)
    image_path = FACES_DIR / f"face_{face_identifier.split('_')[-1].split('.')[0]}.jpg"
    
    # If not found, try the exact requested filename (legacy support)
    if not image_path.exists():
        image_path = FACES_DIR / face_identifier
    
    if not image_path.exists():
        logger.error(f"Face image not found: {image_path}")
        return JSONResponse(
            status_code=404,
            content={"error": f"Face image {face_identifier} not found"}
        )
    
    return FileResponse(image_path)

@app.get("/images/{image_name}")
async def get_image(image_name: str):
    """Serve images with proper headers."""
    image_path = IMAGES_DIR / image_name
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        return JSONResponse(status_code=404, content={"error": f"Image {image_name} not found"})
    mime_type, _ = mimetypes.guess_type(image_path)
    mime_type = mime_type or 'application/octet-stream'
    return FileResponse(
        path=image_path,
        media_type=mime_type,
        filename=image_name,
        headers={"Cache-Control": "public, max-age=3600", "Accept-Ranges": "bytes"}
    )

@app.get("/image/{image_path:path}")
async def serve_image(image_path: str):
    """Serve images with proper MIME types and caching."""
    try:
        image_full_path = Path(APP_CONFIG["IMAGES_DIR"]) / image_path
        if not image_full_path.exists():
            return JSONResponse(status_code=404, content={"error": f"Image not found: {image_path}"})
        mime_type, _ = mimetypes.guess_type(str(image_full_path))
        mime_type = mime_type or 'application/octet-stream'
        return FileResponse(
            path=image_full_path,
            media_type=mime_type,
            headers={"Cache-Control": "public, max-age=3600", "Access-Control-Allow-Origin": "*"}
        )
    except Exception as e:
        logger.error(f"Error serving image {image_path}: {e}")
        return JSONResponse(status_code=500, content={"error": f"Failed to serve image: {e}"})

@app.get("/analyze-folder")
async def analyze_folder(db: AsyncSession = Depends(get_db)):
    """Analyze all images in the folder and stream results."""
    try:
        image_files = [f for f in IMAGES_DIR.glob("*") if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
        if not image_files:
            return JSONResponse(status_code=400, content={"error": "No images found in the directory"})
        
        # Create database service
        db_service = DatabaseService(lambda: db)
        return StreamingResponse(analyze_image_stream(image_files, db_service), media_type="text/event-stream")
    except Exception as e:
        logger.error(f"Failed to analyze folder: {e}")
        return JSONResponse(status_code=500, content={"error": "Failed to analyze folder"})

async def analyze_image_stream(image_files: list, db_service: DatabaseService) -> AsyncGenerator[str, None]:
    """Stream analysis results for each image."""
    total_files = len(image_files)
    results = []
    batch_size = 8
    
    for i in range(0, len(image_files), batch_size):
        batch = image_files[i:i+batch_size]
        
        for image_path in batch:
            try:
                # Check if image already exists in database
                existing_image = await db_service.get_image_by_filename(image_path.name)
                is_cached = False
                if existing_image:
                    # Get cached analysis results
                    cached_result = await db_service.get_image_analysis(existing_image.id)
                    if cached_result:
                        logger.info(f"Using cached analysis for {image_path.name}")
                        # Use the cached result directly
                        analysis_data = cached_result
                        # Include the filename in the analysis data
                        analysis_data["filename"] = image_path.name
                        # Add to results without nesting under "analysis"
                        results.append(analysis_data)
                        is_cached = True
                    else:
                        logger.warning(f"Cached image found for {image_path.name} but failed to get analysis.")
                        # Proceed to re-analyze if cache retrieval failed
                        is_cached = False
                else:
                    is_cached = False

                if not is_cached:
                    logger.info(f"Analyzing image: {image_path.name}")
                    # Get a session from the database service
                    session = await db_service.get_session()
                    try:
                        # Analyze image with session (this saves to DB and returns image_id)
                        saved_result = await analyzer.analyze_image_with_session(str(image_path), session)
                        
                        # Check if we got a valid image ID or an error dictionary
                        if isinstance(saved_result, int):
                            # Retrieve the saved analysis data in the correct format
                            analysis_data = await db_service.get_image_analysis(saved_result)
                            if analysis_data:
                                # Include the filename in the analysis data
                                analysis_data["filename"] = image_path.name
                                # Add to results without nesting under "analysis"
                                results.append(analysis_data)
                            else:
                                logger.error(f"Failed to retrieve analysis for {image_path.name} after saving (ID: {saved_result})")
                                # Handle error case, yield an error status
                                error_progress = {
                                    "progress": min(100, int((len(results) / total_files) * 100)),
                                    "current": len(results),
                                    "total": total_files,
                                    "filename": image_path.name,
                                    "error": f"Failed to retrieve analysis after saving",
                                    "complete": False
                                }
                                yield json.dumps(error_progress, cls=NumpyJSONEncoder) + "\n"
                        elif isinstance(saved_result, dict) and 'error' in saved_result:
                            # This is an error case from analyze_image_with_session
                            logger.error(f"Error analyzing image {image_path.name}: {saved_result['error']}")
                            error_progress = {
                                "progress": min(100, int((len(results) / total_files) * 100)),
                                "current": len(results),
                                "total": total_files,
                                "filename": image_path.name,
                                "error": saved_result['error'],
                                "complete": False
                            }
                            yield json.dumps(error_progress, cls=NumpyJSONEncoder) + "\n"
                        else:
                            logger.error(f"Failed to analyze and save image {image_path.name}, unexpected result: {saved_result}")
                            # Handle error case, yield an error status
                            error_progress = {
                                "progress": min(100, int((len(results) / total_files) * 100)),
                                "current": len(results),
                                "total": total_files,
                                "filename": image_path.name,
                                "error": "Unknown error during analysis",
                                "complete": False
                            }
                            yield json.dumps(error_progress, cls=NumpyJSONEncoder) + "\n"
                    except Exception as e:
                        logger.exception(f"Error during analysis or retrieval for {image_path.name}")
                        # Handle error case
                    finally:
                        await session.close() # Ensure session is closed

                # Send progress update regardless of cache status
                progress = {
                    "progress": min(100, int((len(results) / total_files) * 100)),
                    "current": len(results),
                    "total": total_files,
                    "filename": image_path.name,
                    "cached": is_cached,
                    "complete": False
                }
                yield json.dumps(progress, cls=NumpyJSONEncoder) + "\n"

            except Exception as e:
                logger.exception(f"Error processing {image_path.name}: {e}")
                # Yield error status for this specific file
                error_progress = {
                    "progress": min(100, int((len(results) / total_files) * 100)),
                    "current": len(results),
                    "total": total_files,
                    "filename": image_path.name,
                    "error": str(e),
                    "complete": False
                }
                yield json.dumps(error_progress, cls=NumpyJSONEncoder) + "\n"
                continue # Move to the next file

    # Yield final results
    final_data = {
        "progress": 100,
        "current": len(results),
        "total": total_files,
        "complete": True,
        "results": results
    }
    yield json.dumps(final_data, cls=NumpyJSONEncoder) + "\n"

@app.get("/image-analysis/{filename}")
async def get_image_analysis(filename: str, db: AsyncSession = Depends(get_db)):
    """Get cached analysis results for an image."""
    try:
        db_service = DatabaseService(lambda: db)
        image = await db_service.get_image_by_filename(filename)
        if not image:
            return JSONResponse(
                status_code=404,
                content={"error": f"No analysis found for image: {filename}"}
            )
        
        analysis = await db_service.get_image_analysis(image.id)
        if not analysis:
            return JSONResponse(
                status_code=404,
                content={"error": f"No analysis found for image: {filename}"}
            )
            
        # Return the analysis directly instead of nesting it under an "analysis" key
        # This matches what the frontend expects
        return JSONResponse(content=analysis)
        
    except Exception as e:
        logger.error(f"Error retrieving analysis for {filename}: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to retrieve analysis: {str(e)}"}
        )

@app.get("/unique-faces")
async def get_unique_faces(db: AsyncSession = Depends(get_db)):
    """Return unique faces detected across images from the database."""
    try:
        log_human_readable(f"Retrieving unique faces from database")
        
        # Query FaceIdentity and related FaceDetection records from the database
        query = select(FaceIdentity)
        result = await db.execute(query)
        face_identities = result.scalars().all()
        
        unique_faces = []
        for identity in face_identities:
            # Get all face detections for this identity
            detections_query = select(FaceDetection).where(FaceDetection.identity_id == identity.id)
            detections_result = await db.execute(detections_query)
            detections = detections_result.scalars().all()
            
            # Get unique images containing this face
            image_ids = set(detection.image_id for detection in detections)
            images_query = select(DBImage).where(DBImage.id.in_(image_ids))
            images_result = await db.execute(images_query)
            images = images_result.scalars().all()
            
            # Create a mapping of image_id to filename for quick lookup
            image_map = {img.id: img.filename for img in images}
            
            # Create face image paths based on image name and detection ID
            face_images = []
            for detection in detections:
                if detection.image_id in image_map:
                    image_filename = image_map[detection.image_id]
                    image_stem = Path(image_filename).stem
                    face_id = f"face_{image_stem}_{detection.id}"
                    face_path = f"images/faces/{face_id}.jpg"
                    face_images.append(face_path)
            
            # Format the response
            unique_faces.append({
                'id': identity.id,
                'label': identity.label,
                'images': [image_map[img_id] for img_id in image_ids if img_id in image_map],
                'face_images': face_images
            })
        
        log_human_readable(f"Found {len(unique_faces)} unique faces in the database")
        return JSONResponse(content={"unique_faces": unique_faces}, media_type="application/json")
    except Exception as e:
        logger.error(f"Failed to get unique faces from database: {e}")
        return JSONResponse(status_code=500, content={"error": f"Failed to get unique faces: {str(e)}"})

class UploadResponse(BaseModel):
    filename: str
    analysis: dict
    cached: bool = False

@app.put("/face-identity/{identity_id}")
async def update_face_identity(
    identity_id: int, 
    update_data: dict = Body(...), 
    db: AsyncSession = Depends(get_db)
):
    """Update a face identity (e.g., assign a name)."""
    try:
        log_human_readable(f"Updating face identity {identity_id} with data: {update_data}")
        
        # Query the face identity
        query = select(FaceIdentity).where(FaceIdentity.id == identity_id)
        result = await db.execute(query)
        identity = result.scalar_one_or_none()
        
        if not identity:
            log_human_readable(f"Face identity {identity_id} not found")
            return JSONResponse(
                status_code=404, 
                content={"error": "Identity not found"}
            )
        
        # Update the label if provided
        if "label" in update_data:
            identity.label = update_data["label"]
            log_human_readable(f"Updated face identity {identity_id} label to '{update_data['label']}'")
            
        await db.commit()
        
        return JSONResponse(
            content={
                "success": True, 
                "identity": {
                    "id": identity.id,
                    "label": identity.label
                }
            }
        )
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to update face identity: {e}")
        return JSONResponse(
            status_code=500, 
            content={"error": f"Failed to update face identity: {str(e)}"}
        )

@app.post("/upload", response_model=UploadResponse)
async def upload_image(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload and analyze a single image, storing results in the database.
    If the image was previously analyzed, returns cached results.
    """
    try:
        # Create database service
        db_service = DatabaseService(lambda: db)
        
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )
            
        # Generate safe filename
        filename = secure_filename(file.filename)
        file_path = IMAGES_DIR / filename
        
        # Check if file already exists and analyzed
        existing_image = await db_service.get_image_by_filename(filename)
        if existing_image:
            cached_analysis = await db_service.get_image_analysis(existing_image.id)
            if cached_analysis:
                return UploadResponse(
                    filename=filename,
                    analysis=cached_analysis,
                    cached=True
                )
        
        # Save uploaded file
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        finally:
            await file.close()
            
        # Use the new analyze_image_with_session method which handles duplicates
        try:
            # This will either analyze the image or return the ID of an existing image
            result = await analyzer.analyze_image_with_session(file_path, db)
            
            # Check if result is an error dictionary or an image ID
            if isinstance(result, dict) and "error" in result:
                logger.error(f"Error analyzing image: {result['error']}")
                analysis = {
                    "filename": filename,
                    "faces": [],
                    "embeddings": [],
                    "objects": [],
                    "text_recognition": {
                        "text_detected": False, 
                        "text_blocks": [],
                        "total_confidence": 0,
                        "categories": [],
                        "raw_text": "",
                        "language": ""
                    },
                    "scene_classification": None,
                    "metadata": {},
                    "exif": {}
                }
                return analysis
            
            # If we got here, result is an image ID
            image_id = result
            
            # Get the analysis results from the database
            analysis = await db_service.get_image_analysis(image_id)
            
            if not analysis:
                logger.error(f"Failed to retrieve analysis for image ID {image_id}")
                analysis = {
                    "faces": [],
                    "embeddings": [],
                    "objects": [],
                    "text_recognition": {"text_detected": False, "text_blocks": []},
                    "scene_classification": None
                }
            
            return UploadResponse(
                filename=filename,
                analysis=analysis,
                cached=False
            )
            
        except Exception as e:
            logger.error(f"Error analyzing image {file_path}: {str(e)}")
            analysis = {
                "faces": [],
                "embeddings": [],
                "objects": [],
                "text_recognition": {"text_detected": False, "text_blocks": []},
                "scene_classification": None
            }
            
            return UploadResponse(
                filename=filename,
                analysis=analysis,
                cached=False
            )
        
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        if 'file_path' in locals() and file_path.exists():
            file_path.unlink()  # Clean up file if analysis failed
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process image: {str(e)}"
        )

@app.post("/scan-text")
async def scan_text(
    file_data: dict,
    db: AsyncSession = Depends(get_db)
):
    """
    Scan text in an image using the TextRecognizer class.
    """
    try:
        filename = file_data.get("filename")
        if not filename:
            raise HTTPException(status_code=400, detail="Filename is required")
            
        # Use the correct image directory from APP_CONFIG
        image_path = IMAGES_DIR / filename
        
        if not image_path.exists():
            raise HTTPException(status_code=404, detail=f"Image not found: {filename}")
            
        # Initialize TextRecognizer if not already initialized
        if not hasattr(analyzer, "text_recognizer"):
            from models.inference.text_recognizer import TextRecognizer
            try:
                analyzer.text_recognizer = TextRecognizer()
            except Exception as e:
                logger.error(f"Failed to initialize TextRecognizer: {str(e)}")
                raise HTTPException(status_code=500, detail="Failed to initialize text recognition model")
            
        # Process the image
        try:
            start_time = time.time()
            result = analyzer.text_recognizer.detect_text(str(image_path))
            processing_time = time.time() - start_time
            
            # Add processing time to the result
            result['processing_time'] = processing_time
            
            return result
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to process image for text recognition")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scanning text: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during text scanning")

# @app.post("/reprocess-face-identities")
# async def reprocess_face_identities(
#     similarity_threshold: float = 0.5,
#     db: AsyncSession = Depends(get_db)
# ):
#     """
#     Reprocess all face detections in the database and update face identities.
    
#     This endpoint will:
#     1. Scan all face detections in the database
#     2. Re-cluster them into unique identities based on embedding similarity
#     3. Update the face_identities table
    
#     Args:
#         similarity_threshold: Threshold for considering two faces as the same person (0.0-1.0)
        
#     Returns:
#         Statistics about the operation
#     """
#     try:
#         log_human_readable(f"Starting face identity reprocessing with threshold {similarity_threshold}")
        
#         # Step 1: Get all face detections from the database
#         detections_query = select(FaceDetection)
#         detections_result = await db.execute(detections_query)
#         face_detections = detections_result.scalars().all()
        
#         if not face_detections:
#             return JSONResponse(
#                 content={"message": "No face detections found in database", "stats": {"processed": 0}}
#             )
        
#         log_human_readable(f"Found {len(face_detections)} face detections to process")
        
#         # Step 2: Clear existing identities to ensure a fresh start
#         # First, clear identity references from face detections
#         await db.execute(
#             FaceDetection.__table__.update().values(identity_id=None)
#         )
#         # Then delete all face identities
#         await db.execute(delete(FaceIdentity))
#         await db.commit()
#         log_human_readable("Cleared existing face identities")
        
#         # Step 3: Group face detections by embedding similarity
#         identities = []  # List of (identity_id, reference_embedding) tuples
#         identity_map = {}  # Maps detection.id to identity_id
#         stats = {
#             "total_detections": len(face_detections),
#             "new_identities": 0,
#             "assigned_detections": 0,
#             "unassigned_detections": 0
#         }
        
#         # Initialize face detector for similarity calculations
#         from models.inference.face_detector import FaceDetector
#         face_detector = FaceDetector()
        
#         # Process each detection
#         for detection in face_detections:
#             embedding = np.array(detection.embedding, dtype=np.float32)
#             embedding /= (np.linalg.norm(embedding) + 1e-6)
            
#             # Find best matching identity
#             best_match = None
#             best_similarity = 0
            
#             for identity_id, reference_embedding in identities:
#                 reference_embedding = np.array(reference_embedding, dtype=np.float32)
#                 similarity = face_detector._compute_similarity(embedding, reference_embedding)
                
#                 log_human_readable(f"Comparing face {detection.id} with identity {identity_id}: similarity={similarity:.3f}")
                
#                 if similarity > similarity_threshold and similarity > best_similarity:
#                     best_similarity = similarity
#                     best_match = identity_id
            
#             if best_match is not None:
#                 # Assign to existing identity
#                 identity_map[detection.id] = best_match
#                 stats["assigned_detections"] += 1
#                 log_human_readable(f"Assigned face {detection.id} to existing identity {best_match} with similarity {best_similarity:.3f}")
#             else:
#                 # Create new identity
#                 new_identity_id = len(identities)
#                 identities.append((new_identity_id, embedding.tolist()))
#                 identity_map[detection.id] = new_identity_id
#                 stats["new_identities"] += 1
#                 stats["assigned_detections"] += 1
#                 log_human_readable(f"Created new identity {new_identity_id} for face {detection.id}")
        
#         # Step 4: Create face identities in the database
#         db_identities = {}  # Maps our identity_id to database identity
        
#         for identity_id, reference_embedding in identities:
#             # Create a new identity in the database
#             new_identity = FaceIdentity(
#                 label=f"Person_{identity_id}",
#                 reference_embedding=reference_embedding
#             )
#             db.add(new_identity)
#             await db.flush()
#             db_identities[identity_id] = new_identity
        
#         # Step 5: Update face detections with identity references
#         for detection in face_detections:
#             if detection.id in identity_map:
#                 local_identity_id = identity_map[detection.id]
#                 db_identity = db_identities[local_identity_id]
#                 detection.identity_id = db_identity.id
#             else:
#                 stats["unassigned_detections"] += 1
        
#         # Commit all changes
#         await db.commit()
        
#         log_human_readable(f"Face identity reprocessing complete. Created {stats['new_identities']} identities.")
        
#         return JSONResponse(
#             content={
#                 "success": True,
#                 "message": "Face identities reprocessed successfully",
#                 "stats": stats
#             }
#         )
#     except Exception as e:
#         await db.rollback()
#         logger.error(f"Failed to reprocess face identities: {e}")
#         return JSONResponse(
#             status_code=500,
#             content={"error": f"Failed to reprocess face identities: {str(e)}"}
#         )

@app.post("/reprocess-face-identities")
async def reprocess_face_identities(
    similarity_threshold: float = 0.5,
    db: AsyncSession = Depends(get_db)
):
    """
    Reprocess all face detections in the database and update face identities.
    """
    try:
        log_human_readable(f"Starting face identity reprocessing with threshold {similarity_threshold}")

        # Step 1: Get all face detections
        detections_query = select(FaceDetection)
        detections_result = await db.execute(detections_query)
        face_detections = detections_result.scalars().all()

        if not face_detections:
            return JSONResponse(content={"message": "No face detections found in database", "stats": {"processed": 0}})

        log_human_readable(f"Found {len(face_detections)} face detections to process")

        # Step 2: Clear existing identities
        await db.execute(FaceDetection.__table__.update().values(identity_id=None))
        await db.execute(delete(FaceIdentity))
        await db.commit()

        log_human_readable("Cleared existing face identities")

        # Step 3: Group by similarity
        identities = []  # [(identity_id, normalized_embedding)]
        identity_map = {}
        stats = {
            "total_detections": len(face_detections),
            "new_identities": 0,
            "assigned_detections": 0,
            "unassigned_detections": 0
        }

        from models.inference.face_detector import FaceDetector
        face_detector = FaceDetector()
        identity_counter = 0

        for detection in face_detections:
            try:
                embedding = np.array(detection.embedding, dtype=np.float32)
                embedding /= (np.linalg.norm(embedding) + 1e-6)

                best_match = None
                best_similarity = 0.0

                for identity_id, ref_embedding in identities:
                    ref_embedding = np.array(ref_embedding, dtype=np.float32)
                    ref_embedding /= (np.linalg.norm(ref_embedding) + 1e-6)

                    similarity = face_detector._compute_similarity(embedding, ref_embedding)
                    if similarity > similarity_threshold and similarity > best_similarity:
                        best_similarity = similarity
                        best_match = identity_id

                if best_match is not None:
                    identity_map[detection.id] = best_match
                    stats["assigned_detections"] += 1
                    log_human_readable(f"Assigned detection {detection.id} to identity {best_match} (sim: {best_similarity:.3f})")
                else:
                    # Create new identity
                    identity_id = identity_counter
                    identity_counter += 1
                    identities.append((identity_id, embedding.tolist()))
                    identity_map[detection.id] = identity_id
                    stats["new_identities"] += 1
                    stats["assigned_detections"] += 1
                    log_human_readable(f"Created new identity {identity_id} for detection {detection.id}")

            except Exception as e:
                logger.error(f"Failed to process detection {detection.id}: {e}")
                stats["unassigned_detections"] += 1

        # Step 4: Insert FaceIdentity records
        db_identity_map = {}
        for identity_id, ref_embedding in identities:
            identity = FaceIdentity(
                label=f"Person_{identity_id}",
                reference_embedding=ref_embedding
            )
            db.add(identity)
            await db.flush()
            db_identity_map[identity_id] = identity

        # Step 5: Assign identities to detections
        for detection in face_detections:
            if detection.id in identity_map:
                identity_id = identity_map[detection.id]
                detection.identity_id = db_identity_map[identity_id].id
            else:
                stats["unassigned_detections"] += 1

        await db.commit()

        log_human_readable(f"Face reprocessing complete. {stats['new_identities']} new identities created.")

        return JSONResponse(
            content={
                "success": True,
                "message": "Face identities reprocessed successfully",
                "stats": stats
            }
        )

    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to reprocess face identities: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to reprocess face identities: {str(e)}"}
        )

@app.post("/recut-face-images")
async def recut_face_images(db: AsyncSession = Depends(get_async_session)):
    """
    Reprocess all face detections and regenerate cropped face images
    """
    try:
        # Query all face detections with their source images
        result = await db.execute(
            select(FaceDetection, DBImage)
            .join(DBImage, FaceDetection.image_id == DBImage.id)
        )
        detections = result.all()
        
        success = 0
        failed = 0
        
        for detection, image in detections:
            try:
                # Build paths
                src_path = os.path.join(APP_CONFIG["IMAGES_DIR"], image.filename)
                face_filename = f"face_{detection.id}.jpg"
                dest_path = os.path.join(APP_CONFIG["FACES_DIR"], face_filename)
                
                # Verify source image exists and is readable
                if not os.path.exists(src_path):
                    raise FileNotFoundError(f"Source image not found: {src_path}")
                
                # Load original image with better error handling
                try:
                    img = cv2.imread(src_path)
                    if img is None:
                        raise ValueError(f"Could not load image {src_path} - may be corrupt or unsupported format")
                except Exception as load_error:
                    raise ValueError(f"Failed to load image {src_path}: {str(load_error)}")
                
                # Get original coordinates
                x1, y1, x2, y2 = detection.bounding_box
                h, w = img.shape[:2]
                
                # Convert if normalized (0-1 range)
                if all(0 <= coord <= 1 for coord in [x1, y1, x2, y2]):
                    x1, x2 = int(x1 * w), int(x2 * w)
                    y1, y2 = int(y1 * h), int(y2 * h)

                # Apply padding safely
                x1 = max(0, x1 - 10)
                y1 = max(0, y1 - 10)
                x2 = min(w, x2 + 10)
                y2 = min(h, y2 + 10)

                # Final validation
                if x1 >= x2 or y1 >= y2:
                    raise ValueError(f"Invalid box after padding: {x1},{y1}-{x2},{y2} (image size {w}x{h})")

                # Crop and save face
                face_img = img[y1:y2, x1:x2]
                if not cv2.imwrite(dest_path, face_img):
                    raise RuntimeError(f"Failed to save face image to {dest_path}")
                success += 1
                
            except Exception as e:
                logger.error(f"Failed to recut face {detection.id}: {str(e)}")
                failed += 1
        
        return {
            "success": success, 
            "failed": failed,
            "faces_dir": os.path.abspath(APP_CONFIG["FACES_DIR"])
        }
        
    except Exception as e:
        logger.error(f"Recutting failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)