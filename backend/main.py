import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
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
from sqlalchemy import select, func, distinct
from database.models import Image as DBImage, TextDetection, ObjectDetection, SceneClassification
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

@app.get("/images/faces/{image_name}")
async def get_face_image(image_name: str):
    """Serve face images with proper headers."""
    image_path = FACES_DIR / image_name
    if not image_path.exists():
        logger.error(f"Face image not found: {image_path}")
        return JSONResponse(status_code=404, content={"error": f"Face image {image_name} not found"})
    mime_type, _ = mimetypes.guess_type(image_path)
    mime_type = mime_type or 'application/octet-stream'
    return FileResponse(
        path=image_path,
        media_type=mime_type,
        filename=image_name,
        headers={"Cache-Control": "public, max-age=3600", "Accept-Ranges": "bytes"}
    )

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
async def get_unique_faces():
    """Return unique faces detected across images."""
    try:
        unique_faces = analyzer.face_detector.get_unique_faces()
        return JSONResponse(content={"unique_faces": unique_faces}, media_type="application/json")
    except Exception as e:
        logger.error(f"Failed to get unique faces: {e}")
        return JSONResponse(status_code=500, content={"error": "Failed to get unique faces"})

@app.get("/health")
async def health_check():
    """Health check for server and models."""
    try:
        models_status = {
            "face_detector": analyzer.face_detector.app is not None,
            "object_detector": analyzer.object_detector.model is not None,
            "scene_classifier": analyzer.scene_classifier.model is not None,
            "text_recognizer": analyzer.text_recognizer.initialized
        }
        if not all(models_status.values()):
            failed_models = [name for name, status in models_status.items() if not status]
            return JSONResponse(status_code=500, content={
                "status": "unhealthy",
                "error": f"Failed models: {', '.join(failed_models)}",
                "models": models_status
            })
        return JSONResponse(content={"status": "healthy", "models": models_status})
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(status_code=500, content={"status": "unhealthy", "error": str(e)})

class SearchQuery(BaseModel):
    query: str

@app.post("/search-text")
async def search_text(search_query: SearchQuery):
    """Search for text in analyzed images."""
    try:
        logger.info(f"Received search query: {search_query.query}")
        images_dir = Path(APP_CONFIG["IMAGES_DIR"])
        cache_dir = Path("cache")
        cache_dir.mkdir(exist_ok=True)
        
        if not images_dir.exists():
            logger.error("Images directory not found")
            return JSONResponse(content={"success": False, "error": "Images directory not found"})
            
        # Find all image files
        image_files = []
        for ext in [".jpg", ".jpeg", ".png"]:
            image_files.extend(images_dir.glob(f"*{ext}"))
        logger.info(f"Found {len(image_files)} images to search")
        
        # Create a task for each image to process in parallel
        loop = asyncio.get_event_loop()
        tasks = []
        for image_file in image_files:
            task = loop.create_task(process_image_for_text_search(image_file, search_query.query, cache_dir))
            tasks.append(task)
        
        # Process all images in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and None results
        valid_results = [r for r in results if r is not None and not isinstance(r, Exception)]
        
        # Process and sanitize results for JSON serialization
        processed_results = []
        for result in valid_results:
            processed_matches = []
            for match in result.get('matches', []):
                bbox = None
                if match.get('bbox'):
                    try:
                        bbox = {
                            'x_min': int(float(match['bbox'].get('x_min', 0))),
                            'y_min': int(float(match['bbox'].get('y_min', 0))),
                            'x_max': int(float(match['bbox'].get('x_max', 0))),
                            'y_max': int(float(match['bbox'].get('y_max', 0)))
                        }
                        if not (bbox['x_min'] < bbox['x_max'] and bbox['y_min'] < bbox['y_max']):
                            bbox = None
                    except Exception as e:
                        logger.warning(f"Invalid bbox in {result['filename']}: {e}")
                processed_matches.append({
                    'text': str(match.get('text', '')).strip(),
                    'confidence': float(match.get('confidence', 0.0)),
                    'bbox': bbox
                })
            if processed_matches:
                processed_results.append({'filename': result.get('filename', ''), 'matches': processed_matches})

        response_data = {"success": True, "results": processed_results}
        return JSONResponse(
            content=response_data,
            headers={'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-cache'}
        )
    except Exception as e:
        logger.error(f"Error in text search: {e}")
        return JSONResponse(
            content={
                'success': False,
                'error': str(e),
                'details': {'type': type(e).__name__, 'message': str(e)}
            },
            status_code=500,
            headers={'Content-Type': 'application/json', 'Cache-Control': 'no-cache'}
        )

async def process_image_for_text_search(image_file: Path, query: str, cache_dir: Path):
    """Process a single image for text search with caching."""
    try:
        cache_file = cache_dir / f"{image_file.stem}_text.json"
        
        # Check cache first
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    analysis = json.load(f)
            except json.JSONDecodeError:
                # Cache file is corrupted, re-analyze
                analysis = None
        else:
            analysis = None
            
        # If no valid cache, analyze the image
        if analysis is None:
            analysis = {
                "filename": image_file.name,
                "text_recognition": analyzer.text_recognizer.detect_text(str(image_file))
            }
            # Save to cache
            with open(cache_file, 'w') as f:
                json.dump(analysis, f)
                
        # Check if text was detected and search for matches
        if analysis["text_recognition"].get("text_detected"):
            matches = analyzer.text_recognizer.search_text(
                query, analysis["text_recognition"].get("text_blocks", [])
            )
            if matches:
                return {"filename": image_file.name, "matches": matches}
                
        return None
    except Exception as e:
        logger.error(f"Error processing {image_file.name} for text search: {e}")
        return None

class UploadResponse(BaseModel):
    filename: str
    analysis: dict
    cached: bool = False

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)