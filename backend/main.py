import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import cv2
import numpy as np
import json
import asyncio
import logging
import mimetypes
from PIL import Image as PILImage
from PIL.ExifTags import TAGS
from datetime import datetime
import pytz
from fractions import Fraction
from typing import AsyncGenerator, List, Optional, Dict, Any
import time
from sqlalchemy.ext.asyncio import AsyncSession
from database.config import get_async_session, get_db
from sqlalchemy import select, func, distinct
from database.models import Image as DBImage, TextDetection, ObjectDetection, SceneClassification

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

# Set higher log level for most modules to reduce noise
logging.getLogger('models.inference').setLevel(logging.WARNING)
logging.getLogger('uvicorn').setLevel(logging.WARNING)
logging.getLogger('fastapi').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

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
logger.info("PhotoCortex models loaded and ready")

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

async def analyze_image_stream(image_files: list) -> AsyncGenerator[str, None]:
    """Stream analysis results for each image."""
    total_files = len(image_files)
    results = []
    # Process images in larger batches for better performance
    batch_size = 8  # Increased batch size
    
    for i in range(0, len(image_files), batch_size):
        batch = image_files[i:i+batch_size]
        batch_tasks = []
        
        for image_path in batch:
            try:
                # Extract metadata first to check for GPS data
                metadata = await asyncio.get_event_loop().run_in_executor(None, extract_image_metadata, image_path)
                
                # Load the image only once
                image = cv2.imread(str(image_path))
                if image is None or image.size == 0 or len(image.shape) != 3:
                    continue
                if image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                
                # Run face and object detection in parallel, but text recognition separately
                # since it's the most resource-intensive
                loop = asyncio.get_event_loop()
                tasks = asyncio.gather(
                    loop.run_in_executor(None, analyzer.face_detector.detect_faces, image, image_path.name),
                    loop.run_in_executor(None, analyzer.object_detector.detect_objects, image),
                    loop.run_in_executor(None, analyzer.scene_classifier.predict_scene, image)
                )
                
                batch_tasks.append((image_path, tasks, metadata))
            except Exception as e:
                logger.error(f"Error setting up tasks for {image_path.name}: {e}")
                continue
        
        # Process all images in the batch concurrently
        for idx, (image_path, tasks, metadata) in enumerate(batch_tasks):
            try:
                face_results, object_results, scene_results = await tasks
                
                # Run text recognition separately to avoid memory issues
                text_results = await asyncio.get_event_loop().run_in_executor(
                    None, analyzer.text_recognizer.detect_text, str(image_path)
                )
                
                result = {
                    "filename": image_path.name,
                    "faces": face_results.get("faces", []),
                    "objects": sorted(list(set(obj["class"] for obj in object_results))) if object_results else [],
                    "scene_classification": {
                        "scene_type": scene_results.get("scene_type", "unknown"),
                        "confidence": float(scene_results.get("confidence", 0.0)),
                        "all_scene_scores": {k: float(v) for k, v in scene_results.get("all_scene_scores", {}).items()}
                    } if scene_results else {"scene_type": "unknown", "confidence": 0.0, "all_scene_scores": {}},
                    "text_recognition": {
                        "text_detected": text_results.get("text_detected", False),
                        "text_blocks": text_results.get("text_blocks", []),
                        "total_confidence": float(text_results.get("total_confidence", 0.0)),
                        "categories": text_results.get("categories", []),
                        "raw_text": text_results.get("raw_text", ""),
                        "language": text_results.get("language", "unknown")
                    } if text_results else {
                        "text_detected": False,
                        "text_blocks": [],
                        "total_confidence": 0.0,
                        "categories": [],
                        "raw_text": "",
                        "language": "unknown"
                    },
                    "metadata": metadata or {}
                }

                # Convert any numpy values to Python native types
                result = json.loads(json.dumps(result, cls=NumpyJSONEncoder))
                results.append(result)
                
                current_count = i + idx + 1
                progress = {
                    "progress": min(100, (current_count / total_files) * 100),
                    "current": current_count,
                    "total": total_files,
                    "latest_result": result,
                    "complete": False
                }
                
                # Ensure proper JSON formatting with newline
                yield json.dumps(progress, cls=NumpyJSONEncoder) + "\n"
                
            except Exception as e:
                logger.error(f"Error processing {image_path.name}: {e}")
                continue
    
    final_update = {
        "progress": 100,
        "current": total_files,
        "total": total_files,
        "results": results,
        "complete": True
    }
    
    # Ensure proper JSON formatting with newline
    yield json.dumps(final_update, cls=NumpyJSONEncoder) + "\n"

@app.get("/analyze-folder")
async def analyze_folder():
    """Analyze all images in the folder and stream results."""
    try:
        image_files = [f for f in IMAGES_DIR.glob("*") if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
        if not image_files:
            return JSONResponse(status_code=400, content={"error": "No images found in the directory"})
        return StreamingResponse(analyze_image_stream(image_files), media_type="text/event-stream")
    except Exception as e:
        logger.error(f"Failed to analyze folder: {e}")
        return JSONResponse(status_code=500, content={"error": "Failed to analyze folder"})

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

@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_async_session)
):
    """Upload and analyze a single image."""
    try:
        # Save the uploaded file
        file_path = IMAGES_DIR / file.filename
        
        # Check if image already exists in database
        query = select(DBImage).where(DBImage.filename == file.filename)
        result = await db.execute(query)
        existing_image = result.scalar_one_or_none()
        
        if existing_image:
            # Image exists, return cached analysis
            try:
                query = select(
                    DBImage,
                    func.coalesce(func.array_agg(distinct(TextDetection.text)).filter(TextDetection.text != None), []).label('texts'),
                    func.coalesce(func.array_agg(distinct(ObjectDetection.label)).filter(ObjectDetection.label != None), []).label('objects'),
                    func.coalesce(func.array_agg(distinct(SceneClassification.scene_type)).filter(SceneClassification.scene_type != None), []).label('scenes')
                ).outerjoin(DBImage.text_detections)\
                 .outerjoin(DBImage.object_detections)\
                 .outerjoin(DBImage.scene_classifications)\
                 .where(DBImage.id == existing_image.id)\
                 .group_by(DBImage.id)
                
                result = await db.execute(query)
                image_data = result.first()
                await db.commit()  # Commit after successful read
                
                if image_data:
                    logger.info(f"Found existing image: {file.filename}")
                    return JSONResponse(content={
                        "success": True,
                        "exists": True,
                        "result": {
                            "filename": image_data.DBImage.filename,
                            "metadata": {
                                "date_taken": image_data.DBImage.date_taken,
                                "camera_make": image_data.DBImage.camera_make,
                                "camera_model": image_data.DBImage.camera_model,
                                "focal_length": image_data.DBImage.focal_length,
                                "exposure_time": image_data.DBImage.exposure_time,
                                "f_number": image_data.DBImage.f_number,
                                "iso": image_data.DBImage.iso,
                                "dimensions": image_data.DBImage.dimensions,
                                "format": image_data.DBImage.format,
                                "file_size": image_data.DBImage.file_size
                            },
                            "text_recognition": {
                                "text_detected": bool(image_data.texts and image_data.texts[0]),
                                "text_blocks": image_data.texts if image_data.texts and image_data.texts[0] else []
                            },
                            "object_detection": image_data.objects if image_data.objects and image_data.objects[0] else [],
                            "scene_classification": {
                                "scene_type": image_data.scenes[0] if image_data.scenes and image_data.scenes[0] else "Unknown",
                                "confidence": 1.0  # Using cached result
                            }
                        }
                    })
            except Exception as query_error:
                await db.rollback()
                logger.error(f"Error querying existing image: {query_error}")
                raise

        # Image doesn't exist, save it
        content = await file.read()
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        # Analyze the image and save results to database
        result = await analyzer.analyze_image_with_session(file_path, db)
        
        if "error" in result:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": result["error"]}
            )
        
        # Convert numpy types to Python types before JSON serialization
        result = json.loads(json.dumps(result, cls=NumpyJSONEncoder))
        await db.commit()  # Commit the transaction
        
        return JSONResponse(
            content={
                "success": True,
                "exists": False,
                "result": result
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
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

# Custom filter to remove unnecessary log messages
class LogFilter(logging.Filter):
    def filter(self, record):
        # Allow specific initialization messages we want to see
        if record.name == "__main__" and "ready" in record.getMessage().lower():
            return True
        # Filter out most other messages
        if record.levelno == logging.INFO:
            return False
        return True

# Apply filter to console handler only
for handler in logging.root.handlers:
    if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
        handler.addFilter(LogFilter())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)