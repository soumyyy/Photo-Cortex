import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import json
import logging
import mimetypes
import cv2
import numpy as np
import asyncio
from PIL import Image
from PIL.ExifTags import TAGS
from PIL.TiffImagePlugin import IFDRational
from fractions import Fraction
from typing import AsyncGenerator, List, Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('app.log')]
)
logger = logging.getLogger(__name__)

# Prepare EXIF tag mapping
TAGS = TAGS

def convert_to_serializable(value):
    """Convert EXIF values to JSON serializable format."""
    if isinstance(value, Fraction):
        return float(value)
    elif isinstance(value, tuple):
        return tuple(convert_to_serializable(x) for x in value)
    elif isinstance(value, bytes):
        return value.decode('utf-8', errors='ignore')
    return value

def extract_metadata(image_path: Path) -> dict:
    """Extract EXIF metadata from an image file."""
    try:
        with Image.open(image_path) as img:
            try:
                exif_info = img._getexif()
                if exif_info:
                    # Convert all EXIF values to serializable format
                    exif_info = {k: convert_to_serializable(v) for k, v in exif_info.items()}
                    
                    for tag, value in exif_info.items():
                        decoded = TAGS.get(tag, tag)
                        if decoded == 'GPSInfo':
                            # logger.info(f"Found raw GPS data in {image_path.name}: {value}")
                            try:
                                # Extract GPS coordinates
                                lat_dms = value[2]  # Latitude in degrees, minutes, seconds
                                lon_dms = value[4]  # Longitude in degrees, minutes, seconds
                                lat_ref = value[1]  # 'N' or 'S'
                                lon_ref = value[3]  # 'E' or 'W'

                                # Convert to decimal degrees
                                lat_deg = float(lat_dms[0]) + float(lat_dms[1])/60 + float(lat_dms[2])/3600
                                lon_deg = float(lon_dms[0]) + float(lon_dms[1])/60 + float(lon_dms[2])/3600

                                # Apply hemisphere reference
                                if lat_ref == 'S': lat_deg = -lat_deg
                                if lon_ref == 'W': lon_deg = -lon_deg

                                # logger.info(f"Converted GPS coordinates for {image_path.name}: {lat_deg}, {lon_deg}")
                                
                                # Create basic metadata with GPS
                                return {
                                    "date_taken": str(exif_info.get(36867)),  # DateTimeOriginal
                                    "camera_make": str(exif_info.get(271)),   # Make
                                    "camera_model": str(exif_info.get(272)),  # Model
                                    "focal_length": convert_to_serializable(exif_info.get(37386)),
                                    "exposure_time": convert_to_serializable(exif_info.get(33434)),
                                    "f_number": convert_to_serializable(exif_info.get(33437)),
                                    "iso": convert_to_serializable(exif_info.get(34855)),
                                    "gps": {
                                        "latitude": lat_deg,
                                        "longitude": lon_deg
                                    },
                                    "dimensions": f"{img.width}x{img.height}",
                                    "format": img.format,
                                    "file_size": f"{image_path.stat().st_size / 1024:.1f} KB"
                                }
                            except Exception as e:
                                logger.error(f"Error converting GPS data for {image_path.name}: {e}")
            except Exception as e:
                logger.warning(f"Error reading EXIF: {e}")

            # If we get here, either no GPS data or error occurred
            # Return basic metadata without GPS
            return {
                "date_taken": None,
                "camera_make": None,
                "camera_model": None,
                "focal_length": None,
                "exposure_time": None,
                "f_number": None,
                "iso": None,
                "gps": None,
                "dimensions": f"{img.width}x{img.height}",
                "format": img.format,
                "file_size": f"{image_path.stat().st_size / 1024:.1f} KB"
            }
    except Exception as e:
        logger.error(f"Failed to extract metadata from {image_path.name}: {str(e)}")
        return {}

class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types and PIL EXIF types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (IFDRational, Fraction)):
            return float(obj)
        elif isinstance(obj, bytes):
            return obj.decode('utf-8', errors='ignore')
        elif isinstance(obj, Path):
            return str(obj)
        return super().default(obj)

# Initialize the analyzer only once
from models.inference import ImageAnalyzer
analyzer = ImageAnalyzer()

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
                metadata = await asyncio.get_event_loop().run_in_executor(None, extract_metadata, image_path)
                
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
                    "objects": [obj["class"] for obj in object_results] if object_results else [],
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

@app.get("/analyze_face/{image_name}")
async def analyze_face(image_name: str):
    """Analyze facial attributes in an image."""
    try:
        # Use consistent directory paths with other handlers
        image_path = Path(APP_CONFIG["IMAGES_DIR"]) / image_name
        
        if not image_path.exists():
            return JSONResponse({"error": "Image not found"}, status_code=404)
            
        # Get cached result if available
        cache_key = f"face_analysis_{image_name}"
        
        # Try to use cached result
        if hasattr(analyzer, "analysis_cache") and cache_key in analyzer.analysis_cache:
            return JSONResponse(content=analyzer.analysis_cache[cache_key], encoder=NumpyJSONEncoder)
        
        # Analyze the image
        image = cv2.imread(str(image_path))
        if image is None:
            return JSONResponse({"error": "Could not read image"}, status_code=400)
            
        results = analyzer.face_detector.detect_faces(image, image_name, analyze_attributes=True)
        
        if "error" in results:
            return JSONResponse(content=results, status_code=400)
            
        # Store result in cache
        if not hasattr(analyzer, "analysis_cache"):
            analyzer.analysis_cache = {}
        analyzer.analysis_cache[cache_key] = results
        
        return JSONResponse(content=results, encoder=NumpyJSONEncoder)
    except Exception as e:
        logger.error(f"Error analyzing face: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)