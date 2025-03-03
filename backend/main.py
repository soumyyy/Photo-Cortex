from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from models.inference import ImageAnalyzer
from pathlib import Path
import logging
import cv2
import numpy as np
import json
import asyncio
from typing import AsyncGenerator
from PIL import Image
from PIL.ExifTags import TAGS
import datetime
from fractions import Fraction

# Configure logging at the start of the file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

def extract_metadata(image_path: Path) -> dict:
    """Extract EXIF metadata from image."""
    try:
        with Image.open(image_path) as img:
            # Get EXIF data
            exif = img.getexif()
            if not exif:
                return {
                    "date_taken": None,
                    "camera_make": None,
                    "camera_model": None,
                    "focal_length": None,
                    "exposure_time": None,
                    "f_number": None,
                    "iso": None,
                    "dimensions": f"{img.width}x{img.height}",
                    "format": img.format,
                    "file_size": f"{image_path.stat().st_size / 1024:.1f} KB",
                    "gps": None
                }

            metadata = {}
            for tag_id in exif:
                tag = TAGS.get(tag_id, tag_id)
                value = exif.get(tag_id)
                # Convert fractions to float where needed
                if isinstance(value, tuple) and all(isinstance(x, Fraction) for x in value):
                    value = tuple(float(x) for x in value)
                elif isinstance(value, Fraction):
                    value = float(value)
                metadata[tag] = value

            # Get GPS data if available
            gps_info = None
            if hasattr(img, '_getexif') and img._getexif() is not None:
                exif = img._getexif()
                if 34853 in exif:  # GPSInfo tag
                    gps_data = exif[34853]
                    try:
                        lat = gps_data[2]
                        lon = gps_data[4]
                        
                        # Convert fractions to float
                        lat_dec = float(lat[0]) + float(lat[1])/60 + float(lat[2])/3600
                        lon_dec = float(lon[0]) + float(lon[1])/60 + float(lon[2])/3600
                        
                        # Apply N/S and E/W
                        if gps_data[1] == 'S': lat_dec = -lat_dec
                        if gps_data[3] == 'W': lon_dec = -lon_dec
                        
                        gps_info = {
                            "latitude": lat_dec,
                            "longitude": lon_dec
                        }
                    except (KeyError, IndexError, TypeError, ZeroDivisionError):
                        pass

            # Extract common EXIF fields
            return {
                "date_taken": metadata.get("DateTime"),
                "camera_make": metadata.get("Make"),
                "camera_model": metadata.get("Model"),
                "focal_length": metadata.get("FocalLength"),
                "exposure_time": metadata.get("ExposureTime"),
                "f_number": metadata.get("FNumber"),
                "iso": metadata.get("ISOSpeedRatings"),
                "dimensions": f"{img.width}x{img.height}",
                "format": img.format,
                "file_size": f"{image_path.stat().st_size / 1024:.1f} KB",
                "gps": gps_info
            }
    except Exception as e:
        logger.error(f"Failed to extract metadata from {image_path.name}: {str(e)}")
        return {}

class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return [self.default(x) for x in obj.tolist()]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            if not np.isfinite(obj):
                return 0.0
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

# Initialize analyzer
analyzer = ImageAnalyzer()

app = FastAPI()

# Configure static file serving
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import mimetypes

# Ensure directories exist
import os
os.makedirs("images/faces", exist_ok=True)

# Configure static file serving
@app.get("/images/{image_path:path}")
async def serve_image(image_path: str):
    """Serve images with proper headers and MIME types."""
    try:
        image_full_path = Path("images") / image_path
        if not image_full_path.exists():
            return JSONResponse(
                status_code=404,
                content={"error": f"Image not found: {image_path}"}
            )

        # Determine MIME type
        mime_type, _ = mimetypes.guess_type(str(image_full_path))
        if not mime_type:
            mime_type = 'application/octet-stream'

        return FileResponse(
            path=image_full_path,
            media_type=mime_type,
            headers={
                "Cache-Control": "public, max-age=3600",
                "Access-Control-Allow-Origin": "*"
            }
        )
    except Exception as e:
        logger.error(f"Error serving image {image_path}: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to serve image: {str(e)}"}
        )

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Initialize analyzer
analyzer = ImageAnalyzer()

# Default images directory
IMAGES_DIR = Path("./images")
IMAGES_DIR.mkdir(exist_ok=True)

from starlette.responses import FileResponse
from starlette.background import BackgroundTask
import mimetypes

# Configure common image mime types
mimetypes.init()
mimetypes.add_type('image/jpeg', '.jpeg')
mimetypes.add_type('image/jpeg', '.jpg')
mimetypes.add_type('image/png', '.png')

@app.get("/images/{image_name}")
async def get_image(image_name: str):
    """Serve images with proper headers and caching."""
    image_path = IMAGES_DIR / image_name
    
    if not image_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": f"Image {image_name} not found"}
        )
    
    # Get mime type
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = 'application/octet-stream'
    
    # Use FileResponse for efficient file handling
    return FileResponse(
        path=image_path,
        media_type=mime_type,
        filename=image_name,
        headers={
            "Cache-Control": "public, max-age=3600",
            "Accept-Ranges": "bytes"
        }
    )

async def analyze_image_stream(image_files: list) -> AsyncGenerator[str, None]:
    """Stream analysis results as they're processed."""
    total_files = len(image_files)
    results = []
    
    try:
        for i, image_path in enumerate(image_files):
            try:
                # Read image with error handling
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                
                # Basic image validation
                if image.size == 0 or len(image.shape) != 3:
                    continue
                    
                # Convert RGBA to RGB if needed
                if image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                
                try:
                    face_results = await asyncio.get_event_loop().run_in_executor(
                        None, analyzer.face_detector.detect_faces, image, image_path.name
                    )
                    
                    object_results = await asyncio.get_event_loop().run_in_executor(
                        None, analyzer.object_detector.detect_objects, image
                    )
                    
                    scene_results = await asyncio.get_event_loop().run_in_executor(
                        None, analyzer.scene_classifier.predict_scene, image
                    )
                    
                    # Perform text recognition
                    text_results = await asyncio.get_event_loop().run_in_executor(
                        None, analyzer.text_recognizer.detect_text, str(image_path)
                    )
                    
                    metadata = await asyncio.get_event_loop().run_in_executor(
                        None, extract_metadata, image_path
                    )
                    
                except Exception as e:
                    continue

                # Store results
                result = {
                    "filename": image_path.name,
                    "faces": face_results.get("faces", []),
                    "objects": [obj["class"] for obj in object_results] if object_results else [],
                    "scene_classification": scene_results or {"scene_type": "unknown", "confidence": 0.0},
                    "text_recognition": text_results,
                    "metadata": metadata or {}
                }
                results.append(result)
                
                # Send progress update
                progress = {
                    "progress": ((i + 1) / total_files) * 100,
                    "current": i + 1,
                    "total": total_files,
                    "latest_result": result,
                    "complete": False
                }
                yield json.dumps(progress, cls=NumpyJSONEncoder) + "\n"
                
            except Exception:
                continue
            
            # Small delay between images
            await asyncio.sleep(0.1)
    
    except Exception as e:
        yield json.dumps({
            "error": "Processing failed",
            "message": str(e)
        }) + "\n"
        return
    
    # Send final results
    try:
        final_update = {
            "progress": 100,
            "current": total_files,
            "total": total_files,
            "results": results,
            "complete": True
        }
        yield json.dumps(final_update, cls=NumpyJSONEncoder) + "\n"
    except Exception:
        pass

@app.get("/analyze-folder")
async def analyze_folder():
    """Analyze all images in the images directory with progress updates."""
    try:
        image_files = [f for f in IMAGES_DIR.glob("*") if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
        total_files = len(image_files)
        
        if total_files == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "No images found in the directory"}
            )

        return StreamingResponse(
            analyze_image_stream(image_files),
            media_type="text/event-stream"
        )

    except Exception as e:
        logger.error(f"Failed to analyze folder: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to analyze folder"}
        )

@app.get("/unique-faces")
async def get_unique_faces():
    """Get list of unique faces detected across all images."""
    try:
        unique_faces = analyzer.face_detector.get_unique_faces()
        return JSONResponse(
            content={"unique_faces": unique_faces},
            media_type="application/json"
        )
    except Exception as e:
        logger.error(f"Failed to get unique faces: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to get unique faces"}
        )

@app.get("/health")
async def health_check():
    """Check if the server and models are healthy."""
    try:
        # Check if models are initialized
        models_status = {
            "face_detector": analyzer.face_detector.app is not None,
            "object_detector": analyzer.object_detector.model is not None,
            "scene_classifier": analyzer.scene_classifier.model is not None,
            "text_recognizer": analyzer.text_recognizer.initialized
        }
        
        # Check if any model failed to initialize
        all_models_healthy = all(models_status.values())
        
        if not all_models_healthy:
            failed_models = [name for name, status in models_status.items() if not status]
            return JSONResponse(
                status_code=500,
                content={
                    "status": "unhealthy",
                    "error": f"Failed models: {', '.join(failed_models)}",
                    "models": models_status
                }
            )
            
        return JSONResponse(
            content={
                "status": "healthy",
                "models": models_status
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )

from pydantic import BaseModel

class SearchQuery(BaseModel):
    query: str

@app.post("/search-text")
async def search_text(search_query: SearchQuery):
    """Search for text in analyzed images."""
    try:
        logger.info(f"Received search query: {search_query.query}")
        # Get all image files
        images_dir = Path("images")
        if not images_dir.exists():
            logger.error("Images directory not found")
            return JSONResponse({
                "success": False,
                "error": "Images directory not found"
            })
            
        image_files = []
        for ext in [".jpg", ".jpeg", ".png"]:
            image_files.extend(images_dir.glob(f"*{ext}"))
        
        logger.info(f"Found {len(image_files)} images to search")
        results = []
        
        for image_file in image_files:
            try:
                logger.debug(f"Processing image: {image_file.name}")
                # Load cached analysis if available
                cache_file = Path(f"cache/{image_file.stem}_text.json")
                analysis = None
                
                if cache_file.exists():
                    try:
                        with open(cache_file, 'r') as f:
                            analysis = json.load(f)
                    except Exception as e:
                        logger.warning(f"Failed to load cache for {image_file.name}: {e}")
                
                if not analysis:
                    # Analyze image synchronously since text recognition is CPU-bound
                    analysis = {
                        "filename": image_file.name,
                        "text_recognition": analyzer.text_recognizer.detect_text(str(image_file))
                    }
                    
                    # Cache the results
                    try:
                        cache_file.parent.mkdir(exist_ok=True)
                        with open(cache_file, 'w') as f:
                            json.dump(analysis, f)
                    except Exception as e:
                        logger.warning(f"Failed to cache analysis for {image_file.name}: {e}")
                
                if analysis["text_recognition"]["text_detected"]:
                    logger.debug(f"Text detected in {image_file.name}, searching...")
                    matches = analyzer.text_recognizer.search_text(
                        search_query.query,
                        analysis["text_recognition"]["text_blocks"]
                    )
                    if matches:
                        logger.info(f"Found {len(matches)} matches in {image_file.name}")
                        results.append({
                            "filename": image_file.name,
                            "matches": matches
                        })
            except Exception as e:
                logger.error(f"Error analyzing {image_file.name}: {str(e)}")
                continue
        
        logger.info(f"Converting {len(results)} results to JSON format")
        # Convert results to JSON-serializable format
        serializable_results = []
        for result in results:
            try:
                matches = []
                for match in result['matches']:
                    try:
                        # Ensure bbox values are valid
                        bbox = None
                        if match.get('bbox'):
                            bbox_data = match['bbox']
                            try:
                                bbox = {
                                    'x_min': max(0.0, min(1.0, float(bbox_data['x_min']))),
                                    'y_min': max(0.0, min(1.0, float(bbox_data['y_min']))),
                                    'x_max': max(0.0, min(1.0, float(bbox_data['x_max']))),
                                    'y_max': max(0.0, min(1.0, float(bbox_data['y_max'])))
                                }
                                # Validate bbox coordinates
                                if not (bbox['x_min'] < bbox['x_max'] and bbox['y_min'] < bbox['y_max']):
                                    logger.warning(f"Invalid bbox coordinates in {result['filename']}: {bbox}")
                                    bbox = None
                            except (KeyError, ValueError, TypeError) as e:
                                logger.warning(f"Invalid bbox data in {result['filename']}: {e}")
                                bbox = None
                        
                        serialized_match = {
                            'text': str(match['text']).strip(),
                            'confidence': float(match.get('confidence', 0)),
                            'bbox': bbox
                        }
                        matches.append(serialized_match)
                        logger.debug(f"Serialized match in {result['filename']}: {serialized_match}")
                    except Exception as e:
                        logger.error(f"Error serializing match in {result['filename']}: {str(e)}")
                        continue
                
                if matches:  # Only add results with valid matches
                    serializable_results.append({
                        'filename': str(result['filename']),
                        'matches': matches
                    })
                    logger.debug(f"Added result for {result['filename']} with {len(matches)} matches")
            except Exception as e:
                logger.error(f"Error serializing result for {result['filename']}: {str(e)}")
                continue

        from json import JSONEncoder
        class CustomJSONEncoder(JSONEncoder):
            def default(self, obj):
                try:
                    return super().default(obj)
                except TypeError:
                    return str(obj)

        def sanitize_dict(d):
            if not isinstance(d, dict):
                return d
            return {str(k): sanitize_dict(v) if isinstance(v, (dict, list)) else str(v) if not isinstance(v, (int, float, bool, type(None))) else v
                    for k, v in d.items()}

        def sanitize_list(lst):
            return [sanitize_dict(item) if isinstance(item, dict) else item for item in lst]

        # Process and sanitize results
        processed_results = []
        for result in serializable_results:
            try:
                processed_matches = []
                for match in result.get('matches', []):
                    try:
                        # Ensure all values are properly typed
                        bbox = None
                        if match.get('bbox'):
                            try:
                                bbox = {
                                    'x_min': int(float(match['bbox'].get('x_min', 0))),
                                    'y_min': int(float(match['bbox'].get('y_min', 0))),
                                    'x_max': int(float(match['bbox'].get('x_max', 0))),
                                    'y_max': int(float(match['bbox'].get('y_max', 0)))
                                }
                            except (ValueError, TypeError, KeyError) as e:
                                logger.warning(f'Invalid bbox data: {e}')
                                continue

                        processed_match = {
                            'text': str(match.get('text', '')).strip(),
                            'confidence': float(match.get('confidence', 0.0)),
                            'bbox': bbox
                        }
                        processed_matches.append(processed_match)
                    except Exception as e:
                        logger.warning(f'Error processing match: {e}')
                        continue

                if processed_matches:  # Only add results with valid matches
                    processed_results.append({
                        'filename': str(result.get('filename', '')),
                        'matches': processed_matches
                    })
            except Exception as e:
                logger.warning(f'Error processing result: {e}')
                continue

        # Final sanitization of the entire response
        response_data = {
            'success': True,
            'results': sanitize_list(processed_results)
        }
        
        # Use FastAPI's JSONResponse with explicit serialization
        # Validate and encode response
        try:
            # Use custom encoder for the first pass to catch any serialization issues
            json_str = json.dumps(response_data, cls=CustomJSONEncoder, ensure_ascii=True)
            
            # Verify the JSON is valid by parsing it
            parsed_data = json.loads(json_str)
            
            # Send the parsed (and validated) data
            return JSONResponse(
                content=parsed_data,  # Use parsed data to ensure it's valid JSON
                headers={
                    'Content-Type': 'application/json; charset=utf-8',
                    'Cache-Control': 'no-cache'
                }
            )
        except Exception as json_error:
            error_msg = str(json_error)
            logger.error(f'JSON serialization error: {error_msg}')
            
            # Create a safe error response
            safe_error = {
                'success': False,
                'error': 'Failed to process response',
                'details': error_msg[:200]  # Limit error message length
            }
            
            return JSONResponse(
                content=safe_error,
                status_code=500,
                headers={
                    'Content-Type': 'application/json; charset=utf-8',
                    'Cache-Control': 'no-cache'
                }
            )
    except Exception as e:
        logger.error(f"Error in text search: {str(e)}")
        error_message = str(e)
        return JSONResponse(
            content={
                'success': False,
                'error': error_message,
                'details': {
                    'type': type(e).__name__,
                    'message': error_message
                }
            },
            status_code=500,
            headers={
                'Content-Type': 'application/json',
                'Cache-Control': 'no-cache'
            }
        )

@app.get("/analyze_face/{image_name}")
async def analyze_face(image_name: str):
    """Analyze facial attributes in an image."""
    try:
        image_path = Path("images") / image_name
        if not image_path.exists():
            return JSONResponse({"error": "Image not found"}, status_code=404)
            
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            return JSONResponse({"error": "Could not read image"}, status_code=400)
            
        # Get face detections with attributes
        results = analyzer.face_detector.detect_faces(image, image_name, analyze_attributes=True)
        if "error" in results:
            return JSONResponse(content=results, status_code=400)

        return JSONResponse(content=results, encoder=NumpyJSONEncoder)
        cache_dir = Path("cache")
        cache_dir.mkdir(exist_ok=True)
        
        # Save visualization temporarily
        vis_path = cache_dir / f"vis_{image_name}"
        cv2.imwrite(str(vis_path), vis_image)
        
        # Return results with proper JSON encoding
        return JSONResponse(
            content={
                "results": results,
                "visualization": f"vis_{image_name}"
            },
            headers={
                "Content-Type": "application/json",
                "Cache-Control": "no-cache"
            }
        )
        
    except Exception as e:
        logger.error(f"Error analyzing face: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)