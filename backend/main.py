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
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize analyzer
analyzer = ImageAnalyzer()

# Default images directory
IMAGES_DIR = Path("./images")
IMAGES_DIR.mkdir(exist_ok=True)

# Mount the images directory
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")

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
                    
                    # Skip text recognition temporarily
                    text_results = {
                        "text_detected": False,
                        "text_blocks": [],
                        "total_confidence": 0.0
                    }
                    
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)