import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import logging
import json
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from sqlalchemy.orm import selectinload
from database.database_service import DatabaseService
from database.config import get_async_session, get_db
from pydantic import BaseModel, Field
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
from database.models import Image as DBImage, TextDetection, ObjectDetection, SceneClassification, FaceIdentity, FaceDetection, ExifMetadata
from fastapi.responses import JSONResponse
from fastapi import status
from fastapi.encoders import jsonable_encoder
import shutil
from models.inference.face_detector import FaceDetector
from werkzeug.utils import secure_filename
import magic

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('app.log')  # File output with full details
    ],
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Configure individual loggers
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)  # Show SQL queries
logging.getLogger('uvicorn.access').setLevel(logging.INFO)
logging.getLogger('fastapi').setLevel(logging.INFO)
logging.getLogger('models').setLevel(logging.DEBUG)  # Show model operations

# Get the main logger
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
from models.inference.image_analyzer import ImageAnalyzer
analyzer = ImageAnalyzer()
log_human_readable("PhotoCortex models loaded and ready")

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Keep path resolution simple and do it once
BASE_DIR = Path(__file__).parent.resolve()
IMAGE_DIR = BASE_DIR / "image"
FACE_DIR = IMAGE_DIR / "faces"

# Include API_BASE_URL in config
APP_CONFIG = {
    "IMAGE_DIR": str(IMAGE_DIR),
    "FACE_DIR": str(FACE_DIR),
    "API_BASE_URL": "http://localhost:8000"  # Add this
}

# Create directories in one go
for directory in [IMAGE_DIR, FACE_DIR]:
    os.makedirs(directory, exist_ok=True)

# Keep the static file mount as is
app.mount("/image", StaticFiles(directory=str(IMAGE_DIR)), name="image")

# Validation functions
def validate_image_id(image_id: int) -> int:
    if image_id <= 0:
        raise HTTPException(status_code=400, detail="Invalid image ID")
    return image_id

def validate_threshold(threshold: float = Query(0.7, ge=0.0, le=1.0)) -> float:
    return threshold

def validate_pagination(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100)
) -> Tuple[int, int]:
    return page, page_size

# Response models
class UploadResponse(BaseModel):
    filename: str
    analysis: dict
    cached: bool = False

class SimilarImageInfo(BaseModel):
    image_id: int
    filename: str
    similarity_score: float

class SimilarityGroupResponse(BaseModel):
    group_id: int
    group_type: str
    key_image_id: int
    members: List[SimilarImageInfo]

class SimilarityGroupCreate(BaseModel):
    group_type: str = Field("visual", pattern="^(visual|semantic|face)$")
    threshold: float = Field(0.7, ge=0.0, le=1.0)

class PaginatedSimilarImages(BaseModel):
    total: int
    items: List[SimilarImageInfo]
    page: int
    page_size: int

class EmbeddingResponse(BaseModel):
    image_id: int
    success: bool
    message: str
    error: Optional[str] = None

@app.get("/config")
async def get_config():
    """Get application configuration."""
    return JSONResponse(content=APP_CONFIG)

@app.get("/image/faces/{face_identifier}")
async def get_face_image(face_identifier: str):
    """Serve face images with proper headers."""
    # The face_identifier should already be in the correct format (face_{image_stem}_{id}.jpg)
    # Just need to handle the .jpg extension if not present
    if not face_identifier.endswith('.jpg'):
        face_identifier = f"{face_identifier}.jpg"
    
    image_path = FACE_DIR / face_identifier
    
    if not image_path.exists():
        logger.error(f"Face image not found: {image_path}")
        return JSONResponse(
            status_code=404,
            content={"error": f"Face image {face_identifier} not found"}
        )
    
    mime_type, _ = mimetypes.guess_type(str(image_path))
    mime_type = mime_type or 'application/octet-stream'
    return FileResponse(
        path=image_path,
        media_type=mime_type,
        headers={"Cache-Control": "public, max-age=3600"}
    )

@app.get("/image/{image_name}")
async def get_image(image_name: str):
    """Serve images with proper headers."""
    image_path = IMAGE_DIR / image_name
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
        image_full_path = Path(APP_CONFIG["IMAGE_DIR"]) / image_path
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

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to ensure consistent handling of spaces and special characters."""
    # Replace spaces with underscores
    filename = filename.replace(" ", "_")
    # Remove any other potentially problematic characters
    filename = "".join(c for c in filename if c.isalnum() or c in "._-")
    return filename

@app.post("/analyze-image")
async def analyze_image(
    file_data: dict,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Analyze a single image and return its results."""
    db_service = DatabaseService(lambda: db)
    
    try:
        filename = file_data.get("filename")
        if not filename:
            raise HTTPException(status_code=400, detail="Filename is required")
            
        # Sanitize filename before any database operations
        sanitized_filename = sanitize_filename(filename)
        image_path = IMAGE_DIR / sanitized_filename
        
        if not image_path.exists():
            # Try with original filename if sanitized version doesn't exist
            original_path = IMAGE_DIR / filename
            if not original_path.exists():
                raise HTTPException(status_code=404, detail=f"Image not found: {filename}")
            else:
                # If original exists but sanitized doesn't, rename it
                original_path.rename(image_path)
                
        # Use sanitized filename for all database operations
        image = await db_service.get_image_by_filename(sanitized_filename)
        if image:
            analysis = await db_service.get_image_analysis(image.id)
            if analysis:
                return JSONResponse(content={
                    "filename": sanitized_filename,
                    "metadata": analysis.get("metadata", {}),
                    "faces": analysis.get("faces", []),
                    "objects": analysis.get("objects", []),
                    "scene_classification": analysis.get("scene_classification"),
                    "text_recognition": analysis.get("text_recognition", {}),
                    "cached": True
                })

        # Extract metadata
        metadata = extract_image_metadata(image_path)
        
        # Analyze image
        faces = await analyzer.detect_faces(str(image_path))
        objects = await analyzer.detect_objects(str(image_path))
        scene = await analyzer.classify_scene(str(image_path))
        text_recognition = await analyzer.text_recognizer.detect_text(str(image_path))
        
        # Create analysis result
        analysis_result = {
            "filename": sanitized_filename,
            "metadata": metadata,
            "faces": faces,
            "objects": objects,
            "scene_classification": scene,
            "text_recognition": text_recognition,
            "cached": False
        }
        
        # Save to database
        await db_service.save_image_analysis(
            filename=sanitized_filename,
            metadata=metadata,
            faces=faces,
            objects=objects,
            scene_classification=scene,
            text_recognition=text_recognition
        )
        
        # Use NumpyJSONEncoder to handle numpy arrays and special types
        return JSONResponse(
            content=json.loads(json.dumps(analysis_result, cls=NumpyJSONEncoder))
        )
        
    except Exception as e:
        logger.error(f"Error analyzing image {filename}: {e}")
        
        # Even if there was an error, try to get the analysis that was saved
        try:
            # Use sanitized filename for DB lookup
            sanitized_filename = sanitize_filename(filename)
            image = await db_service.get_image_by_filename(sanitized_filename)
            if image:
                # Then get the full analysis
                analysis = await db_service.get_image_analysis(image.id)
                if analysis:
                    logger.info(f"Successfully retrieved analysis for {filename} after error")
                    return JSONResponse(
                        status_code=200,
                        content={
                            "analysis": analysis,
                            "warning": "Analysis completed with some errors"
                        }
                    )
        except Exception as db_error:
            logger.error(f"Failed to get analysis from DB after error: {db_error}")
        
        # If we couldn't get the analysis, return the original error
        raise HTTPException(
            status_code=500, 
            detail=f"Error analyzing image: {str(e) if str(e) else 'Unknown error'}"
        )

async def analyze_image_stream(image_files: list, db_service: DatabaseService) -> AsyncGenerator[str, None]:
    """Stream analysis results for each image."""
    total_files = len(image_files)
    results = []
    
    for i, image_path in enumerate(image_files):
        try:
            # Convert string path to Path object if needed
            image_path_obj = Path(str(image_path)).resolve()
            
            # Get a session from the database service
            async with await db_service.get_session() as session:
                # Call analyze-image endpoint for each file
                analysis = await analyze_image(
                    {"filename": image_path_obj.name},
                    session
                )
                
                # Extract content from JSONResponse
                if isinstance(analysis, JSONResponse):
                    analysis = json.loads(analysis.body)
                
                results.append(analysis)
                
                # Send progress update
                progress = {
                    "progress": min(100, int(((i + 1) / total_files) * 100)),
                    "current": i + 1,
                    "total": total_files,
                    "filename": image_path_obj.name,
                    "cached": analysis.get("cached", False),
                    "complete": False,
                    "latest_result": analysis
                }
                yield json.dumps(progress, cls=NumpyJSONEncoder) + "\n"
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            yield json.dumps({
                "error": f"Failed to process {image_path}: {str(e)}",
                "progress": min(100, int(((i + 1) / total_files) * 100)),
                "current": i + 1,
                "total": total_files
            }) + "\n"
            continue
    
    # Send final complete message with all results
    yield json.dumps({
        "complete": True,
        "results": results,
        "total": total_files
    }, cls=NumpyJSONEncoder) + "\n"

@app.get("/analyze-folder")
async def analyze_folder(db: AsyncSession = Depends(get_db)):
    """Analyze all images in the folder and stream results."""
    try:
        image_files = [f for f in IMAGE_DIR.glob("*") if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
        if not image_files:
            return JSONResponse(status_code=400, content={"error": "No images found in the directory"})
        
        # Create database service
        db_service = DatabaseService(lambda: db)
        return StreamingResponse(analyze_image_stream(image_files, db_service), media_type="text/event-stream")
    except Exception as e:
        logger.error(f"Failed to analyze folder: {e}")
        return JSONResponse(status_code=500, content={"error": "Failed to analyze folder"})

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
    """Get unique faces from the database."""
    try:
        # Query to get all face identities with their detections
        query = select(FaceIdentity).options(
            selectinload(FaceIdentity.detections).selectinload(FaceDetection.image)
        )
        result = await db.execute(query)
        identities = result.scalars().all()
        
        # Convert identities to list of dicts with detection count
        identity_list = []
        for identity in identities:
            detections = []
            image_ids = set()  # Track unique images
            
            for detection in identity.detections:
                if detection.image:
                    image_ids.add(detection.image.id)
                    detections.append({
                        "id": detection.id,
                        "image_id": detection.image.id,
                        "image_path": f"image/faces/face_{detection.image.id}_{detection.id}.jpg"  
                    })
            
            identity_list.append({
                "id": identity.id,
                "label": identity.label,
                "images": [f"image/{img.filename}" for img in [d.image for d in identity.detections if d.image]],  
                "detections": detections,
                "detection_count": len(detections)  # Add count for sorting
            })
        
        # Sort by detection count in decreasing order
        identity_list.sort(key=lambda x: x["detection_count"], reverse=True)
        
        # Remove the count from final output
        for identity in identity_list:
            del identity["detection_count"]
        
        return JSONResponse(content=identity_list)
        
    except Exception as e:
        logger.error(f"Error getting unique faces: {e}")
        return JSONResponse(
            status_code=500, 
            content={"error": str(e)}
        )

@app.post("/api/image/{image_id}/compute-embeddings", 
          response_model=EmbeddingResponse,
          responses={
              404: {"description": "Image not found"},
              415: {"description": "Unsupported image format"},
              500: {"description": "Processing error"}
          })
async def compute_image_embeddings(
    image_id: int = validate_image_id,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Compute and store embeddings for an image.
    
    Args:
        image_id: ID of the image to process
        
    Returns:
        EmbeddingResponse with success status and any error messages
        
    Raises:
        HTTPException: If image not found or processing fails
    """
    try:
        # Get image record
        image = await db.get(DBImage, image_id)
        if not image:
            raise HTTPException(status_code=404, detail="Image not found")
            
        image_path = IMAGE_DIR / image.filename
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image file not found")

        # Check file type
        mime = magic.from_file(str(image_path), mime=True)
        if not mime.startswith('image/'):
            raise HTTPException(
                status_code=415,
                detail=f"Invalid file type: {mime}. Expected image/*"
            )

        # Initialize similarity processor if needed
        if not analyzer.similarity_processor:
            analyzer.set_similarity_processor(DatabaseService(get_async_session))

        # Process image
        result = await analyzer.similarity_processor.process_single_image(
            str(image_path),
            image_id
        )
        
        if not result['success']:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process image: {result.get('error', 'Unknown error')}"
            )
            
        return {
            "image_id": image_id,
            "success": True,
            "message": "Embeddings computed and stored successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing embeddings for image {image_id}: {e}")
        return {
            "image_id": image_id,
            "success": False,
            "message": "Failed to compute embeddings",
            "error": str(e)
        }

@app.get("/api/image/{image_id}/similar", 
         response_model=PaginatedSimilarImages,
         responses={
             404: {"description": "Image not found"},
             500: {"description": "Processing error"}
         })
async def find_similar_images(
    image_id: int = validate_image_id,
    threshold: float = validate_threshold,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
) -> PaginatedSimilarImages:
    """Find similar images for a given image.
    
    Args:
        image_id: ID of the source image
        threshold: Similarity threshold (0.0 to 1.0)
        page: Page number for pagination
        page_size: Number of items per page
        
    Returns:
        PaginatedSimilarImages with list of similar images
        
    Raises:
        HTTPException: If image not found or processing fails
    """
    try:
        # Check if image exists
        image = await db.get(DBImage, image_id)
        if not image:
            raise HTTPException(status_code=404, detail="Image not found")

        # Initialize similarity processor if needed
        if not analyzer.similarity_processor:
            analyzer.set_similarity_processor(DatabaseService(get_async_session))
            
        # Find similar images
        similar_images = await analyzer.similarity_processor.find_similar_images(
            image_id,
            threshold=threshold
        )
        
        # Apply pagination
        start = (page - 1) * page_size
        end = start + page_size
        paginated_items = similar_images[start:end]
        
        return {
            "total": len(similar_images),
            "items": paginated_items,
            "page": page,
            "page_size": page_size
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding similar images for {image_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error finding similar images: {str(e)}"
        )

@app.post("/api/similarity-groups", 
          response_model=SimilarityGroupResponse,
          responses={
              400: {"description": "Invalid request or no similar images found"},
              404: {"description": "Image not found"},
              500: {"description": "Processing error"}
          })
async def create_similarity_group(
    group_data: SimilarityGroupCreate,
    image_id: int = validate_image_id,
    db: AsyncSession = Depends(get_db)
) -> SimilarityGroupResponse:
    """Create a new similarity group with the given image as key.
    
    Args:
        group_data: Group creation parameters
        image_id: ID of the key image
        
    Returns:
        SimilarityGroupResponse with group details and members
        
    Raises:
        HTTPException: If creation fails or invalid parameters
    """
    try:
        # Check if image exists
        image = await db.get(DBImage, image_id)
        if not image:
            raise HTTPException(status_code=404, detail="Image not found")

        # Initialize similarity processor if needed
        if not analyzer.similarity_processor:
            analyzer.set_similarity_processor(DatabaseService(get_async_session))
            
        # Find similar images first
        similar_images = await analyzer.similarity_processor.find_similar_images(
            image_id,
            threshold=group_data.threshold
        )
        
        if not similar_images:
            raise HTTPException(
                status_code=400,
                detail="No similar images found with the given threshold"
            )
            
        # Create the group
        group = await analyzer.similarity_processor.create_similarity_group(
            image_id,
            similar_images,
            group_type=group_data.group_type
        )
        
        if not group:
            raise HTTPException(
                status_code=500,
                detail="Failed to create similarity group"
            )
            
        return group
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating similarity group for image {image_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error creating similarity group: {str(e)}"
        )

@app.get("/api/similarity-groups/{group_id}", 
         response_model=SimilarityGroupResponse,
         responses={
             404: {"description": "Group not found"},
             500: {"description": "Processing error"}
         })
async def get_similarity_group(
    group_id: int,
    db: AsyncSession = Depends(get_db)
) -> SimilarityGroupResponse:
    """Get details of a similarity group.
    
    Args:
        group_id: ID of the similarity group
        
    Returns:
        SimilarityGroupResponse with group details and members
        
    Raises:
        HTTPException: If group not found or processing fails
    """
    try:
        # Get group from database
        group = await db.get(SimilarImageGroup, group_id)
        if not group:
            raise HTTPException(status_code=404, detail="Similarity group not found")
            
        # Get all members
        stmt = select(DBImage, SimilarImageGroupMember).join(
            SimilarImageGroupMember,
            DBImage.id == SimilarImageGroupMember.image_id
        ).where(SimilarImageGroupMember.group_id == group_id)
        
        result = await db.execute(stmt)
        members = result.all()
        
        return {
            "group_id": group.id,
            "group_type": group.group_type,
            "key_image_id": group.key_image_id,
            "members": [
                {
                    "image_id": img.id,
                    "filename": img.filename,
                    "similarity_score": member.similarity_score
                }
                for img, member in members
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting similarity group {group_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting similarity group: {str(e)}"
        )

@app.get("/api/image/{image_id}/similarity-groups", 
         response_model=List[SimilarityGroupResponse],
         responses={
             404: {"description": "Image not found"},
             500: {"description": "Processing error"}
         })
async def get_image_similarity_groups(
    image_id: int = validate_image_id,
    db: AsyncSession = Depends(get_db)
) -> List[SimilarityGroupResponse]:
    """Get all similarity groups that contain this image.
    
    Args:
        image_id: ID of the image
        
    Returns:
        List of SimilarityGroupResponse containing group details
        
    Raises:
        HTTPException: If image not found or processing fails
    """
    try:
        # Check if image exists
        image = await db.get(DBImage, image_id)
        if not image:
            raise HTTPException(status_code=404, detail="Image not found")

        # Initialize similarity processor if needed
        if not analyzer.similarity_processor:
            analyzer.set_similarity_processor(DatabaseService(get_async_session))
            
        # Get groups from database service
        groups = await analyzer.similarity_processor.db_service.get_similar_images(image_id)
        return groups
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting similarity groups for image {image_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting similarity groups: {str(e)}"
        )

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
        file_path = IMAGE_DIR / filename
        
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
        image_path = IMAGE_DIR / filename
        
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

@app.post("/reprocess-faces")
async def reprocess_faces(db: AsyncSession = Depends(get_db)):
    """Reprocess all face detections and identities in the database."""
    try:
        log_human_readable("Starting face detection reprocessing...")
        
        # Get all images from database
        query = select(DBImage)
        result = await db.execute(query)
        images = result.scalars().all()
        
        if not images:
            return JSONResponse(
                content={"message": "No images found in database"},
                status_code=404
            )
        
        # Clear existing face detections and identities
        await db.execute(FaceDetection.__table__.delete())
        await db.execute(FaceIdentity.__table__.delete())
        await db.commit()
        
        # Clear existing face crops
        if FACE_DIR.exists():
            for face_file in FACE_DIR.glob("face_*.jpg"):
                face_file.unlink()
        
        total = len(images)
        processed = 0
        total_faces = 0
        successful_face_crops = set()
        unique_identities = set()
        
        for image in images:
            try:
                # Load image
                image_path = IMAGE_DIR / image.filename
                if not image_path.exists():
                    logger.error(f"Image not found: {image_path}")
                    continue
                
                # Read image
                img = cv2.imread(str(image_path))
                if img is None:
                    logger.error(f"Failed to read image: {image_path}")
                    continue
                
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Detect faces using the face detector directly
                result = await analyzer.face_detector.detect_faces(img, image.id, db)
                
                if result.get("faces_detected", False):
                    faces = result.get("faces", [])
                    total_faces += len(faces)
                    
                    # Track successful face crops and unique identities
                    for face in faces:
                        if face.get("face_image"):
                            successful_face_crops.add(face["face_image"])
                        if face.get("identity_id"):
                            unique_identities.add(face["identity_id"])
                
                processed += 1
                if processed % 5 == 0:
                    log_human_readable(f"Processed {processed}/{total} images, found {total_faces} faces so far...")
                
            except Exception as e:
                logger.error(f"Error processing image {image.filename}: {e}")
                continue
        
        await db.commit()
        
        summary = (
            f"Face detection reprocessing complete.\n"
            f"Processed {processed}/{total} images\n"
            f"Total faces detected: {total_faces}\n"
            f"Unique identities found: {len(unique_identities)}\n"
            f"Face crops saved: {len(successful_face_crops)}"
        )
        log_human_readable(summary)
        
        return JSONResponse(
            content={
                "message": "Face detection reprocessing complete",
                "processed_images": processed,
                "total_images": total,
                "total_faces_detected": total_faces,
                "unique_identities": len(unique_identities),
                "face_crops_saved": len(successful_face_crops)
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to reprocess faces: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/recut-face-images")
async def recut_face_images(db: AsyncSession = Depends(get_db)):
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
                src_path = os.path.join(APP_CONFIG["IMAGE_DIR"], image.filename)
                face_filename = f"face_{detection.id}.jpg"
                dest_path = os.path.join(APP_CONFIG["IMAGE_DIR"], face_filename)
                
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
            "faces_dir": os.path.abspath(APP_CONFIG["IMAGE_DIR"])
        }
        
    except Exception as e:
        logger.error(f"Recutting failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete-image/{image_id}", status_code=status.HTTP_200_OK)
async def delete_image_endpoint(
    image_id: int = 0, 
    db: AsyncSession = Depends(get_db)
):
    """
    Delete an image, its associated analysis data from the database,
    any orphaned FaceIdentity records, and the corresponding image files (original and face cutouts) from the filesystem.
    """
    logger.info(f"Attempting to delete image with ID: {image_id}")

    # --- 1. Fetch Image Record and Related Data --- 
    stmt = (
        select(DBImage)
        .options(selectinload(DBImage.face_detections))
        .where(DBImage.id == image_id)
    )
    result = await db.execute(stmt)
    image_record = result.scalar_one_or_none()

    if not image_record:
        logger.warning(f"Image with ID {image_id} not found for deletion.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Image with ID {image_id} not found")

    image_filename = image_record.filename
    face_detection_ids = [face.id for face in image_record.face_detections]
    face_identity_ids = [face.identity_id for face in image_record.face_detections if face.identity_id is not None]
    logger.debug(f"Found image record: {image_filename}. Found {len(face_detection_ids)} associated face detections.")

    # --- 2. Delete Database Records --- 
    try:
        related_tables = [
            FaceDetection, ObjectDetection, TextDetection, 
            SceneClassification, ExifMetadata
        ]
        for table in related_tables:
            delete_stmt = delete(table).where(table.image_id == image_id)
            await db.execute(delete_stmt)
            logger.debug(f"Executed delete for {table.__tablename__} related to image ID {image_id}.")
        await db.delete(image_record)
        logger.debug(f"Deleted image record for image ID {image_id}.")
        await db.commit()
        logger.info(f"Successfully deleted database records for image ID {image_id}.")
    except Exception as e:
        await db.rollback()
        logger.exception(f"Database error during deletion for image ID {image_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database error during deletion: {str(e)}")

    # --- 3. Delete Orphaned FaceIdentity Records ---
    deleted_identities = []
    try:
        if face_identity_ids:
            for identity_id in set(face_identity_ids):
                # Check if this identity is still referenced by any FaceDetection
                check_stmt = select(FaceDetection).where(FaceDetection.identity_id == identity_id)
                result = await db.execute(check_stmt)
                still_exists = result.scalar_one_or_none()
                if not still_exists:
                    # Delete the orphaned FaceIdentity
                    identity = await db.get(FaceIdentity, identity_id)
                    if identity:
                        await db.delete(identity)
                        deleted_identities.append(identity_id)
            if deleted_identities:
                await db.commit()
                logger.info(f"Deleted orphaned FaceIdentity records: {deleted_identities}")
    except Exception as e:
        await db.rollback()
        logger.error(f"Error deleting orphaned FaceIdentity records: {e}")

    # --- 4. Delete Filesystem Files --- 
    deleted_files = []
    failed_files = []
    original_image_path = IMAGE_DIR / image_filename
    try:
        if original_image_path.is_file():
            original_image_path.unlink()
            logger.info(f"Deleted original image file: {original_image_path}")
            deleted_files.append(str(original_image_path))
        else:
            logger.warning(f"Original image file not found (or not a file), skipping deletion: {original_image_path}")
    except Exception as e:
        logger.error(f"Error deleting original image file {original_image_path}: {e}")
        failed_files.append(str(original_image_path))
    for detection_id in face_detection_ids:
        face_filename = f"face_{image_id}_{detection_id}.jpg"
        face_image_path = IMAGE_DIR / face_filename
        try:
            if face_image_path.is_file():
                face_image_path.unlink()
                logger.info(f"Deleted face cutout file: {face_image_path}")
                deleted_files.append(str(face_image_path))
            else:
                 logger.warning(f"Face cutout file not found (or not a file), skipping deletion: {face_image_path}")
        except Exception as e:
            logger.error(f"Error deleting face cutout file {face_image_path}: {e}")
            failed_files.append(str(face_image_path))
    response_detail = f"Image ID {image_id} ({image_filename}) and associated data deleted."
    if deleted_identities:
        response_detail += f" Deleted orphaned FaceIdentity IDs: {deleted_identities}."
    if failed_files:
        response_detail += f" Failed to delete files: {', '.join(failed_files)}"
    return {"message": response_detail, "deleted_db_records": True, "deleted_files": deleted_files, "failed_files": failed_files, "deleted_identities": deleted_identities}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)