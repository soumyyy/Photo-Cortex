from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio
import sys
import os

from .image_analyzer import ImageAnalyzer
from ...database.database_service import DatabaseService
from ...database.config import async_session

logger = logging.getLogger(__name__)

class ImageProcessingPipeline:
    def __init__(self):
        """Initialize the image processing pipeline."""
        self.analyzer = ImageAnalyzer()
        self.db_service = DatabaseService(async_session)
        self.executor = ThreadPoolExecutor(max_workers=4)
        logger.info("ImageProcessingPipeline initialized")

    async def process_image(self, image_path: Path, session: AsyncSession) -> Dict[str, Any]:
        """Process a single image."""
        try:
            # 1. Get filename
            filename = image_path.name
            
            # 2. Check if image already exists in database
            existing_image = await self.db_service.get_image_by_filename(filename)
            if existing_image:
                logger.info(f"Image {filename} already exists in database")
                return {
                    "id": existing_image.id,
                    "filename": existing_image.filename,
                    "status": "exists"
                }
            
            logger.info(f"Processing new image: {filename}")
            
            # 3. Run full analysis
            analysis_result = await self.analyzer.analyze_image_with_session(
                image_path,
                session
            )
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            raise

    async def batch_process_images(self, image_paths: list[Path], session: AsyncSession) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process multiple images in sequence, yielding results as they complete.
        
        Args:
            image_paths: List of paths to image files
            session: AsyncSession for database operations
            
        Yields:
            Dict containing analysis results for each image
        """
        tasks = []
        async with session:  # Ensure session is properly managed
            for image_path in image_paths:
                try:
                    # Create a new session for each image
                    async with session.begin():
                        result = await self.process_image(image_path, session)
                        yield {
                            "filename": image_path.name,
                            **result
                        }
                except Exception as e:
                    logger.error(f"Error processing {image_path.name}: {str(e)}")
                    yield {
                        "filename": image_path.name,
                        "success": False,
                        "error": str(e)
                    }
                finally:
                    await session.close()  # Ensure session is closed

    async def cleanup(self):
        """Cleanup resources."""
        if self.executor:
            self.executor.shutdown(wait=True)
        await self.db_service.cleanup()