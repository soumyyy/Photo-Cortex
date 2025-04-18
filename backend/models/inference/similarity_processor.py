import torch
import clip
from ultralytics import YOLO
import insightface
import numpy as np
import cv2
from PIL import Image
import imagehash
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity as compare_ssim
import magic
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from database.database_service import DatabaseService

logger = logging.getLogger(__name__)

class SimilarityProcessor:
    def __init__(self, db_service: DatabaseService):
        """Initialize models and database service."""
        # Database service
        self.db_service = db_service
        
        # Device & models
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.yolo_model = YOLO("yolov8n.pt")
        self.face_model = insightface.app.FaceAnalysis()
        self.face_model.prepare(
            ctx_id=0 if torch.cuda.is_available() else -1,
            det_size=(640, 640)
        )

    async def process_single_image(self, image_path: str, image_id: int) -> Dict[str, Any]:
        """Process a single image and store its embeddings in the database."""
        try:
            # Load and process image
            img = Image.open(image_path).convert('RGB')
            arr = np.array(img)
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            ph = imagehash.phash(img)
            
            # Compute embeddings
            embeds = self._compute_embeddings(img, arr)
            
            # Store embeddings in database
            await self._store_embeddings(image_id, embeds, str(ph))
            
            return {
                'image_id': image_id,
                'embeddings': embeds,
                'phash': ph,
                'success': True
            }
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return {'image_id': image_id, 'success': False, 'error': str(e)}

    def _compute_embeddings(self, pil: Image.Image, arr: np.ndarray) -> Dict[str, Any]:
        """Compute all types of embeddings for an image."""
        return {
            'clip': self._clip_embedding(pil),
            'face': self._face_embeddings(arr),
            'objects': self._object_detection(arr)
        }

    def _clip_embedding(self, image: Image.Image) -> np.ndarray:
        """Compute CLIP embedding for an image."""
        with torch.no_grad():
            inp = self.preprocess(image).unsqueeze(0).to(self.device)
            emb = self.clip_model.encode_image(inp).cpu().numpy()[0]
            return emb / np.linalg.norm(emb)

    def _face_embeddings(self, arr: np.ndarray) -> List[np.ndarray]:
        """Compute face embeddings for an image."""
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        faces = self.face_model.get(arr)
        return [f.embedding/np.linalg.norm(f.embedding) for f in faces] if faces else []

    def _object_detection(self, arr: np.ndarray) -> List[int]:
        """Detect objects in an image."""
        res = self.yolo_model(arr)
        return [int(r.boxes.cls.cpu().numpy()[0]) for r in res if len(r.boxes)]

    async def _store_embeddings(self, image_id: int, embeddings: Dict[str, Any], phash: str) -> None:
        """Store all embeddings in the database."""
        try:
            # Store CLIP embedding
            await self.db_service.save_image_embedding(
                image_id=image_id,
                embedding_type='clip',
                embedding=embeddings['clip'].tolist()
            )
            
            # Store face embeddings
            for idx, face_emb in enumerate(embeddings['face']):
                await self.db_service.save_image_embedding(
                    image_id=image_id,
                    embedding_type=f'face_{idx}',
                    embedding=face_emb.tolist()
                )
            
            # Store object classes as a special type of embedding
            await self.db_service.save_image_embedding(
                image_id=image_id,
                embedding_type='objects',
                embedding=embeddings['objects']  # List of object class IDs
            )
            
            # Store perceptual hash
            await self.db_service.save_image_embedding(
                image_id=image_id,
                embedding_type='phash',
                embedding=[int(phash, 16)]  # Convert hash to number
            )
            
        except Exception as e:
            logger.error(f"Error storing embeddings for image {image_id}: {e}")
            raise

    async def find_similar_images(self, image_id: int, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find similar images for a given image ID."""
        try:
            # Get embeddings for the query image
            query_embeddings = await self.db_service.get_image_embeddings(image_id)
            if not query_embeddings:
                raise ValueError(f"No embeddings found for image {image_id}")
            
            # Convert embeddings to proper format
            query_embs = self._format_embeddings(query_embeddings)
            
            # Get all other images' embeddings
            # TODO: Optimize this to not load all embeddings at once
            all_images = await self.db_service.get_all_images()
            similar_images = []
            
            for img in all_images:
                if img.id == image_id:
                    continue
                    
                img_embeddings = await self.db_service.get_image_embeddings(img.id)
                if not img_embeddings:
                    continue
                
                img_embs = self._format_embeddings(img_embeddings)
                
                # Compute similarity
                similarity = self.compute_fused_similarity(
                    query_embs, img_embs,
                    int(query_embs['phash'][0]), int(img_embs['phash'][0])
                )
                
                if similarity >= threshold:
                    similar_images.append({
                        'image_id': img.id,
                        'filename': img.filename,
                        'similarity_score': float(similarity)
                    })
            
            return sorted(similar_images, key=lambda x: x['similarity_score'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error finding similar images for {image_id}: {e}")
            return []

    def _format_embeddings(self, db_embeddings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format database embeddings into the structure expected by similarity computation."""
        formatted = {
            'clip': None,
            'face': [],
            'objects': [],
            'phash': None
        }
        
        for emb in db_embeddings:
            if emb['type'] == 'clip':
                formatted['clip'] = np.array(emb['embedding'])
            elif emb['type'].startswith('face_'):
                formatted['face'].append(np.array(emb['embedding']))
            elif emb['type'] == 'objects':
                formatted['objects'] = emb['embedding']
            elif emb['type'] == 'phash':
                formatted['phash'] = emb['embedding']
        
        return formatted

    def compute_fused_similarity(self, emb1: Dict[str, Any], emb2: Dict[str, Any], ph1: int, ph2: int) -> float:
        """Compute fused similarity between two images using all available features."""
        w = {'hash': 0.4, 'clip': 0.3, 'face': 0.2, 'objects': 0.1}
        
        # pHash similarity
        hash_sim = max(0.0, 1 - bin(ph1 ^ ph2).count('1')/64)
        
        # CLIP similarity
        clip_sim = float(cosine_similarity([emb1['clip']], [emb2['clip']])[0,0]) if emb1['clip'] is not None and emb2['clip'] is not None else 0.0
        
        # Face similarity
        face_sim = self._face_similarity(emb1['face'], emb2['face'])
        
        # Object similarity
        o1, o2 = emb1['objects'], emb2['objects']
        obj_sim = len(set(o1) & set(o2)) / max(1, len(set(o1 + o2))) if o1 and o2 else 0.0
        
        return (
            w['hash'] * hash_sim +
            w['clip'] * clip_sim +
            w['face'] * face_sim +
            w['objects'] * obj_sim
        )

    def _face_similarity(self, f1: List[np.ndarray], f2: List[np.ndarray]) -> float:
        """Compute similarity between face embeddings of two images."""
        if not f1 or not f2:
            return 0.0
            
        matches, used = [], set()
        for e1 in f1:
            best, idx = 0.0, -1
            for i, e2 in enumerate(f2):
                if i in used:
                    continue
                sim = float(cosine_similarity([e1], [e2])[0,0])
                if sim > best:
                    best, idx = sim, i
            if best > 0.5:
                matches.append(best)
                used.add(idx)
                
        if not matches:
            return 0.0
            
        completeness = len(matches) / max(len(f1), len(f2))
        avg_sim = sum(matches) / len(matches)
        return completeness * avg_sim

    async def create_similarity_group(self, image_id: int, similar_images: List[Dict[str, Any]], group_type: str = 'visual') -> Optional[Dict[str, Any]]:
        """Create a similarity group from the results."""
        try:
            # Format member scores for database
            member_scores = [
                {'image_id': img['image_id'], 'score': img['similarity_score']}
                for img in similar_images
            ]
            
            # Create the group
            group = await self.db_service.create_similar_image_group(
                group_type=group_type,
                key_image_id=image_id,
                member_scores=member_scores
            )
            
            if group:
                return {
                    'group_id': group.id,
                    'group_type': group.group_type,
                    'key_image_id': group.key_image_id,
                    'members': similar_images
                }
            return None
            
        except Exception as e:
            logger.error(f"Error creating similarity group for image {image_id}: {e}")
            return None