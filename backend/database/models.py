from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, JSON, Index
from sqlalchemy.orm import relationship
from geoalchemy2 import Geography
from sqlalchemy.dialects.postgresql import ARRAY
from pgvector.sqlalchemy import Vector
import datetime

Base = declarative_base()

class Image(Base):
    __tablename__ = 'images'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String, unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    
    # Core metadata (frequently accessed)
    dimensions = Column(String, nullable=False)
    format = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)  # in bytes for precision
    date_taken = Column(DateTime, index=True)
    
    # Location data
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    location = Column(Geography(geometry_type='POINT', srid=4326), nullable=True)
    
    # ML features
    embedding = Column(Vector(512), nullable=True)  # CLIP embedding
    
    # Relationships
    exif_metadata = relationship("ExifMetadata", back_populates="image", uselist=False)
    face_detections = relationship("FaceDetection", back_populates="image")
    object_detections = relationship("ObjectDetection", back_populates="image")
    text_detections = relationship("TextDetection", back_populates="image")
    scene_classifications = relationship("SceneClassification", back_populates="image")

class ExifMetadata(Base):
    __tablename__ = 'exif_metadata'
    
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('images.id'), unique=True, nullable=False)
    camera_make = Column(String)
    camera_model = Column(String)
    focal_length = Column(Float)
    exposure_time = Column(String)
    f_number = Column(Float)
    iso = Column(Integer)
    
    image = relationship("Image", back_populates="exif_metadata")

class FaceDetection(Base):
    __tablename__ = 'face_detections'
    
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('images.id'), index=True, nullable=False)
    confidence = Column(Float, nullable=False)
    embedding = Column(Vector(512), nullable=False)
    bounding_box = Column(JSON, nullable=False)  # {x1, y1, x2, y2}
    landmarks = Column(JSON, nullable=True)
    similarity_score = Column(Float)
    identity_id = Column(Integer, ForeignKey('face_identities.id'), nullable=True, index=True)
    
    image = relationship("Image", back_populates="face_detections")
    identity = relationship("FaceIdentity", back_populates="detections")

class FaceIdentity(Base):
    __tablename__ = 'face_identities'
    
    id = Column(Integer, primary_key=True)
    label = Column(String, nullable=False)
    reference_embedding = Column(Vector(512), nullable=False)
    
    detections = relationship("FaceDetection", back_populates="identity")

class ObjectDetection(Base):
    __tablename__ = 'object_detections'
    
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('images.id'), index=True, nullable=False)
    label = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    bounding_box = Column(JSON, nullable=False)  # {x1, y1, x2, y2}
    
    image = relationship("Image", back_populates="object_detections")

class TextDetection(Base):
    __tablename__ = 'text_detections'
    
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('images.id'), index=True, nullable=False)
    text = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    bounding_box = Column(JSON, nullable=False)  # {x1, y1, x2, y2}
    
    image = relationship("Image", back_populates="text_detections")

class SceneClassification(Base):
    __tablename__ = 'scene_classifications'
    
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('images.id'), index=True, nullable=False)
    scene_type = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    
    image = relationship("Image", back_populates="scene_classifications")