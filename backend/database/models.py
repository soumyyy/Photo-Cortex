from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, JSON
from sqlalchemy.orm import relationship
from geoalchemy2 import Geography
from sqlalchemy.dialects.postgresql import ARRAY
from pgvector.sqlalchemy import Vector
import datetime

Base = declarative_base()

class Image(Base):
    __tablename__ = 'images'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # EXIF metadata
    date_taken = Column(DateTime)
    camera_make = Column(String)
    camera_model = Column(String)
    focal_length = Column(Float)
    exposure_time = Column(String)
    f_number = Column(Float)
    iso = Column(Integer)
    dimensions = Column(String)
    format = Column(String)
    file_size = Column(Float)  # in KB
    
    # GPS coordinates using PostGIS
    location = Column(Geography(geometry_type='POINT', srid=4326))
    
    # CLIP embedding
    embedding = Column(Vector(512))  # Adjust dimension based on your CLIP model
    
    # Relationships
    face_detections = relationship("FaceDetection", back_populates="image")
    object_detections = relationship("ObjectDetection", back_populates="image")
    text_detections = relationship("TextDetection", back_populates="image")
    scene_classifications = relationship("SceneClassification", back_populates="image")

class FaceDetection(Base):
    __tablename__ = 'face_detections'
    
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('images.id'))
    embedding = Column(Vector(512))  # Face embedding
    bounding_box = Column(JSON)  # {x1, y1, x2, y2}
    landmarks = Column(JSON)
    similarity_score = Column(Float)
    identity_id = Column(Integer, ForeignKey('face_identities.id'), nullable=True)
    
    image = relationship("Image", back_populates="face_detections")
    identity = relationship("FaceIdentity", back_populates="detections")

class FaceIdentity(Base):
    __tablename__ = 'face_identities'
    
    id = Column(Integer, primary_key=True)
    label = Column(String)
    reference_embedding = Column(Vector(512))
    
    detections = relationship("FaceDetection", back_populates="identity")

class ObjectDetection(Base):
    __tablename__ = 'object_detections'
    
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('images.id'))
    label = Column(String)
    confidence = Column(Float)
    bounding_box = Column(JSON)  # {x1, y1, x2, y2}
    
    image = relationship("Image", back_populates="object_detections")

class TextDetection(Base):
    __tablename__ = 'text_detections'
    
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('images.id'))
    text = Column(String)
    confidence = Column(Float)
    bounding_box = Column(JSON)  # {x1, y1, x2, y2}
    
    image = relationship("Image", back_populates="text_detections")

class SceneClassification(Base):
    __tablename__ = 'scene_classifications'
    
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('images.id'))
    scene_type = Column(String)
    confidence = Column(Float)
    
    image = relationship("Image", back_populates="scene_classifications")