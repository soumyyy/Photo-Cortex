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
    embeddings = relationship("ImageEmbedding", back_populates="image", cascade="all, delete-orphan")
    similarity_groups = relationship("SimilarImageGroup", secondary="similar_image_group_members", back_populates="members")

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

class ImageEmbedding(Base):
    __tablename__ = 'image_embeddings'
    
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('images.id', ondelete='CASCADE'), index=True, nullable=False)
    embedding_type = Column(String, nullable=False)  # 'clip', 'face', 'object', 'phash'
    embedding = Column(Vector(512), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    
    # Relationships
    image = relationship("Image", back_populates="embeddings")
    
    # Indexes for faster querying
    __table_args__ = (
        Index('idx_image_embeddings_type', 'embedding_type'),
        Index('idx_image_embeddings_image_type', 'image_id', 'embedding_type', unique=True),
    )

class SimilarImageGroup(Base):
    __tablename__ = 'similar_image_groups'
    
    id = Column(Integer, primary_key=True)
    group_type = Column(String, nullable=False)  # 'visual', 'face', 'object', 'phash'
    key_image_id = Column(Integer, ForeignKey('images.id', ondelete='SET NULL'), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    
    # Relationships
    key_image = relationship("Image", foreign_keys=[key_image_id])
    members = relationship("Image", secondary="similar_image_group_members", back_populates="similarity_groups")
    
    # Indexes
    __table_args__ = (
        Index('idx_similar_groups_type', 'group_type'),
    )

class SimilarImageGroupMember(Base):
    __tablename__ = 'similar_image_group_members'
    
    group_id = Column(Integer, ForeignKey('similar_image_groups.id', ondelete='CASCADE'), primary_key=True)
    image_id = Column(Integer, ForeignKey('images.id', ondelete='CASCADE'), primary_key=True)
    similarity_score = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_group_members_image', 'image_id'),
        Index('idx_group_members_score', 'similarity_score'),
    )