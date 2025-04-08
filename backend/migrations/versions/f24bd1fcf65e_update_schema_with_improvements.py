"""update_schema_with_improvements

Revision ID: f24bd1fcf65e
Revises: d28028b1feae
Create Date: 2025-04-08 00:51:23.000000

"""
from alembic import op
import sqlalchemy as sa
from geoalchemy2 import Geography
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision = 'f24bd1fcf65e'
down_revision = 'd28028b1feae'
branch_labels = None
depends_on = None


def upgrade():
    # Create exif_metadata table
    op.create_table('exif_metadata',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('image_id', sa.Integer(), nullable=False),
        sa.Column('camera_make', sa.String(), nullable=True),
        sa.Column('camera_model', sa.String(), nullable=True),
        sa.Column('focal_length', sa.Float(), nullable=True),
        sa.Column('exposure_time', sa.String(), nullable=True),
        sa.Column('f_number', sa.Float(), nullable=True),
        sa.Column('iso', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['image_id'], ['images.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('image_id')
    )

    # Add new columns to images
    op.add_column('images', sa.Column('latitude', sa.Float(), nullable=True))
    op.add_column('images', sa.Column('longitude', sa.Float(), nullable=True))
    
    # Add indexes
    op.create_index('ix_images_filename', 'images', ['filename'])
    op.create_index('ix_images_date_taken', 'images', ['date_taken'])
    op.create_index('ix_face_detections_image_id', 'face_detections', ['image_id'])
    op.create_index('ix_face_detections_identity_id', 'face_detections', ['identity_id'])
    op.create_index('ix_object_detections_image_id', 'object_detections', ['image_id'])
    op.create_index('ix_text_detections_image_id', 'text_detections', ['image_id'])
    op.create_index('ix_scene_classifications_image_id', 'scene_classifications', ['image_id'])

    # Add confidence column to face_detections
    op.add_column('face_detections', sa.Column('confidence', sa.Float(), nullable=True))

    # Update nullable constraints for images table
    op.alter_column('images', 'created_at',
               existing_type=postgresql.TIMESTAMP(),
               nullable=False)
    op.alter_column('images', 'dimensions',
               existing_type=sa.VARCHAR(),
               nullable=False)
    op.alter_column('images', 'format',
               existing_type=sa.VARCHAR(),
               nullable=False)
    op.alter_column('images', 'file_size',
               existing_type=sa.Float(),
               type_=sa.Integer(),
               nullable=False)

    # Update image_id to non-nullable for all tables
    for table in ['face_detections', 'object_detections', 'text_detections', 'scene_classifications']:
        op.alter_column(table, 'image_id',
                   existing_type=sa.INTEGER(),
                   nullable=False)

    # Update bounding_box to non-nullable only for tables that have it
    for table in ['face_detections', 'object_detections', 'text_detections']:
        op.alter_column(table, 'bounding_box',
                   existing_type=postgresql.JSON(astext_type=sa.Text()),
                   nullable=False)

    # Add specific constraints for each table
    op.alter_column('face_identities', 'label',
               existing_type=sa.VARCHAR(),
               nullable=False)
    op.alter_column('object_detections', 'label',
               existing_type=sa.VARCHAR(),
               nullable=False)
    op.alter_column('object_detections', 'confidence',
               existing_type=sa.FLOAT(),
               nullable=False)
    op.alter_column('text_detections', 'text',
               existing_type=sa.VARCHAR(),
               nullable=False)
    op.alter_column('text_detections', 'confidence',
               existing_type=sa.FLOAT(),
               nullable=False)
    op.alter_column('scene_classifications', 'scene_type',
               existing_type=sa.VARCHAR(),
               nullable=False)
    op.alter_column('scene_classifications', 'confidence',
               existing_type=sa.FLOAT(),
               nullable=False)


def downgrade():
    # Remove indexes
    op.drop_index('ix_scene_classifications_image_id')
    op.drop_index('ix_text_detections_image_id')
    op.drop_index('ix_object_detections_image_id')
    op.drop_index('ix_face_detections_identity_id')
    op.drop_index('ix_face_detections_image_id')
    op.drop_index('ix_images_date_taken')
    op.drop_index('ix_images_filename')

    # Remove new columns
    op.drop_column('images', 'longitude')
    op.drop_column('images', 'latitude')
    op.drop_column('face_detections', 'confidence')

    # Drop exif_metadata table
    op.drop_table('exif_metadata')

    # Revert nullable constraints for images
    op.alter_column('images', 'file_size',
               existing_type=sa.Integer(),
               type_=sa.Float(),
               nullable=True)
    op.alter_column('images', 'format',
               existing_type=sa.VARCHAR(),
               nullable=True)
    op.alter_column('images', 'dimensions',
               existing_type=sa.VARCHAR(),
               nullable=True)
    op.alter_column('images', 'created_at',
               existing_type=postgresql.TIMESTAMP(),
               nullable=True)

    # Revert image_id to nullable for all tables
    for table in ['face_detections', 'object_detections', 'text_detections', 'scene_classifications']:
        op.alter_column(table, 'image_id',
                   existing_type=sa.INTEGER(),
                   nullable=True)

    # Revert bounding_box to nullable only for tables that have it
    for table in ['face_detections', 'object_detections', 'text_detections']:
        op.alter_column(table, 'bounding_box',
                   existing_type=postgresql.JSON(astext_type=sa.Text()),
                   nullable=True)

    # Revert specific constraints for each table
    op.alter_column('face_identities', 'label',
               existing_type=sa.VARCHAR(),
               nullable=True)
    op.alter_column('object_detections', 'label',
               existing_type=sa.VARCHAR(),
               nullable=True)
    op.alter_column('object_detections', 'confidence',
               existing_type=sa.FLOAT(),
               nullable=True)
    op.alter_column('text_detections', 'text',
               existing_type=sa.VARCHAR(),
               nullable=True)
    op.alter_column('text_detections', 'confidence',
               existing_type=sa.FLOAT(),
               nullable=True)
    op.alter_column('scene_classifications', 'scene_type',
               existing_type=sa.VARCHAR(),
               nullable=True)
    op.alter_column('scene_classifications', 'confidence',
               existing_type=sa.FLOAT(),
               nullable=True)
