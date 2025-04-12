# PhotoCortex

PhotoCortex is an AI-powered photo analysis and organization platform that helps you understand and explore your photo collection in new ways. It combines modern web technologies with state-of-the-art computer vision to provide rich insights about your images.

## Tech Stack Used

### Backend Technologies
- **Python 3.9+**
- **FastAPI** - Modern async web framework
- **PostgreSQL** - Primary database

### Frontend Technologies
- **Next.js 13+** - React framework with App Router
- **React 18** - UI library
- **TypeScript** - Type safety
- **Tailwind CSS** - Utility-first styling

## Features

### Image Analysis
- **Face Detection & Recognition**
  - Advanced face detection using InsightFace
  - Automatic face grouping by identity
  - Face attribute analysis (smile, eyes)
  - Face crop storage with proper color handling
  - Identity management with custom labels
  
- **Scene Understanding**
  - Object detection with confidence scores
  - Scene classification and categorization
  - Text recognition (OCR) for embedded text
  - EXIF metadata extraction including GPS coordinates

### Interactive Views
- **Photo Grid**
  - Responsive masonry layout
  - Advanced filtering options
  - Real-time analysis feedback
  - Detailed image information panel

- **People View**
  - Identity-based face grouping
  - Face grid with detection thumbnails
  - Custom naming for identities
  - Detection count-based sorting
  
- **Location Features**
  - Map view for geotagged photos
  - Location clustering
  - GPS coordinate extraction

## Project Structure
```
/backend
├── models/
│   ├── inference/        # ML model implementations
│   └── weights/          # Pre-trained model weights
├── database/
│   ├── models.py         # SQLAlchemy models
│   └── database_service.py
└── main.py              # FastAPI application

/frontend
├── src/
│   ├── app/             # Next.js app router
│   ├── components/      # React components
│   └── lib/            # Utility functions
└── public/             # Static assets
```

Made with ❤️