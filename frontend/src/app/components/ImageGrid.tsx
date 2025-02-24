import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Dialog } from '@headlessui/react';
import { XMarkIcon, CameraIcon, CalendarIcon, DocumentIcon, MagnifyingGlassIcon } from '@heroicons/react/24/outline';
import Image from 'next/image';

interface TextBlock {
  id: number;
  text: string;
  confidence: number;
  bbox: {
    x_min: number;
    y_min: number;
    x_max: number;
    y_max: number;
  };
  points: number[][];
}

interface ImageAnalysis {
  filename: string;
  faces: Array<{
    confidence: number;
    bbox: number[];
    face_image?: string;
  }>;
  objects: string[];
  scene_classification: {
    scene_type: string;
    confidence: number;
  };
  text_recognition: {
    text_detected: boolean;
    text_blocks: TextBlock[];
    total_confidence: number;
  };
  metadata: {
    date_taken: string | null;
    camera_make: string | null;
    camera_model: string | null;
    focal_length: string | null;
    exposure_time: string | null;
    f_number: string | null;
    iso: string | null;
    dimensions: string;
    format: string;
    file_size: string;
  };
}

interface ImageGridProps {
  images: ImageAnalysis[];
}

export default function ImageGrid({ images }: ImageGridProps) {
  const [selectedImage, setSelectedImage] = useState<ImageAnalysis | null>(null);
  const [showTextScan, setShowTextScan] = useState(false);
  const [imageSize, setImageSize] = useState<{ width: number; height: number } | null>(null);
  const [loadedImages, setLoadedImages] = useState<Set<string>>(new Set());

  // Use intersection observer for lazy loading
  const observerRef = useRef<IntersectionObserver | null>(null);
  const imageRefs = useRef<{ [key: string]: HTMLDivElement | null }>({});

  useEffect(() => {
    observerRef.current = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const img = entry.target as HTMLDivElement;
            const filename = img.dataset.filename;
            if (filename) {
              setLoadedImages((prev) => new Set([...Array.from(prev), filename]));
            }
          }
        });
      },
      { rootMargin: '50px' }
    );

    return () => {
      if (observerRef.current) {
        observerRef.current.disconnect();
      }
    };
  }, []);

  // Observe image elements
  useEffect(() => {
    images.forEach((image) => {
      const element = imageRefs.current[image.filename];
      if (element && observerRef.current) {
        observerRef.current.observe(element);
      }
    });

    return () => {
      if (observerRef.current) {
        observerRef.current.disconnect();
      }
    };
  }, [images]);

  const handleImageLoad = (event: React.SyntheticEvent<HTMLImageElement>) => {
    const img = event.currentTarget;
    setImageSize({
      width: img.naturalWidth,
      height: img.naturalHeight
    });
  };

  return (
    <div>
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
        {images.map((image) => (
          <div 
            key={image.filename}
            ref={(el: HTMLDivElement | null) => {
              imageRefs.current[image.filename] = el;
            }}
            data-filename={image.filename}
            className="relative group cursor-pointer rounded-lg overflow-hidden bg-gray-800 aspect-square"
            style={{ maxWidth: '300px' }}
            onClick={() => {
              setSelectedImage(image);
              setShowTextScan(false);
            }}
          >
            {Array.from(loadedImages).includes(image.filename) ? (
              <div className="relative w-full h-full">
                <Image 
                  src={`http://localhost:8000/images/${encodeURIComponent(image.filename)}`} 
                  alt={image.filename}
                  className="object-cover"
                  fill
                  sizes="(max-width: 640px) 50vw, (max-width: 1024px) 33vw, 20vw"
                  priority={false}
                  loading="lazy"
                  quality={75}
                  onLoad={handleImageLoad}
                  onError={(e) => {
                    console.error(`Failed to load image: ${image.filename}`, e);
                    // Remove from loaded images if it fails
                    setLoadedImages(prev => {
                      const newSet = new Set(prev);
                      newSet.delete(image.filename);
                      return newSet;
                    });
                  }}
                />
              </div>
            ) : (
              <div className="w-full h-full flex items-center justify-center">
                <div className="animate-pulse bg-gray-700 w-full h-full" />
              </div>
            )}
            <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-200">
              <div className="absolute bottom-0 left-0 right-0 p-3 text-white">
                <p className="text-sm truncate">{image.filename}</p>
                <p className="text-xs text-gray-300">{image.scene_classification.scene_type}</p>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Image Details Modal */}
      <Dialog
        open={selectedImage !== null}
        onClose={() => {
          setSelectedImage(null);
          setShowTextScan(false);
        }}
        className="relative z-50"
      >
        <div className="fixed inset-0 bg-black/70" aria-hidden="true" />
        
        <div className="fixed inset-0 flex items-center justify-center p-4">
          <Dialog.Panel className="mx-auto max-w-4xl w-full bg-gray-900 rounded-xl shadow-2xl">
            <div className="relative">
              <button
                onClick={() => {
                  setSelectedImage(null);
                  setShowTextScan(false);
                }}
                className="absolute top-2 right-2 z-20 p-2 rounded-lg bg-gray-800/50 hover:bg-gray-700/50 transition-colors"
              >
                <XMarkIcon className="w-6 h-6 text-white" />
              </button>
              
              <div className="p-6">
                {selectedImage && (
                  <>
                    <div className="relative mb-6 rounded-lg overflow-hidden max-h-[calc(100vh-8rem)]">
                      {/* Main image with conditional dimming */}
                      <div className={`relative ${showTextScan ? 'brightness-50' : ''} transition-all duration-300`}>
                        <div className="relative w-full" style={{ aspectRatio: '16/9' }}>
                          <Image
                            src={`http://localhost:8000/images/${encodeURIComponent(selectedImage.filename)}`}
                            alt={selectedImage.filename}
                            className="object-contain"
                            fill
                            sizes="100vw"
                            priority={true}
                            quality={90}
                            onLoad={handleImageLoad}
                            onError={(e) => {
                              console.error(`Failed to load modal image: ${selectedImage.filename}`, e);
                            }}
                          />
                        </div>
                      </div>

                      {/* Text Detection Overlay */}
                      {showTextScan && imageSize && selectedImage.text_recognition && (
                        <div className="absolute inset-0 overflow-hidden">
                          {/* Dimming overlay for non-text areas */}
                          <div className="absolute inset-0 bg-black/60 transition-opacity duration-300" />
                          
                          {selectedImage.text_recognition.text_blocks?.map((block) => {
                            if (!block || !block.points || !block.bbox) return null;
                            
                            const scaleX = 100 / imageSize.width;
                            const scaleY = 100 / imageSize.height;
                            
                            const points = block.points.map(point => 
                              point ? `${point[0] * scaleX}% ${point[1] * scaleY}%` : ''
                            ).filter(Boolean).join(', ');

                            if (!points) return null;

                            return (
                              <div
                                key={block.id}
                                className="absolute transition-all duration-300"
                                style={{
                                  left: `${block.bbox.x_min * scaleX}%`,
                                  top: `${block.bbox.y_min * scaleY}%`,
                                  width: `${(block.bbox.x_max - block.bbox.x_min) * scaleX}%`,
                                  height: `${(block.bbox.y_max - block.bbox.y_min) * scaleY}%`,
                                  zIndex: 10
                                }}
                              >
                                {/* Text highlight effect */}
                                <div 
                                  className="absolute inset-0"
                                  style={{
                                    clipPath: points ? `polygon(${points})` : 'none',
                                    background: 'rgba(255, 255, 100, 0.3)',
                                    backdropFilter: 'brightness(1.5) contrast(1.2)',
                                    WebkitBackdropFilter: 'brightness(1.5) contrast(1.2)',
                                    boxShadow: '0 0 10px rgba(255, 255, 0, 0.4)',
                                    border: '1px solid rgba(255, 255, 0, 0.6)',
                                    borderRadius: '2px'
                                  }}
                                />
                              </div>
                            );
                          })}
                        </div>
                      )}

                      {/* Scan Button */}
                      <button
                        onClick={() => {
                          setShowTextScan(!showTextScan);
                        }}
                        className={`absolute bottom-4 right-4 px-4 py-2 rounded-lg flex items-center space-x-2 transition-colors ${
                          showTextScan 
                            ? 'bg-yellow-500 text-gray-900 hover:bg-yellow-400' 
                            : 'bg-blue-500 text-white hover:bg-blue-400'
                        }`}
                      >
                        <MagnifyingGlassIcon className="w-5 h-5" />
                        <span>{showTextScan ? 'Hide Text' : 'Scan Text'}</span>
                      </button>
                    </div>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="space-y-4">
                        <div className="p-4 bg-gray-800/50 rounded-lg">
                          <h4 className="font-medium text-blue-400 mb-2">Image Analysis</h4>
                          {selectedImage.faces.length > 0 && (
                            <p className="text-white text-sm mb-2">
                              👤 {selectedImage.faces.length} faces detected
                            </p>
                          )}
                          <p className="text-white text-sm mb-2">
                            🏷️ Scene: {selectedImage.scene_classification.scene_type}
                          </p>
                          {selectedImage.objects.length > 0 && (
                            <p className="text-white text-sm">
                              📦 Objects: {selectedImage.objects.join(', ')}
                            </p>
                          )}
                        </div>
                      </div>

                      <div className="space-y-4">
                        <div className="p-4 bg-gray-800/50 rounded-lg">
                          <h4 className="font-medium text-purple-400 mb-2">Image Details</h4>
                          {selectedImage.metadata.date_taken && (
                            <p className="text-white text-sm mb-2">
                              <CalendarIcon className="w-4 h-4 inline mr-2" />
                              {selectedImage.metadata.date_taken}
                            </p>
                          )}
                          {(selectedImage.metadata.camera_make || selectedImage.metadata.camera_model) && (
                            <p className="text-white text-sm mb-2">
                              <CameraIcon className="w-4 h-4 inline mr-2" />
                              {[selectedImage.metadata.camera_make, selectedImage.metadata.camera_model]
                                .filter(Boolean)
                                .join(' ')}
                            </p>
                          )}
                          <p className="text-white text-sm mb-2">
                            <DocumentIcon className="w-4 h-4 inline mr-2" />
                            {selectedImage.metadata.dimensions} • {selectedImage.metadata.format} • {selectedImage.metadata.file_size}
                          </p>
                          {selectedImage.metadata.focal_length && (
                            <p className="text-white text-sm">
                              📸 {selectedImage.metadata.focal_length}mm
                              {selectedImage.metadata.f_number && ` f/${selectedImage.metadata.f_number}`}
                              {selectedImage.metadata.iso && ` ISO ${selectedImage.metadata.iso}`}
                            </p>
                          )}
                        </div>
                      </div>
                    </div>
                  </>
                )}
              </div>
            </div>
          </Dialog.Panel>
        </div>
      </Dialog>
    </div>
  );
}