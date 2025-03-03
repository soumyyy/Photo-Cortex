import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Dialog } from '@headlessui/react';
import { XMarkIcon, CameraIcon, CalendarIcon } from '@heroicons/react/24/outline';
import Image from 'next/image';

interface FaceAttributes {
  age: string;
  gender: string;
  smile_intensity: number;
  eye_status: string;
  eye_metrics: {
    left_ear: number;
    right_ear: number;
  };
  landmarks: number[][];
}

interface Face {
  bbox: number[];
  score: number;
  face_id?: string;
  face_image?: string;
  attributes?: FaceAttributes;
  embedding?: number[];
}

interface ImageAnalysis {
  filename: string;
  faces: Face[];
  objects: string[];
  scene_classification: {
    scene_type: string;
    confidence: number;
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

export default function ImageGrid({ images }: ImageGridProps): React.ReactElement {
  const [selectedImage, setSelectedImage] = useState<ImageAnalysis | null>(null);
  const [selectedFace, setSelectedFace] = useState<Face | null>(null);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [loadedImages, setLoadedImages] = useState<Set<string>>(new Set());
  const [showFaceOverlay, setShowFaceOverlay] = useState(false);

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

  const handleImageLoad = useCallback((event: React.SyntheticEvent<HTMLImageElement>) => {
    const img = event.target as HTMLImageElement;
    // Use natural dimensions for accurate face box calculations
    setImageSize({ width: img.naturalWidth, height: img.naturalHeight });
  }, []);

  const formatDate = (dateStr: string | null): string => {
    if (!dateStr) return 'Date not available';
    try {
      const date = new Date(dateStr);
      return date.toLocaleString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });
    } catch (e) {
      console.error('Error formatting date:', e);
      return 'Invalid date';
    }
  };

  const formatShutterSpeed = (exposure_time: string | number | null): string => {
    if (!exposure_time) return '';
    const time = typeof exposure_time === 'string' ? parseFloat(exposure_time) : exposure_time;
    if (isNaN(time) || time === 0) return '';
    return `1/${Math.round(1/time)}s`;
  };

  return (
    <div className="min-h-screen bg-[#050505] px-4 py-8 sm:px-6 sm:py-12">
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-6 max-w-8xl mx-auto">
        {images.map((image) => (
          <div 
            key={image.filename}
            ref={(el) => {
              imageRefs.current[image.filename] = el;
            }}
            className="relative group cursor-pointer rounded-xl overflow-hidden bg-[#0a0a0a] aspect-square transition-all duration-500 hover:scale-[1.02] hover:shadow-2xl hover:shadow-black/40 border border-white/[0.02]"
            data-filename={image.filename}
            style={{ maxWidth: '300px' }}
            onClick={() => setSelectedImage(image)}
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
                  unoptimized={true}
                  onLoad={handleImageLoad}
                  onError={(e) => {
                    console.error(`Failed to load image: ${image.filename}`, e);
                  }}
                />
              </div>
            ) : (
              <div className="w-full h-full flex items-center justify-center">
                <div className="animate-pulse bg-[#111111] w-full h-full" />
              </div>
            )}
            <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/30 to-transparent opacity-0 group-hover:opacity-100 transition-all duration-300 ease-out">
              <div className="absolute bottom-0 left-0 right-0 p-4 transform translate-y-1 group-hover:translate-y-0 transition-transform duration-300">
                <p className="text-sm font-medium text-white/90 truncate mb-1">{image.filename}</p>
                <p className="text-xs text-white/70">{image.scene_classification.scene_type}</p>
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
          setShowFaceOverlay(false);
          setSelectedFace(null);
        }}
        className="relative z-50"
      >
        <div className="fixed inset-0 bg-black/90 backdrop-blur-sm" aria-hidden="true" />
        
        <div className="fixed inset-0 flex items-center justify-center p-4">
          <Dialog.Panel className="mx-auto max-w-6xl w-full bg-[#0a0a0a] rounded-2xl shadow-2xl border border-white/[0.02] backdrop-blur-xl">
            <div className="relative">
              <button
                onClick={() => {
                  setSelectedImage(null);
                  setShowFaceOverlay(false);
                  setSelectedFace(null);
                }}
                className="absolute top-4 right-4 z-20 p-2.5 rounded-full bg-black/20 hover:bg-black/40 backdrop-blur-md transition-all duration-300 hover:scale-110 border border-white/[0.05] hover:border-white/[0.1]"
              >
                <XMarkIcon className="w-6 h-6 text-white" />
              </button>
              
              <div className="p-8">
                {selectedImage && (
                  <div className="space-y-8">
                    <div className="relative rounded-2xl overflow-hidden border border-white/5">
                      <div className="flex">
                        {/* Left side - Image */}
                        <div className="flex-1 relative" style={{ aspectRatio: '16/9' }}>
                          <div className="relative w-full h-full">
                            <Image
                              src={`http://localhost:8000/images/${encodeURIComponent(selectedImage.filename)}`}
                              alt={selectedImage.filename}
                              className="object-contain"
                              fill
                              sizes="100vw"
                              priority={true}
                              quality={90}
                              unoptimized={true}
                              onLoad={handleImageLoad}
                              onError={(e) => {
                                console.error(`Failed to load modal image: ${selectedImage.filename}`, e);
                              }}
                            />
                            {/* Face detection overlay */}
                            {showFaceOverlay && selectedImage.faces?.map((face, index) => {
                              // Calculate relative coordinates based on natural image dimensions
                              const relativeWidth = ((face.bbox[2] - face.bbox[0]) / imageSize.width);
                              const relativeHeight = ((face.bbox[3] - face.bbox[1]) / imageSize.height);
                              const relativeX = (face.bbox[0] / imageSize.width);
                              const relativeY = (face.bbox[1] / imageSize.height);

                              return (
                                <div
                                  key={index}
                                  className={`absolute border-2 ${
                                    selectedFace === face ? 'border-blue-400' : 'border-green-400'
                                  } transition-colors cursor-pointer`}
                                  style={{
                                    left: `${relativeX * 100}%`,
                                    top: `${relativeY * 100}%`,
                                    width: `${relativeWidth * 100}%`,
                                    height: `${relativeHeight * 100}%`,
                                    transform: 'translate(0%, 0%)'  // Ensure no additional translation
                                  }}
                                  onMouseEnter={() => setSelectedFace(face)}
                                  onMouseLeave={() => setSelectedFace(null)}
                                />
                              );
                            })}
                          </div>
                        </div>

                        {/* Right side - Info panels */}
                        <div className="w-96 ml-4 flex flex-col min-h-[600px]">
                          <h3 className="text-xl font-semibold text-white mb-4">Details</h3>
                          
                          <div className="space-y-4 flex-1">
                            {/* Face Attributes Panel */}
                            <div className="bg-white/10 backdrop-blur-md rounded-lg p-4 shadow-lg h-[200px]">
                              <div className="flex items-center justify-between mb-4">
                                <h3 className="text-lg font-semibold text-white">Face Attributes</h3>
                                <button
                                  onClick={() => setShowFaceOverlay(!showFaceOverlay)}
                                  className={`px-3 py-1.5 rounded-lg transition-all ${
                                    showFaceOverlay
                                      ? 'bg-green-500/20 text-green-400'
                                      : 'bg-white/5 text-white/60 hover:text-white'
                                  }`}
                                >
                                  {showFaceOverlay ? 'Hide Faces' : 'Show Faces'}
                                </button>
                              </div>
                              {selectedFace && selectedFace.attributes && (
                                <div className="mt-4 space-y-2">
                                  <h4 className="text-sm font-medium text-white/80">Face Attributes</h4>
                                  <div className="grid grid-cols-2 gap-4 text-sm text-white/60">
                                    <div>
                                      <p>Age: <span className={`
                                        ${selectedFace.attributes.age === 'young' ? 'text-blue-400' : ''}
                                        ${selectedFace.attributes.age === 'old' ? 'text-purple-400' : ''}
                                        ${selectedFace.attributes.age === 'unknown' ? 'text-white/40' : ''}
                                      `}>
                                        {selectedFace.attributes.age.charAt(0).toUpperCase() + selectedFace.attributes.age.slice(1)}
                                      </span></p>
                                      <p>Gender: {selectedFace.attributes.gender}</p>
                                      <p>Smile: {Math.round(selectedFace.attributes.smile_intensity * 100)}%</p>
                                    </div>
                                    <div>
                                      <p>Eyes: <span className={`
                                        ${selectedFace.attributes.eye_status === 'open' ? 'text-green-400' : ''}
                                        ${selectedFace.attributes.eye_status === 'partially open' ? 'text-yellow-400' : ''}
                                        ${selectedFace.attributes.eye_status === 'closed' ? 'text-red-400' : ''}
                                        ${selectedFace.attributes.eye_status === 'unknown' ? 'text-white/40' : ''}
                                      `}>
                                        {selectedFace.attributes.eye_status.charAt(0).toUpperCase() + selectedFace.attributes.eye_status.slice(1)}
                                      </span></p>
                                      <p>Confidence: {Math.round(selectedFace.score * 100)}%</p>
                                    </div>
                                  </div>
                                </div>
                              )}
                            </div>

                            {/* Image Analysis Panel */}
                            <div className="bg-white/10 backdrop-blur-md rounded-lg p-4 shadow-lg">
                              <h3 className="text-lg font-semibold mb-3 text-white">Image Analysis</h3>
                              <div className="space-y-2">
                                <p className="text-white/90">Scene: {selectedImage.scene_classification.scene_type}</p>
                                <p className="text-white/90">Objects: {selectedImage.objects.join(', ')}</p>
                              </div>
                            </div>

                            {/* Image Details Panel */}
                            <div className="bg-white/10 backdrop-blur-md rounded-lg p-4 shadow-lg">
                              <h3 className="text-lg font-semibold mb-3 text-white">Image Details</h3>
                              <div className="space-y-2">
                                <div className="mt-4 grid grid-cols-2 gap-4">
                                  <div className="space-y-2">
                                    <div className="flex items-center space-x-2 text-white/60">
                                      <CalendarIcon className="w-4 h-4" />
                                      <span className="text-sm">{formatDate(selectedImage.metadata.date_taken)}</span>
                                    </div>
                                    {selectedImage.metadata.camera_make && (
                                      <div className="flex items-center space-x-2 text-white/60">
                                        <CameraIcon className="w-4 h-4" />
                                        <span className="text-sm">
                                          {selectedImage.metadata.camera_make} {selectedImage.metadata.camera_model}
                                        </span>
                                      </div>
                                    )}
                                  </div>
                                  <div className="space-y-2 text-sm text-white/60">
                                    {selectedImage.metadata.focal_length && (
                                      <p>Focal Length: {selectedImage.metadata.focal_length}mm</p>
                                    )}
                                    {selectedImage.metadata.f_number && (
                                      <p>f/{selectedImage.metadata.f_number}</p>
                                    )}
                                    {selectedImage.metadata.exposure_time && (
                                      <p>{formatShutterSpeed(selectedImage.metadata.exposure_time)}</p>
                                    )}
                                    {selectedImage.metadata.iso && (
                                      <p>ISO {selectedImage.metadata.iso}</p>
                                    )}
                                  </div>
                                </div>
                                <p className="text-white/90">Dimensions: {selectedImage.metadata.dimensions}</p>
                                <p className="text-white/90">Format: {selectedImage.metadata.format}</p>
                                <p className="text-white/90">Size: {selectedImage.metadata.file_size}</p>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </Dialog.Panel>
        </div>
      </Dialog>
    </div>
  );
}