import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Dialog } from '@headlessui/react';
import { XMarkIcon, CameraIcon, CalendarIcon, MapPinIcon } from '@heroicons/react/24/outline';
import Image from 'next/image';
import ImageMap from './ImageMap';
import { ImageAnalysis, Face } from '../types/ImageAnalysis';

interface ImageGridProps {
  images: ImageAnalysis[];
}

export default function ImageGrid({ images }: ImageGridProps): React.ReactElement {
  const [selectedImage, setSelectedImage] = useState<ImageAnalysis | null>(null);
  const [selectedFace, setSelectedFace] = useState<Face | null>(null);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [loadedImages, setLoadedImages] = useState<Set<string>>(new Set());
  const [showFaceOverlay, setShowFaceOverlay] = useState(false);
  const [config, setConfig] = useState<{ API_BASE_URL: string }>({ API_BASE_URL: 'http://localhost:8000' });

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

  useEffect(() => {
    fetch('http://localhost:8000/config')
      .then(res => res.json())
      .then(data => setConfig(data))
      .catch(err => console.error('Failed to fetch config:', err));
  }, []);

  const getImageUrl = (filename: string) => `${config.API_BASE_URL}/image/${encodeURIComponent(filename)}`;

  const handleImageLoad = useCallback((event: React.SyntheticEvent<HTMLImageElement>) => {
    const img = event.target as HTMLImageElement;
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

  const getSceneTypeColor = (sceneType: string): string => {
    const colorMap: { [key: string]: string } = {
      'Nature': 'bg-emerald-500/20 text-emerald-400',
      'City': 'bg-blue-500/20 text-blue-400',
      'Event': 'bg-purple-500/20 text-purple-400',
      'Party': 'bg-pink-500/20 text-pink-400',
      'Food': 'bg-orange-500/20 text-orange-400',
      'Documents': 'bg-gray-500/20 text-gray-400',
      'Receipts': 'bg-yellow-500/20 text-yellow-400',
      'Other': 'bg-white/10 text-white/60',
      'Unknown': 'bg-white/10 text-white/40'
    };

    return colorMap[sceneType] || colorMap['Unknown'];
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
                  src={getImageUrl(image.filename)}
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
                <p className="text-xs text-white/70">{image.scene_classification?.scene_type}</p>
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
        <div className="fixed inset-0 bg-black/95 backdrop-blur-sm" aria-hidden="true" />
        
        <div className="fixed inset-0 flex items-center justify-center p-4">
          <Dialog.Panel className="mx-auto max-w-7xl w-full bg-[#0a0a0a]/80 rounded-3xl shadow-2xl border border-white/[0.02] backdrop-blur-xl overflow-hidden">
            {selectedImage && (
              <div className="flex h-[85vh]">
                {/* Left side - Image */}
                <div className="flex-1 relative">
                  <div className="absolute inset-0">
                    <Image
                      src={getImageUrl(selectedImage.filename)}
                      alt={selectedImage.filename}
                      className="object-contain"
                      fill
                      sizes="100vw"
                      priority={true}
                      quality={90}
                      unoptimized={true}
                      onLoad={handleImageLoad}
                    />
                    {/* Face detection overlay */}
                    {showFaceOverlay && selectedImage.faces?.map((face, index) => {
                      // Calculate relative coordinates based on the container size
                      const imgElement = document.querySelector('.object-contain') as HTMLImageElement;
                      if (!imgElement) return null;

                      // Get actual rendered dimensions
                      const containerWidth = imgElement.clientWidth;
                      const containerHeight = imgElement.clientHeight;
                      
                      // Calculate scale factors
                      const scaleX = containerWidth / imageSize.width;
                      const scaleY = containerHeight / imageSize.height;
                      const scale = Math.min(scaleX, scaleY);

                      // Calculate actual dimensions and position
                      const width = (face.bbox[2] - face.bbox[0]) * scale;
                      const height = (face.bbox[3] - face.bbox[1]) * scale;
                      
                      // Calculate offset for centering
                      const offsetX = (containerWidth - imageSize.width * scale) / 2;
                      const offsetY = (containerHeight - imageSize.height * scale) / 2;

                      // Calculate final position
                      const left = face.bbox[0] * scale + offsetX;
                      const top = face.bbox[1] * scale + offsetY;

                      return (
                        <div
                          key={index}
                          className={`absolute border-2 transition-all duration-200 ${
                            selectedFace === face ? 'border-blue-500 shadow-lg' : 'border-white/40'
                          }`}
                          style={{
                            left: `${left}px`,
                            top: `${top}px`,
                            width: `${width}px`,
                            height: `${height}px`,
                            cursor: 'pointer'
                          }}
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedFace(face);
                          }}
                        />
                      );
                    })}
                  </div>
                </div>

                {/* Right side - Info panel */}
                <div className="w-80 border-l border-white/[0.05] bg-black/20 backdrop-blur-xl p-6 overflow-y-auto">
                  <div className="flex items-center justify-between mb-6">
                    <h3 className="text-lg font-medium text-white/90">{selectedImage.filename}</h3>
                    <button
                      onClick={() => {
                        setSelectedImage(null);
                        setShowFaceOverlay(false);
                        setSelectedFace(null);
                      }}
                      className="p-1.5 rounded-full hover:bg-white/5 transition-colors"
                    >
                      <XMarkIcon className="w-5 h-5 text-white/70" />
                    </button>
                  </div>

                  <div className="space-y-6">
                    {/* Scene Type */}
                    {selectedImage.scene_classification && (
                      <div className="space-y-1.5">
                        <div className={`inline-flex items-center px-2.5 py-1 rounded-full text-sm ${getSceneTypeColor(selectedImage.scene_classification.scene_type)}`}>
                          {selectedImage.scene_classification.scene_type}
                          <span className="ml-1.5 text-xs opacity-80">
                            {Math.round(selectedImage.scene_classification.confidence * 100)}%
                          </span>
                        </div>
                      </div>
                    )}

                    {/* Face Detection */}
                    {selectedImage.faces.length > 0 && (
                      <div className="space-y-3">
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-white/70">Faces Detected</span>
                          <button
                            onClick={() => setShowFaceOverlay(!showFaceOverlay)}
                            className={`text-xs px-2 py-1 rounded-md transition-all ${
                              showFaceOverlay
                                ? 'bg-green-500/20 text-green-400'
                                : 'bg-white/5 text-white/60 hover:bg-white/10'
                            }`}
                          >
                            {showFaceOverlay ? 'Hide' : 'Show'}
                          </button>
                        </div>
                        
                        {/* Fixed height container for face attributes */}
                        <div className="h-[140px] bg-white/5 rounded-lg p-3 text-sm">
                          {selectedFace && selectedFace.attributes ? (
                            <div className="space-y-2 animate-fadeIn">
                              <div className="flex items-center justify-between text-white/80">
                                <span>Age</span>
                                <span>{selectedFace.attributes.age}</span>
                              </div>
                              <div className="flex items-center justify-between text-white/80">
                                <span>Gender</span>
                                <span>{selectedFace.attributes.gender}</span>
                              </div>
                              <div className="flex items-center justify-between text-white/80">
                                <span>Smile</span>
                                <span>{Math.round(selectedFace.attributes.smile_intensity * 100)}%</span>
                              </div>
                              <div className="flex items-center justify-between text-white/80">
                                <span>Eyes</span>
                                <span>{selectedFace.attributes.eye_status}</span>
                              </div>
                            </div>
                          ) : (
                            <div className="h-full flex items-center justify-center text-white/40 text-sm">
                              Hover over a face to see details
                            </div>
                          )}
                        </div>
                      </div>
                    )}

                    {/* Text Recognition Results */}
                    {selectedImage.text_recognition?.text_detected && selectedImage.text_recognition.raw_text && (
                      <div className="mb-6 space-y-4">
                        <div className="space-y-2">
                          <h4 className="text-sm font-medium text-white/70">Detected Text</h4>
                          <div className="bg-white/5 rounded-lg p-4">
                            <p className="text-sm text-white/80 whitespace-pre-wrap break-words">
                              {selectedImage.text_recognition.raw_text}
                            </p>
                          </div>
                        </div>
                        
                        {selectedImage.text_recognition.categories?.length > 0 && (
                          <div className="space-y-2">
                            <h4 className="text-sm font-medium text-white/70">Text Categories</h4>
                            <div className="flex flex-wrap gap-2">
                              {selectedImage.text_recognition.categories.map((category, idx) => (
                                <span
                                  key={idx}
                                  className="px-2 py-1 text-xs font-medium rounded-md bg-indigo-500/20 text-indigo-400 border border-indigo-500/30"
                                >
                                  {category}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                        
                        <div className="flex items-center space-x-2 text-sm text-white/60">
                          <span>Confidence:</span>
                          <div className="flex-1 h-1.5 bg-white/10 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-indigo-500 rounded-full"
                              style={{ width: `${selectedImage.text_recognition.total_confidence * 100}%` }}
                            />
                          </div>
                          <span>{Math.round(selectedImage.text_recognition.total_confidence * 100)}%</span>
                        </div>
                        
                        {selectedImage.text_recognition.language && selectedImage.text_recognition.language !== 'unknown' && (
                          <div className="text-sm text-white/60">
                            <span>Language: </span>
                            <span className="text-white/80">{selectedImage.text_recognition.language}</span>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Camera Info */}
                    {(selectedImage.metadata.camera_make || selectedImage.metadata.camera_model) && (
                      <div className="space-y-2">
                        <h4 className="text-sm font-medium text-white/70">Camera Info</h4>
                        <div className="grid grid-cols-2 gap-2 text-xs text-white/60">
                          {selectedImage.metadata.camera_make && (
                            <div>
                              <span className="block text-white/40">Make</span>
                              <span>{selectedImage.metadata.camera_make}</span>
                            </div>
                          )}
                          {selectedImage.metadata.camera_model && (
                            <div>
                              <span className="block text-white/40">Model</span>
                              <span>{selectedImage.metadata.camera_model}</span>
                            </div>
                          )}
                          {selectedImage.metadata.focal_length && (
                            <div>
                              <span className="block text-white/40">Focal Length</span>
                              <span>{selectedImage.metadata.focal_length}</span>
                            </div>
                          )}
                          {selectedImage.metadata.f_number && (
                            <div>
                              <span className="block text-white/40">F-Stop</span>
                              <span>ƒ/{selectedImage.metadata.f_number}</span>
                            </div>
                          )}
                          {selectedImage.metadata.exposure_time && (
                            <div>
                              <span className="block text-white/40">Shutter</span>
                              <span>{formatShutterSpeed(selectedImage.metadata.exposure_time)}</span>
                            </div>
                          )}
                          {selectedImage.metadata.iso && (
                            <div>
                              <span className="block text-white/40">ISO</span>
                              <span>{selectedImage.metadata.iso}</span>
                            </div>
                          )}
                        </div>
                      </div>
                    )}

                    {/* File Info */}
                    <div className="space-y-2">
                      <h4 className="text-sm font-medium text-white/70">File Info</h4>
                      <div className="grid grid-cols-2 gap-2 text-xs text-white/60">
                        <div>
                          <span className="block text-white/40">Dimensions</span>
                          <span>{selectedImage.metadata.dimensions}</span>
                        </div>
                        <div>
                          <span className="block text-white/40">Format</span>
                          <span>{selectedImage.metadata.format}</span>
                        </div>
                        <div>
                          <span className="block text-white/40">Size</span>
                          <span>{selectedImage.metadata.file_size}</span>
                        </div>
                        {selectedImage.metadata.date_taken && (
                          <div>
                            <span className="block text-white/40">Date</span>
                            <span>{formatDate(selectedImage.metadata.date_taken)}</span>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Location Map */}
                    {selectedImage.metadata.gps && (
                      <div className="space-y-2 pt-4 border-t border-white/[0.05]">
                        <h4 className="text-sm font-medium text-white/70 flex items-center gap-1.5">
                          <MapPinIcon className="w-4 h-4" />
                          Location
                        </h4>
                        <div className="h-[200px] bg-white/5 rounded-lg overflow-hidden">
                          <ImageMap
                            singleLocation={{
                              latitude: selectedImage.metadata.gps.latitude,
                              longitude: selectedImage.metadata.gps.longitude
                            }}
                          />
                        </div>
                        <div className="text-xs text-white/60 mt-1">
                          {selectedImage.metadata.gps.latitude.toFixed(6)}, {selectedImage.metadata.gps.longitude.toFixed(6)}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </Dialog.Panel>
        </div>
      </Dialog>
    </div>
  );
}