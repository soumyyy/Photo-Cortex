import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Dialog } from '@headlessui/react';
import { XMarkIcon, CameraIcon, CalendarIcon, MapPinIcon, DocumentTextIcon } from '@heroicons/react/24/outline';
import Image from 'next/image';
import ImageMap from './ImageMap';
import { ImageAnalysis, Face } from '../types/ImageAnalysis';
import toast, { Toaster } from 'react-hot-toast';

// Simple spinner component
const Spinner = ({ className = "" }: { className?: string }) => (
  <svg 
    className={`animate-spin ${className}`} 
    xmlns="http://www.w3.org/2000/svg" 
    fill="none" 
    viewBox="0 0 24 24"
  >
    <circle 
      className="opacity-25" 
      cx="12" 
      cy="12" 
      r="10" 
      stroke="currentColor" 
      strokeWidth="4"
    />
    <path 
      className="opacity-75" 
      fill="currentColor" 
      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
    />
  </svg>
);

interface ImageGridProps {
  images: ImageAnalysis[];
}
interface TextBlock {
  text: string;
  bbox: [number, number, number, number]; // [x_min, y_min, x_max, y_max]
}

const ImageGrid: React.FC<ImageGridProps> = ({ images }) => {
  const [selectedImage, setSelectedImage] = useState<ImageAnalysis | null>(null);
  const [selectedFace, setSelectedFace] = useState<Face | null>(null);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [loadedImages, setLoadedImages] = useState<Set<string>>(new Set());
  const [loadingStates, setLoadingStates] = useState<{ [key: string]: boolean }>({});
  const [showFaceOverlay, setShowFaceOverlay] = useState(false);
  const [config, setConfig] = useState<{ API_BASE_URL: string }>({ API_BASE_URL: 'http://localhost:8000' });
  const [isScanning, setIsScanning] = useState<string | null>(null); // Track scanning state by image filename
  const [scanResults, setScanResults] = useState<{ [key: string]: any }>({});  // Store results by image filename
  const [showTextOverlays, setShowTextOverlays] = useState<{ [key: string]: boolean }>({}); // Track overlay state by image filename
  const [debugMode, setDebugMode] = useState(false);

  const observerRef = useRef<IntersectionObserver | null>(null);
  const imageRefs = useRef<{ [key: string]: HTMLDivElement | null }>({});
  const imageContainerRef = useRef<HTMLDivElement | null>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const previewRef = useRef<HTMLDivElement>(null);

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

  const getImageUrl = (filename: string) => {
    // Ensure the filename is properly encoded
    const encodedFilename = encodeURIComponent(filename);
    return `${config.API_BASE_URL}/images/${encodedFilename}`;
  };

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

  const handleScanClick = async (image: ImageAnalysis) => {
    if (isScanning) return;
    
    const imageFilename = image.filename;
    
    // Toggle text overlay off if it's already on
    if (showTextOverlays[imageFilename]) {
      setShowTextOverlays(prev => ({ ...prev, [imageFilename]: false }));
      return;
    }
    
    setIsScanning(imageFilename);
    
    try {
      const response = await fetch(`${config.API_BASE_URL}/scan-text`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          filename: imageFilename
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to scan text');
      }
      
      const result = await response.json();
      if (!result || !result.text_blocks || !Array.isArray(result.text_blocks)) {
        throw new Error('Invalid response format from server');
      }
      
      setScanResults(prev => ({ ...prev, [imageFilename]: result }));
      setShowTextOverlays(prev => ({ ...prev, [imageFilename]: true }));
      toast.success(`Found ${result.text_blocks.length} text regions in ${result.processing_time.toFixed(2)}s`);
    } catch (error: unknown) {
      console.error('Error scanning text:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to scan text');
    } finally {
      setIsScanning(null);
    }
  };

  const renderTextHighlights = (image: ImageAnalysis) => {
    const imageFilename = image.filename;
    if (!showTextOverlays[imageFilename] || !scanResults[imageFilename]?.text_blocks || !imageRef.current) {
      return null;
    }

    const imageRect = imageRef.current.getBoundingClientRect();
    const scaleX = imageRect.width / imageRef.current.naturalWidth;
    const scaleY = imageRect.height / imageRef.current.naturalHeight;

    return (
      <div className="highlight-container">
        {scanResults[imageFilename].text_blocks.map((block: any, index: number) => {
          const { bbox } = block;
          const style = {
            left: `${bbox.x_min * scaleX}px`,
            top: `${bbox.y_min * scaleY}px`,
            width: `${(bbox.x_max - bbox.x_min) * scaleX}px`,
            height: `${(bbox.y_max - bbox.y_min) * scaleY}px`,
            animationDelay: `${index * 0.05}s`, // Stagger the animations
          };

          return (
            <div
              key={index}
              className="text-region-highlight group"
              style={style}
              onClick={(e) => {
                e.stopPropagation();
                navigator.clipboard.writeText(block.text);
                toast.success('Text copied to clipboard!');
              }}
            >
              <span className="tooltip">{block.text}</span>
            </div>
          );
        })}
      </div>
    );
  };

  const renderFaceOverlays = (image: ImageAnalysis) => {
    if (!showFaceOverlay || !image.faces || !imageRef.current) {
      return null;
    }

    console.log('Face data:', image.faces);

    const imageRect = imageRef.current.getBoundingClientRect();
    const scaleX = imageRect.width / imageRef.current.naturalWidth;
    const scaleY = imageRect.height / imageRef.current.naturalHeight;

    return (
      <div className="highlight-container">
        {image.faces.map((face, index) => {
          // bbox comes as [x1, y1, x2, y2] (top-left and bottom-right coordinates)
          const [x1, y1, x2, y2] = face.bbox;
          console.log(`Face ${index}:`, { bbox: face.bbox, attributes: face.attributes });
          
          const style = {
            left: `${x1 * scaleX}px`,
            top: `${y1 * scaleY}px`,
            width: `${(x2 - x1) * scaleX}px`,
            height: `${(y2 - y1) * scaleY}px`,
            animationDelay: `${index * 0.05}s`,
            borderColor: selectedFace === face ? 'rgba(0, 255, 0, 0.8)' : 'rgba(0, 255, 0, 0.5)',
            backgroundColor: selectedFace === face ? 'rgba(0, 255, 0, 0.2)' : 'rgba(0, 255, 0, 0.1)',
          };

          return (
            <div
              key={index}
              className={`face-region-highlight group ${selectedFace === face ? 'selected' : ''}`}
              style={style}
              onClick={(e) => {
                e.stopPropagation();
                console.log('Face clicked:', face);
                // Update selected face and selected image
                setSelectedFace(face);
                setSelectedImage(image);
              }}
            />
          );
        })}
      </div>
    );
  };

  useEffect(() => {
    console.log('Selected face changed:', selectedFace);
  }, [selectedFace]);

  useEffect(() => {
    if (selectedImage) {
      console.log('Selected image faces:', selectedImage.faces);
    }
    setSelectedFace(null);  // Reset selected face when image changes
  }, [selectedImage]);

  const ImageSkeleton = () => (
    <div className="relative aspect-square w-full h-full overflow-hidden rounded-lg bg-[#0a0a0a]">
      <div className="absolute inset-0 animate-pulse">
        <div className="h-full w-full bg-gradient-to-b from-[#111111] to-[#0a0a0a]" />
      </div>
      <div className="absolute bottom-0 left-0 right-0 p-4">
        <div className="h-4 w-2/3 bg-[#111111] rounded animate-pulse mb-2" />
        <div className="h-3 w-1/3 bg-[#111111] rounded animate-pulse opacity-70" />
      </div>
    </div>
  );

  const renderImagePreview = (image: ImageAnalysis) => {
    const imageUrl = getImageUrl(image.filename);
    const isLoading = loadingStates[image.filename] ?? true;

    return (
      <div 
        ref={el => {
          imageRefs.current[image.filename] = el;
        }}
        data-filename={image.filename}
        className="relative aspect-square w-full h-full overflow-hidden rounded-lg bg-gray-900/20"
      >
        {isLoading && <ImageSkeleton />}
        <Image
          src={imageUrl}
          alt={image.filename}
          fill
          sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
          className={`object-cover transition-all duration-300 z-10 ${isLoading ? 'opacity-0' : 'opacity-100'}`}
          loading="lazy"
          unoptimized
          onLoadingComplete={() => {
            setLoadingStates(prev => ({ ...prev, [image.filename]: false }));
          }}
          onError={(e) => {
            console.error(`Failed to load image: ${image.filename}`, e);
            setLoadingStates(prev => ({ ...prev, [image.filename]: false }));
          }}
        />
      </div>
    );
  };

  const handleKeyPress = useCallback((e: KeyboardEvent) => {
    if (!selectedImage) return;
    
    const currentIndex = images.findIndex(img => img.filename === selectedImage.filename);
    if (e.key === 'ArrowLeft' && currentIndex > 0) {
      setSelectedImage(images[currentIndex - 1]);
      setShowFaceOverlay(false);
      setSelectedFace(null);
    } else if (e.key === 'ArrowRight' && currentIndex < images.length - 1) {
      setSelectedImage(images[currentIndex + 1]);
      setShowFaceOverlay(false);
      setSelectedFace(null);
    } else if (e.key === 'Escape') {
      setSelectedImage(null);
      setShowFaceOverlay(false);
      setSelectedFace(null);
    }
  }, [selectedImage, images]);

  useEffect(() => {
    document.addEventListener('keydown', handleKeyPress);
    return () => {
      document.removeEventListener('keydown', handleKeyPress);
    };
  }, [handleKeyPress]);

  return (
    <div className="min-h-screen bg-[#050505] px-4 py-8 sm:px-6 sm:py-12">
      <Toaster />
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4 auto-rows-fr">
        {images.map((image) => (
          <div 
            key={image.filename}
            className="relative group cursor-pointer rounded-xl overflow-hidden bg-[#0a0a0a] aspect-square transition-all duration-500 hover:scale-[1.02] hover:shadow-2xl hover:shadow-black/40 border border-white/[0.02] min-w-[150px]"
            onClick={() => setSelectedImage(image)}
          >
            {renderImagePreview(image)}
            <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/30 to-transparent opacity-0 group-hover:opacity-100 transition-all duration-300 z-20">
              <div className="absolute bottom-0 left-0 right-0 p-4 transform translate-y-1 group-hover:translate-y-0 transition-transform duration-300">
                <p className="text-sm font-medium text-white/90 mb-1">{image.filename}</p>
                <p className="text-xs text-white/70">{image.scene_classification?.scene_type}</p>
                {image.cached && (
                  <span className="text-blue-300 text-xs">
                    Cached {image.analysis_date && `• ${formatDate(image.analysis_date)}`}
                  </span>
                )}
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
          <Dialog.Panel className="mx-auto max-w-7xl w-full glass-panel shadow-2xl overflow-hidden">
            {selectedImage && (
              <div className="flex h-[85vh]">
                {/* Left side - Image */}
                <div className="flex-1 relative flex flex-col p-4">
                  {/* Top Bar with Glass Effect */}
                  <div className="flex items-center justify-between mb-4 p-3 bg-[#0a0a0a]/80 rounded-lg border border-white/[0.05]">
                    <div className="flex items-center gap-3">
                      <h2 className="text-lg font-medium text-white/90 truncate">{selectedImage.filename}</h2>
                      {selectedImage.scene_classification && (
                        <span className={`px-2 py-1 rounded-lg text-xs ${getSceneTypeColor(selectedImage.scene_classification.scene_type)}`}>
                          {selectedImage.scene_classification.scene_type}
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-2">
                      {/* {renderImageActions(selectedImage)} */}
                    </div>
                  </div>
                  
                  {/* Image Container */}
                  <div className=" bg-[#0a0a0a] flex-1 relative flex items-center justify-center glass-panel p-4">
                    <div 
                      ref={previewRef} 
                      className="relative w-auto h-auto max-w-full"
                    >
                      <div className="relative">
                        <Image
                          ref={imageRef}
                          src={getImageUrl(selectedImage.filename)}
                          alt={selectedImage.filename}
                          width={1920}
                          height={1080}
                          className={`max-h-[75vh] w-auto h-auto object-contain transition-all duration-500 selected-image ${
                            showTextOverlays[selectedImage.filename] ? 'image-dimmed' : ''
                          }`}
                          onLoad={handleImageLoad}
                          priority
                          unoptimized={true}
                        />
                        {renderTextHighlights(selectedImage)}
                        {renderFaceOverlays(selectedImage)}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Right side - Details */}
                <div className="w-80 border-l border-white/[0.05] p-6 overflow-y-auto">
                  <div className="space-y-6">
                    {/* Image Info */}
                    <div>
                      <h3 className="text-lg font-medium text-white/90 mb-4">Image Details</h3>
                      <div className="space-y-4">
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

                        {/* Objects Detected */}
                        {selectedImage.objects && selectedImage.objects.length > 0 && (
                          <div className="space-y-2">
                            <h4 className="text-sm font-medium text-white/70">Objects Detected</h4>
                            <div className="flex flex-wrap gap-2">
                              {selectedImage.objects.map((object, idx) => (
                                <span
                                  key={idx}
                                  className="px-2 py-1 text-xs font-medium rounded-md bg-amber-500/20 text-amber-400 border border-amber-500/30"
                                >
                                  {object}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Face Detection */}
                        {selectedImage.faces.length > 0 && (
                          <div className="space-y-3">
                            <div className="flex items-center justify-between">
                              <span className="text-sm text-white/70">
                                Faces Detected ({selectedImage.faces.length})
                              </span>
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
                            {/* <div className="h-[140px] bg-white/5 rounded-lg p-3 text-sm">
                              {selectedFace && selectedFace.attributes ? (
                                <div className="space-y-2 animate-fadeIn">
                                  <div className="flex items-center justify-between text-white/80">
                                    <span>Age</span>
                                    <span className="capitalize">{selectedFace.attributes.age || 'Unknown'}</span>
                                  </div>
                                  <div className="flex items-center justify-between text-white/80">
                                    <span>Gender</span>
                                    <span className="capitalize">{selectedFace.attributes.gender || 'Unknown'}</span>
                                  </div>
                                  <div className="flex items-center justify-between text-white/80">
                                    <span>Smile</span>
                                    <span>
                                      {selectedFace.attributes.smile_intensity !== undefined 
                                        ? `${Math.round(selectedFace.attributes.smile_intensity * 100)}%` 
                                        : 'Unknown'}
                                    </span>
                                  </div>
                                  <div className="flex items-center justify-between text-white/80">
                                    <span>Eyes</span>
                                    <span className="capitalize">{selectedFace.attributes.eye_status || 'Unknown'}</span>
                                  </div>
                                </div>
                              ) : (
                                <div className="h-full flex items-center justify-center text-white/40 text-sm">
                                  {showFaceOverlay ? 'Click on a face to see details' : 'Enable face detection to see details'}
                                </div>
                              )}
                            </div> */}
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

                        {showTextOverlays[selectedImage.filename] && scanResults[selectedImage.filename] && (
                          <div className="mt-6 glass-panel p-4 rounded-xl animate-fadeIn">
                            <h4 className="text-sm font-medium text-white/80 mb-2 flex items-center">
                              <DocumentTextIcon className="w-4 h-4 mr-2" />
                              OCR Results
                            </h4>
                            
                            <div className="text-sm text-white/70 mb-4">
                              {scanResults[selectedImage.filename].combined_text}
                            </div>
                            
                            {Object.entries(scanResults[selectedImage.filename].entities).some(([_, values]) => (values as string[]).length > 0) && (
                              <div className="mt-4">
                                <h5 className="text-xs font-medium text-white/60 mb-2">Detected Entities</h5>
                                <div className="space-y-2">
                                  {Object.entries(scanResults[selectedImage.filename].entities).map(([type, values]) => {
                                    if ((values as string[]).length === 0) return null;
                                    
                                    return (
                                      <div key={type} className="text-xs">
                                        <span className="text-white/50">{type}:</span>
                                        <div className="flex flex-wrap gap-2 mt-1">
                                          {(values as string[]).map((value, idx) => (
                                            <span key={idx} className="bg-gray-700/50 px-2 py-1 rounded text-white/80">
                                              {value}
                                            </span>
                                          ))}
                                        </div>
                                      </div>
                                    );
                                  })}
                                </div>
                              </div>
                            )}
                            
                            <div className="mt-4 text-xs text-white/40">
                              Processing time: {scanResults[selectedImage.filename].processing_time.toFixed(2)}s
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
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

export default ImageGrid;