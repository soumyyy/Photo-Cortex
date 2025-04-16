'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Tab } from '@headlessui/react';
import ImageGrid from '@/app/components/ImageGrid';
import PeopleGrid from '@/app/components/PeopleGrid';
import ImageMap from '@/app/components/ImageMap';
import { PhotoIcon, UserGroupIcon, MapIcon, ArrowUpTrayIcon } from '@heroicons/react/24/outline';
import { ImageAnalysis } from './types/ImageAnalysis';

interface TabItem {
  name: string;
  icon: React.ForwardRefExoticComponent<React.SVGProps<SVGSVGElement>>;
}

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

const tabs: TabItem[] = [
  { name: 'Photos', icon: PhotoIcon },
  { name: 'People', icon: UserGroupIcon },
  { name: 'Map', icon: MapIcon },
];

export default function Home() {
  const [images, setImages] = useState<ImageAnalysis[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [processedCount, setProcessedCount] = useState(0);
  const [totalImages, setTotalImages] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [config, setConfig] = useState<{ API_BASE_URL: string }>({ API_BASE_URL: 'http://localhost:8000' });
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    // Fetch config from backend
    fetch(`${config.API_BASE_URL}/config`)
      .then(res => res.json())
      .then(data => {
        console.log('Backend config:', data);
        setConfig(data);
      })
      .catch(err => {
        console.error('Failed to fetch backend config:', err);
      });
  }, []);

  useEffect(() => {
    analyzeImages();
  }, [config.API_BASE_URL]);

  const analyzeImages = async () => {
    setLoading(true);
    setError(null);
    setProgress(0);
    setImages([]);
    setProcessedCount(0);
    setTotalImages(0);

    try {
      const response = await fetch(`${config.API_BASE_URL}/analyze-folder`);
      if (!response.body) throw new Error('No response body');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        
        // Keep the last line in the buffer if it's incomplete
        buffer = lines.pop() || '';
        
        for (const line of lines) {
          if (!line.trim()) continue;
          
          try {
            const data = JSON.parse(line);
            
            if (data.error) {
              setError(data.error);
              break;
            }
            
            if (data.complete) {
              console.log('Received complete data:', data.results);
              const processedResults = data.results.map((result: any) => {
                console.log('Processing result:', result.filename, 'Metadata:', result.metadata);
                return {
                  filename: result.filename,
                  metadata: {
                    date_taken: result.metadata?.date_taken || null,
                    camera_make: result.metadata?.camera_make || null,
                    camera_model: result.metadata?.camera_model || null,
                    focal_length: result.metadata?.focal_length || null,
                    exposure_time: result.metadata?.exposure_time || null,
                    f_number: result.metadata?.f_number || null,
                    iso: result.metadata?.iso || null,
                    dimensions: result.metadata?.dimensions || '',
                    format: result.metadata?.format || '',
                    file_size: result.metadata?.file_size || '',
                    gps: result.metadata?.gps || null
                  },
                  faces: result.faces || [],
                  objects: result.objects || [],
                  scene_classification: result.scene_classification || null,
                  text_recognition: result.text_recognition || {
                    text_detected: false,
                    text_blocks: [],
                    total_confidence: 0,
                    categories: [],
                    raw_text: '',
                    language: ''
                  }
                };
              });
              setImages(processedResults);
              setProgress(100);
              break;
            } else {
              setProgress(data.progress);
              setProcessedCount(data.current);
              setTotalImages(data.total);
              
              // Handle single image result
              if (data.latest_result) {
                console.log('Processing latest result:', data.latest_result.filename, 'Metadata:', data.latest_result.metadata);
                const processedResult = {
                  filename: data.latest_result.filename,
                  metadata: {
                    date_taken: data.latest_result.metadata?.date_taken || null,
                    camera_make: data.latest_result.metadata?.camera_make || null,
                    camera_model: data.latest_result.metadata?.camera_model || null,
                    focal_length: data.latest_result.metadata?.focal_length || null,
                    exposure_time: data.latest_result.metadata?.exposure_time || null,
                    f_number: data.latest_result.metadata?.f_number || null,
                    iso: data.latest_result.metadata?.iso || null,
                    dimensions: data.latest_result.metadata?.dimensions || '',
                    format: data.latest_result.metadata?.format || '',
                    file_size: data.latest_result.metadata?.file_size || '',
                    gps: data.latest_result.metadata?.gps || null
                  },
                  faces: data.latest_result.faces || [],
                  objects: data.latest_result.objects || [],
                  scene_classification: data.latest_result.scene_classification || null,
                  text_recognition: data.latest_result.text_recognition || {
                    text_detected: false,
                    text_blocks: [],
                    total_confidence: 0,
                    categories: [],
                    raw_text: '',
                    language: ''
                  }
                };
                setImages(prev => {
                  if (prev.find(img => img.filename === data.latest_result.filename)) {
                    return prev;
                  }
                  return [...prev, processedResult];
                });
              }
            }
          } catch (parseError) {
            console.warn('Failed to parse line:', line);
            continue;
          }
        }
      }
    } catch (err) {
      console.error('Error analyzing images:', err);
      setError(err instanceof Error ? err.message : 'Failed to analyze images');
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      // First upload the file
      const uploadResponse = await fetch(`${config.API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!uploadResponse.ok) {
        throw new Error('Upload failed');
      }

      // Then analyze just this file
      const analyzeResponse = await fetch(`${config.API_BASE_URL}/analyze-image`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ filename: file.name }),
      });

      if (!analyzeResponse.ok) {
        throw new Error('Analysis failed');
      }

      const result = await analyzeResponse.json();
      
      // Add the new image to the existing list
      setImages(prev => {
        const newImage = {
          filename: file.name,
          metadata: result.metadata || {},
          faces: result.faces || [],
          objects: result.objects || [],
          scene_classification: result.scene_classification || null,
        };
        return [...prev, newImage];
      });
      
    } catch (err) {
      setError('Failed to process image');
      console.error('Upload/Analysis error:', err);
    } finally {
      setUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  return (
    <main className="min-h-screen px-4 py-8 sm:px-6 sm:py-12">
      <div className="max-w-screen-2xl mx-auto">
        {/* Header with title and progress */}
        <div className="sticky top-0 z-50 bg-black/80 backdrop-blur-lg px-6 py-4 -mx-6 mb-8">
          <div className="flex items-center justify-between max-w-screen-2xl mx-auto">
            <div className="flex items-center space-x-8">
              <div>
                <h1 className="text-2xl font-semibold text-white/90">PhotoCortex</h1>
                <p className="text-sm text-white/60">Computer Vision Analysis</p>
              </div>
              {loading && (
                <div className="glass-panel px-4 py-2 rounded-xl flex items-center space-x-3">
                  <div className="relative w-4 h-4">
                    <div className="w-full h-full rounded-full border-2 border-t-white/40 animate-spin" />
                  </div>
                  <span className="text-sm text-white/70">
                    {processedCount} of {totalImages} ({Math.round(progress)}%)
                  </span>
                </div>
              )}
            </div>
            <div className="flex items-center space-x-4">
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileUpload}
                className="hidden"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={uploading}
                className={`glass-panel px-4 py-2 rounded-xl flex items-center space-x-2 transition-all duration-200 ${
                  uploading ? 'opacity-50' : 'hover:bg-white/5'
                }`}
              >
                <ArrowUpTrayIcon className="w-5 h-5" />
                <span className="text-sm text-white/90">
                  {uploading ? 'Uploading...' : 'Upload Image'}
                </span>
              </button>
              {loading && (
                <div className="glass-panel p-1 rounded-full overflow-hidden">
                  <div 
                    className="h-1 bg-white/20 rounded-full transition-all duration-500 ease-out relative overflow-hidden"
                    style={{ width: `${progress}%` }}
                  >
                    <div className="absolute inset-0 bg-gradient-to-r from-white/30 via-white/20 to-white/30 animate-shimmer" />
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {error && (
          <div className="mb-8 p-4 glass-panel rounded-xl border-red-500/20 text-red-400/90">
            {error}
          </div>
        )}
        
        <Tab.Group defaultIndex={0}>
          <div className="flex justify-center">
            <Tab.List className="flex space-x-4 mb-12 px-1">
              {tabs.map((tab, index) => (
                <Tab
                  key={index}
                  className={({ selected }) =>
                    `flex items-center space-x-3 tab-button ${
                      selected ? 'tab-button-active' : ''
                    } focus:outline-none focus:ring-2 focus:ring-white/20 focus:ring-offset-2 focus:ring-offset-black rounded-lg`
                  }
                >
                  <tab.icon className="w-5 h-5" />
                  <span>{tab.name}</span>
                </Tab>
              ))}
            </Tab.List>
          </div>

          <Tab.Panels className="focus:outline-none">
            <Tab.Panel className="focus:outline-none">
              <div className="animate-fadeIn">
                {loading && images.length === 0 ? (
                  <div className="flex items-center justify-center py-16">
                    <div className="relative">
                      <div className="w-12 h-12 rounded-full border-2 border-white/10 animate-ping absolute inset-0" />
                      <div className="w-12 h-12 rounded-full border-2 border-t-white/40 animate-spin" />
                    </div>
                    <span className="ml-4 text-white/60">Starting analysis...</span>
                  </div>
                ) : images.length > 0 ? (
                  <div className="w-full">
                    <ImageGrid images={images} />
                  </div>
                ) : (
                  <div className="text-center py-16 text-white/40">
                    No images found in the backend folder
                  </div>
                )}
              </div>
            </Tab.Panel>
            <Tab.Panel className="focus:outline-none">
              <div className="animate-fadeIn">
                <PeopleGrid />
              </div>
            </Tab.Panel>
            <Tab.Panel className="focus:outline-none">
              <div className="animate-fadeIn" style={{ height: 'calc(100vh - 200px)' }}>
                {loading ? (
                  <div className="flex items-center justify-center h-full">
                    <div className="relative">
                      <div className="w-12 h-12 rounded-full border-2 border-white/10 animate-ping absolute inset-0" />
                      <div className="w-12 h-12 rounded-full border-2 border-t-white/40 animate-spin" />
                    </div>
                    <span className="ml-4 text-white/60">Loading map...</span>
                  </div>
                ) : (
                  <div className="h-full w-full rounded-lg overflow-hidden">
                    <ImageMap images={images} />
                  </div>
                )}
              </div>
            </Tab.Panel>
          </Tab.Panels>
        </Tab.Group>
      </div>
    </main>
  );
}