'use client';

import React, { useState, useEffect } from 'react';
import { Tab } from '@headlessui/react';
import ImageGrid from '@/app/components/ImageGrid';
import PeopleGrid from '@/app/components/PeopleGrid';
import ImageMap from '@/app/components/ImageMap';
import { PhotoIcon, UserGroupIcon, MapIcon } from '@heroicons/react/24/outline';

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
    score: number;
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
    gps: {
      latitude: number;
      longitude: number;
    } | null;
  };
}

export default function Home() {
  const [images, setImages] = useState<ImageAnalysis[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [processedCount, setProcessedCount] = useState(0);
  const [totalImages, setTotalImages] = useState(0);

  useEffect(() => {
    analyzeImages();
  }, []);

  const analyzeImages = async () => {
    setLoading(true);
    setError(null);
    setProgress(0);
    setImages([]);
    setProcessedCount(0);
    setTotalImages(0);

    try {
      const response = await fetch('http://localhost:8000/analyze-folder');
      if (!response.body) throw new Error('No response body');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        const updates = chunk.split('\n').filter(Boolean);
        
        for (const update of updates) {
          const data = JSON.parse(update);
          
          if (data.error) {
            setError(data.error);
            break;
          }
          
          if (data.complete) {
            setImages(data.results.map((result: any) => ({
              ...result,
              text_recognition: result.text_recognition || {
                text_detected: false,
                text_blocks: [],
                total_confidence: 0
              }
            })));
            setProgress(100);
            break;
          } else {
            setProgress(data.progress);
            setProcessedCount(data.current);
            setTotalImages(data.total);
            
            // Handle single image result
            if (data.latest_result) {
              setImages(prev => {
                if (prev.find(img => img.filename === data.latest_result.filename)) {
                  return prev;
                }
                return [...prev, data.latest_result];
              });
            }
          }
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to analyze images');
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen px-4 py-8 sm:px-6 sm:py-12">
      <div className="max-w-screen-2xl mx-auto">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-12 space-y-4 sm:space-y-0">
          <div>
            <h1 className="text-3xl font-semibold text-white/90 mb-2">PhotoCortex</h1>
            <p className="text-sm text-white/60">AI-Powered Image Analysis</p>
          </div>
          {loading && (
            <div className="glass-panel px-4 py-2 rounded-xl text-sm text-white/70">
              Processing {processedCount} of {totalImages} images ({Math.round(progress)}%)
            </div>
          )}
        </div>

        {/* Progress Bar */}
        {loading && (
          <div className="glass-panel p-1 rounded-full mb-12 overflow-hidden">
            <div 
              className="h-1 bg-white/20 rounded-full transition-all duration-500 ease-out relative overflow-hidden"
              style={{ width: `${progress}%` }}
            >
              <div className="absolute inset-0 bg-gradient-to-r from-white/30 via-white/20 to-white/30 animate-shimmer" />
            </div>
          </div>
        )}

        {error && (
          <div className="mb-8 p-4 glass-panel rounded-xl border-red-500/20 text-red-400/90">
            {error}
          </div>
        )}

        <Tab.Group>
          <Tab.List className="flex space-x-4 mb-12 px-1">
            <Tab
              className={({ selected }) =>
                `flex items-center space-x-3 tab-button ${
                  selected ? 'tab-button-active' : ''
                }`
              }
            >
              <PhotoIcon className="w-5 h-5" />
              <span>All Images</span>
            </Tab>
            <Tab
              className={({ selected }) =>
                `flex items-center space-x-3 tab-button ${
                  selected ? 'tab-button-active' : ''
                }`
              }
            >
              <UserGroupIcon className="w-5 h-5" />
              <span>People</span>
            </Tab>
            <Tab
              className={({ selected }) =>
                `flex items-center space-x-3 tab-button ${
                  selected ? 'tab-button-active' : ''
                }`
              }
            >
              <MapIcon className="w-5 h-5" />
              <span>Map View</span>
            </Tab>
          </Tab.List>

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
              <div className="animate-fadeIn">
                <ImageMap images={images} />
              </div>
            </Tab.Panel>
          </Tab.Panels>
        </Tab.Group>
      </div>
    </main>
  );
}