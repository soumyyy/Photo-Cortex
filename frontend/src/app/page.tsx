'use client';

import React, { useState, useEffect } from 'react';
import { Tab } from '@headlessui/react';
import ImageGrid from '@/app/components/ImageGrid';
import PeopleGrid from '@/app/components/PeopleGrid';
import ImageMap from '@/app/components/ImageMap';
import { PhotoIcon, UserGroupIcon } from '@heroicons/react/24/outline';

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
    <main className="min-h-screen bg-gray-900 p-8">
      <div className="max-w-7xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-3xl font-semibold text-white">Image Analysis</h1>
          {loading && (
            <div className="text-sm text-gray-400">
              Analyzing {processedCount} of {totalImages} images ({Math.round(progress)}%)
            </div>
          )}
        </div>

        {/* Progress Bar */}
        {loading && (
          <div className="h-1 w-full bg-gray-800 rounded-full mb-8 overflow-hidden">
            <div 
              className="h-full bg-blue-500 rounded-full transition-all duration-300 ease-out"
              style={{ width: `${progress}%` }}
            />
          </div>
        )}

        {error && (
          <div className="mb-4 p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400">
            {error}
          </div>
        )}

        <Tab.Group>
          <Tab.List className="flex space-x-1 mb-6">
            <Tab className={({ selected }) =>
              `flex items-center px-4 py-2.5 text-sm rounded-lg transition-all outline-none
              ${selected 
                ? 'text-white bg-gray-800 shadow-lg' 
                : 'text-gray-400 hover:text-white hover:bg-gray-800/50'}`
            }>
              <PhotoIcon className="w-5 h-5 mr-2" />
              All Images
            </Tab>
            <Tab className={({ selected }) =>
              `flex items-center px-4 py-2.5 text-sm rounded-lg transition-all outline-none
              ${selected 
                ? 'text-white bg-gray-800 shadow-lg'
                : 'text-gray-400 hover:text-white hover:bg-gray-800/50'}`
            }>
              <UserGroupIcon className="w-5 h-5 mr-2" />
              People
            </Tab>
            <Tab className={({ selected }) =>
              `flex items-center px-4 py-2.5 text-sm rounded-lg transition-all outline-none
              ${selected 
                ? 'text-white bg-gray-800 shadow-lg'
                : 'text-gray-400 hover:text-white hover:bg-gray-800/50'}`
            }>
              Map View
            </Tab>
          </Tab.List>

          <Tab.Panels>
            <Tab.Panel>
              {loading && images.length === 0 ? (
                <div className="flex items-center justify-center py-12">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
                  <span className="ml-3 text-gray-400">Starting analysis...</span>
                </div>
              ) : images.length > 0 ? (
                <ImageGrid images={images} />
              ) : (
                <div className="text-center py-12 text-gray-500">
                  No images found in the backend folder
                </div>
              )}
            </Tab.Panel>
            <Tab.Panel>
              <PeopleGrid />
            </Tab.Panel>
            <Tab.Panel>
              <ImageMap images={images} />
            </Tab.Panel>
          </Tab.Panels>
        </Tab.Group>
      </div>
    </main>
  );
}