'use client';

import React, { useEffect, useState } from 'react';
import dynamic from 'next/dynamic';
import { ImageAnalysis } from '../types/ImageAnalysis';

const MapClient = dynamic(() => import('./MapClient'), { ssr: false });

interface MapBounds {
  center: [number, number];
  bounds: [[number, number], [number, number]];
}

interface ImageMapProps {
  images?: ImageAnalysis[];
  singleLocation?: {
    latitude: number;
    longitude: number;
  };
}

const ImageMap = ({ images, singleLocation }: ImageMapProps) => {
  const [mapConfig, setMapConfig] = useState<MapBounds | null>(null);

  useEffect(() => {
    if (singleLocation) {
      // For single location, create a tight bounds around the point
      const lat = singleLocation.latitude;
      const lng = singleLocation.longitude;
      const newConfig: MapBounds = {
        center: [lat, lng],
        bounds: [
          [lat - 0.0001, lng - 0.0001],
          [lat + 0.0001, lng + 0.0001]
        ]
      };
      setMapConfig(newConfig);
    } else if (images && images.length > 0) {
      const withGps = images.filter(img => img.metadata.gps !== null);
      
      if (withGps.length > 0) {
        const lats = withGps.map(img => img.metadata.gps!.latitude);
        const lngs = withGps.map(img => img.metadata.gps!.longitude);
        
        const minLat = Math.min(...lats);
        const maxLat = Math.max(...lats);
        const minLng = Math.min(...lngs);
        const maxLng = Math.max(...lngs);
        
        const centerLat = (minLat + maxLat) / 2;
        const centerLng = (minLng + maxLng) / 2;
        
        const newConfig: MapBounds = {
          center: [centerLat, centerLng],
          bounds: [[minLat, minLng], [maxLat, maxLng]]
        };
        setMapConfig(newConfig);
      }
    }
  }, [images, singleLocation]);

  // Handle no location data
  if ((!images || images.length === 0) && !singleLocation) {
    return (
      <div className="flex items-center justify-center h-full text-white/60">
        No location data available
      </div>
    );
  }

  return (
    <div className="relative w-full h-full rounded-lg overflow-hidden bg-gray-900" style={{ minHeight: '200px' }}>
      {mapConfig && (
        <div className="absolute inset-0" style={{ width: '100%', height: '100%' }}>
          <MapClient 
            images={images} 
            config={mapConfig}
            singleLocation={singleLocation}
          />
        </div>
      )}
    </div>
  );
};

export default ImageMap;