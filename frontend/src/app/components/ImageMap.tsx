'use client';

import React, { useEffect, useState } from 'react';
import dynamic from 'next/dynamic';

const MapClient = dynamic(() => import('./MapClient'), { ssr: false });

interface GPS {
  latitude: number;
  longitude: number;
}

interface ImageMetadata {
  gps: GPS | null;
}

interface ImageAnalysis {
  filename: string;
  metadata: ImageMetadata;
}

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
      console.log('ImageMap: Setting up single location:', singleLocation);
      // Handle single location
      const newConfig: MapBounds = {
        center: [singleLocation.latitude, singleLocation.longitude],
        bounds: [
          [singleLocation.latitude - 0.001, singleLocation.longitude - 0.001],
          [singleLocation.latitude + 0.001, singleLocation.longitude + 0.001]
        ]
      };
      setMapConfig(newConfig);
    } else if (images && images.length > 0) {
      // Handle multiple locations
      const withGps = images.filter(img => img.metadata.gps !== null);
      console.log('ImageMap: Images with GPS:', withGps.length);
      
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
        console.log('ImageMap: Setting map config:', newConfig);
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
    <div className="relative h-[800px] w-full rounded-lg overflow-hidden bg-gray-900">
      {mapConfig && (
        <div className="absolute inset-0" style={{ zIndex: 1 }}>
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