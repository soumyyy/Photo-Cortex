'use client';

import { useEffect, useState } from 'react';
import dynamic from 'next/dynamic';

// Create a client-side only map component
const ClientSideMap = dynamic(
  () => import('./MapClient').then((mod) => mod.default),
  { ssr: false }
);

interface ImageMapProps {
  images: Array<{
    filename: string;
    metadata: {
      gps: {
        latitude: number;
        longitude: number;
      } | null;
    };
  }>;
}

interface MapBounds {
  center: [number, number];
  bounds: [[number, number], [number, number]];
}

const ImageMap = ({ images }: ImageMapProps) => {
  const [mapConfig, setMapConfig] = useState<MapBounds>({
    center: [0, 0],
    bounds: [[0, 0], [0, 0]]
  });

  useEffect(() => {
    const filtered = images.filter(img => img.metadata.gps !== null);
    if (filtered.length > 0) {
      // Calculate center
      const lats = filtered.map(img => img.metadata.gps!.latitude);
      const lngs = filtered.map(img => img.metadata.gps!.longitude);
      
      const minLat = Math.min(...lats);
      const maxLat = Math.max(...lats);
      const minLng = Math.min(...lngs);
      const maxLng = Math.max(...lngs);
      
      setMapConfig({
        center: [(minLat + maxLat) / 2, (minLng + maxLng) / 2],
        bounds: [[minLat, minLng], [maxLat, maxLng]]
      });
    }
  }, [images]);

  if (images.filter(img => img.metadata.gps !== null).length === 0) {
    return (
      <div className="h-96 flex items-center justify-center text-gray-500">
        No images with GPS data available
      </div>
    );
  }

  return (
    <div className="h-[600px] w-full rounded-lg overflow-hidden shadow-lg">
      <ClientSideMap 
        images={images} 
        center={mapConfig.center}
        bounds={mapConfig.bounds}
      />
    </div>
  );
};

export default ImageMap;