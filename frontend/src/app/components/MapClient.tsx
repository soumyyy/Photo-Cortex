'use client';

import { useEffect, useRef, useState } from 'react';
import dynamic from 'next/dynamic';
import L from 'leaflet';

interface ImageData {
  filename: string;
  metadata: {
    gps: {
      latitude: number;
      longitude: number;
    } | null;
  };
}

interface MapBounds {
  center: [number, number];
  bounds: [[number, number], [number, number]];
}

interface MapClientProps {
  images?: ImageData[];
  config: MapBounds;
  singleLocation?: {
    latitude: number;
    longitude: number;
  };
}

const MapClient = ({ images, config, singleLocation }: MapClientProps) => {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<L.Map | null>(null);
  const markersRef = useRef<L.LayerGroup | null>(null);
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
  }, []);

  useEffect(() => {
    if (!isClient) return;

    let mounted = true;
    let initTimer: NodeJS.Timeout;

    const initMap = () => {
      if (!mounted || !mapContainerRef.current) return;

      // Ensure container has dimensions
      const container = mapContainerRef.current;
      if (container.clientHeight === 0 || container.clientWidth === 0) {
        console.warn('Map container has no dimensions, retrying...');
        initTimer = setTimeout(initMap, 100);
        return;
      }

      // Cleanup existing map and markers
      if (mapRef.current) {
        if (markersRef.current) {
          markersRef.current.clearLayers();
          mapRef.current.removeLayer(markersRef.current);
        }
        mapRef.current.remove();
        mapRef.current = null;
      }

      try {
        // Light map style with attribution
        const mapStyle = 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png';

        // Initialize map with explicit dimensions
        container.style.width = '100%';
        container.style.height = '100%';

        const map = L.map(container, {
          zoomControl: true,
          center: config.center,
          zoom: 13,
          layers: [
            L.tileLayer(mapStyle, {
              maxZoom: 19,
              attribution: ' OpenStreetMap contributors'
            })
          ]
        });

        mapRef.current = map;

        // Add zoom control with custom styling
        L.control.zoom({
          position: 'bottomright'
        }).addTo(map);

        // Create custom icon for markers
        const icon = L.divIcon({
          className: 'custom-marker',
          html: `
            <div class="marker-inner">
              <div class="marker-pulse"></div>
            </div>
          `,
          iconSize: [24, 24],
          iconAnchor: [12, 12]
        });

        // Create layer group for markers
        const markerGroup = L.layerGroup().addTo(map);
        markersRef.current = markerGroup;

        if (singleLocation) {
          // Add single marker
          const coords: [number, number] = [singleLocation.latitude, singleLocation.longitude];
          const marker = L.marker(coords, { 
            icon,
            zIndexOffset: 1000
          });
          marker.addTo(markerGroup);
          
          // Set view with a delay to ensure container is ready
          setTimeout(() => {
            if (mounted && mapRef.current) {
              mapRef.current.invalidateSize();
              mapRef.current.setView(coords, 16, {
                animate: false
              });
            }
          }, 100);
        } else if (images) {
          // Add markers for multiple images with GPS data
          const imagesWithGps = images.filter((img): img is ImageData & { metadata: { gps: NonNullable<ImageData['metadata']['gps']> } } => 
            img.metadata.gps !== null
          );

          imagesWithGps.forEach(image => {
            const coords = [image.metadata.gps.latitude, image.metadata.gps.longitude];
            const marker = L.marker(coords as [number, number], { icon });
            marker.addTo(markerGroup);
          });

          // Fit bounds with a delay
          setTimeout(() => {
            if (mounted && mapRef.current && config.bounds) {
              mapRef.current.invalidateSize();
              mapRef.current.fitBounds(config.bounds, {
                animate: false
              });
            }
          }, 100);
        }

        // Final resize after all operations
        setTimeout(() => {
          if (mounted && mapRef.current) {
            mapRef.current.invalidateSize();
          }
        }, 250);

      } catch (error) {
        console.error('Error initializing map:', error);
      }
    };

    // Start initialization with a delay
    initTimer = setTimeout(initMap, 250);

    return () => {
      mounted = false;
      clearTimeout(initTimer);
      if (mapRef.current) {
        if (markersRef.current) {
          markersRef.current.clearLayers();
          mapRef.current.removeLayer(markersRef.current);
        }
        mapRef.current.remove();
        mapRef.current = null;
      }
    };
  }, [config, images, singleLocation, isClient]);

  return (
    <div 
      ref={mapContainerRef} 
      className="absolute inset-0" 
      style={{ minHeight: '400px' }}
    />
  );
};

// Export with dynamic import to prevent SSR issues
export default dynamic(() => Promise.resolve(MapClient), {
  ssr: false
});