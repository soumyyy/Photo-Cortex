'use client';

import { useEffect, useRef } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

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

  useEffect(() => {
    let mounted = true;
    let initTimer: NodeJS.Timeout;

    const initMap = () => {
      if (!mounted || !mapContainerRef.current) return;

      // Debug logs
      console.log('Map initialization starting');
      console.log('Config:', config);
      console.log('Single location:', singleLocation);
      console.log('Container dimensions:', {
        width: mapContainerRef.current.clientWidth,
        height: mapContainerRef.current.clientHeight
      });

      // Cleanup existing map and markers
      if (mapRef.current) {
        if (markersRef.current) {
          markersRef.current.clearLayers();
          mapRef.current.removeLayer(markersRef.current);
        }
        mapRef.current.remove();
        mapRef.current = null;
      }

      // Ensure container has dimensions
      const container = mapContainerRef.current;
      if (container.clientHeight === 0 || container.clientWidth === 0) {
        console.warn('Map container has no dimensions, retrying...');
        initTimer = setTimeout(initMap, 100);
        return;
      }

      try {
        // Light map style with attribution
        const mapStyle = 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png';

        // Initialize map
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

        console.log('Map created:', map);
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
          console.log('Adding marker at coordinates:', coords);
          
          const marker = L.marker(coords, { 
            icon,
            zIndexOffset: 1000 // Ensure marker is above other elements
          });
          marker.addTo(markerGroup);
          
          // Set a closer zoom and ensure proper centering
          map.setView(coords, 16, {
            animate: true,
            duration: 1
          });
          
          // Double-check centering after a short delay
          setTimeout(() => {
            if (mounted && mapRef.current) {
              mapRef.current.invalidateSize();
              mapRef.current.setView(coords, 16, {
                animate: true,
                duration: 0.5
              });
            }
          }, 250);
          
          console.log('Map view set to coordinates');
        } else if (images) {
          // Add markers for multiple images with GPS data
          const imagesWithGps = images.filter((img): img is ImageData & { metadata: { gps: NonNullable<ImageData['metadata']['gps']> } } => 
            img.metadata.gps !== null
          );

          imagesWithGps.forEach(image => {
            const coords = [image.metadata.gps.latitude, image.metadata.gps.longitude];
            console.log('Adding marker at coordinates:', coords);
            const marker = L.marker(coords as [number, number], { icon });
            marker.addTo(markerGroup);
          });

          // Fit bounds for multiple locations
          if (config.bounds) {
            map.fitBounds(config.bounds);
            console.log('Map bounds set');
          }
        }

        // Force a resize after initialization
        requestAnimationFrame(() => {
          if (mounted && mapRef.current) {
            mapRef.current.invalidateSize();
            console.log('Map size invalidated');
          }
        });

      } catch (error) {
        console.error('Error initializing map:', error);
      }
    };

    // Start initialization
    initTimer = setTimeout(initMap, 100);

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
  }, [config, images, singleLocation]);

  return <div ref={mapContainerRef} className="absolute inset-0" />;
};

export default MapClient;