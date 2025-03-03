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

interface MapClientProps {
  images: ImageData[];
  center: [number, number];
  bounds: [[number, number], [number, number]];
}

const MapClient = ({ images, center, bounds }: MapClientProps) => {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<L.Map | null>(null);

  useEffect(() => {
    if (!mapContainerRef.current) return;

    // Dark map style with attribution
    const mapStyle = 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png';

    // Initialize map if not already initialized
    if (!mapRef.current) {
      mapRef.current = L.map(mapContainerRef.current, {
        zoomControl: false,
        center: center,
        layers: [
          L.tileLayer(mapStyle, {
            maxZoom: 19,
            attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          })
        ]
      });

      // Add zoom control with custom styling
      L.control.zoom({
        position: 'bottomright'
      }).addTo(mapRef.current);

      // Create custom icon for markers
      const icon = L.divIcon({
        className: 'custom-marker',
        html: '<div class="marker-inner"></div>',
        iconSize: [12, 12],
        iconAnchor: [6, 6]
      });

      // Fit bounds if provided
      if (bounds) {
        mapRef.current.fitBounds(bounds);
      }
      // Create layer group for markers
      const markerGroup = L.layerGroup().addTo(mapRef.current);

      // Add markers for images with GPS data
      const imagesWithGps = images.filter((img): img is ImageData & { metadata: { gps: NonNullable<ImageData['metadata']['gps']> } } => 
        img.metadata.gps !== null
      );

      imagesWithGps.forEach(image => {
        const marker = L.marker(
          [image.metadata.gps.latitude, image.metadata.gps.longitude],
          { icon }
        );
        
        // Create custom popup
        const popupContent = `
          <div class="custom-popup">
            <img src="http://localhost:8000/images/${image.filename}" alt="${image.filename}" class="popup-image" />
            <div class="popup-filename">${image.filename}</div>
          </div>
        `;
        
        const popup = L.popup({
          className: 'custom-popup-container',
          closeButton: false,
          maxWidth: 300,
          minWidth: 200,
        }).setContent(popupContent);

        marker.bindPopup(popup);
        marker.addTo(markerGroup);
      });
    }

    // Add custom styles to the document
    const style = document.createElement('style');
    style.textContent = `
      .custom-marker {
        background: transparent;
      }
      .marker-inner {
        width: 12px;
        height: 12px;
        background: #6366f1;
        border-radius: 50%;
        box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
        backdrop-filter: blur(8px);
      }
      .marker-inner:hover {
        transform: scale(1.2);
        background: #818cf8;
        box-shadow: 0 0 15px rgba(99, 102, 241, 0.4);
      }
      .custom-popup-container {
        background: rgba(17, 24, 39, 0.95);
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(12px);
      }
      .custom-popup {
        padding: 0;
      }
      .popup-image {
        width: 100%;
        height: 200px;
        object-fit: cover;
        border-radius: 12px 12px 0 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
      }
      .popup-filename {
        padding: 12px;
        font-size: 14px;
        color: rgba(255, 255, 255, 0.9);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        background: rgba(17, 24, 39, 0.95);
      }
      .leaflet-popup-content-wrapper {
        padding: 0;
        border-radius: 12px;
        background: transparent;
      }
      .leaflet-popup-content {
        margin: 0;
      }
      .leaflet-popup-tip {
        background: rgba(17, 24, 39, 0.95);
        border: 1px solid rgba(255, 255, 255, 0.1);
      }
      .leaflet-control-zoom {
        border: none !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
        border-radius: 8px !important;
        overflow: hidden;
      }
      .leaflet-control-zoom a {
        background: rgba(17, 24, 39, 0.95) !important;
        color: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(12px);
      }
      .leaflet-control-zoom a:hover {
        background: rgba(31, 41, 55, 0.95) !important;
      }
    `;
    document.head.appendChild(style);

    // Cleanup function
    return () => {
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
      if (style.parentNode === document.head) {
        document.head.removeChild(style);
      }
    };
  }, [images, center, bounds]);

  return <div ref={mapContainerRef} className="h-full w-full rounded-lg overflow-hidden" />;
};

export default MapClient;