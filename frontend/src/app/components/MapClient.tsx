'use client';

import { useEffect, useRef } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

interface MapClientProps {
  images: Array<{
    filename: string;
    metadata: {
      gps: {
        latitude: number;
        longitude: number;
      } | null;
    };
  }>;
  center: [number, number];
  bounds: [[number, number], [number, number]];
}

const MapClient = ({ images, center, bounds }: MapClientProps) => {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<L.Map | null>(null);

  useEffect(() => {
    if (!mapContainerRef.current) return;

    // Custom map style
    const mapStyle = {
      styles: [
        {
          elementType: "geometry",
          stylers: [{ color: "#f5f5f5" }],
        },
        {
          elementType: "labels.icon",
          stylers: [{ visibility: "off" }],
        },
        {
          elementType: "labels.text.fill",
          stylers: [{ color: "#616161" }],
        },
        {
          elementType: "labels.text.stroke",
          stylers: [{ color: "#f5f5f5" }],
        },
        {
          featureType: "administrative.land_parcel",
          elementType: "labels.text.fill",
          stylers: [{ color: "#bdbdbd" }],
        },
        {
          featureType: "poi",
          elementType: "geometry",
          stylers: [{ color: "#eeeeee" }],
        },
        {
          featureType: "poi",
          elementType: "labels.text.fill",
          stylers: [{ color: "#757575" }],
        },
        {
          featureType: "poi.park",
          elementType: "geometry",
          stylers: [{ color: "#e5e5e5" }],
        },
        {
          featureType: "poi.park",
          elementType: "labels.text.fill",
          stylers: [{ color: "#9e9e9e" }],
        },
        {
          featureType: "road",
          elementType: "geometry",
          stylers: [{ color: "#ffffff" }],
        },
        {
          featureType: "road.arterial",
          elementType: "labels.text.fill",
          stylers: [{ color: "#757575" }],
        },
        {
          featureType: "road.highway",
          elementType: "geometry",
          stylers: [{ color: "#dadada" }],
        },
        {
          featureType: "road.highway",
          elementType: "labels.text.fill",
          stylers: [{ color: "#616161" }],
        },
        {
          featureType: "road.local",
          elementType: "labels.text.fill",
          stylers: [{ color: "#9e9e9e" }],
        },
        {
          featureType: "transit.line",
          elementType: "geometry",
          stylers: [{ color: "#e5e5e5" }],
        },
        {
          featureType: "transit.station",
          elementType: "geometry",
          stylers: [{ color: "#eeeeee" }],
        },
        {
          featureType: "water",
          elementType: "geometry",
          stylers: [{ color: "#c9c9c9" }],
        },
        {
          featureType: "water",
          elementType: "labels.text.fill",
          stylers: [{ color: "#9e9e9e" }],
        },
      ],
    };

    // Initialize map
    if (!mapRef.current) {
      mapRef.current = L.map(mapContainerRef.current, {
        zoomControl: false, // Hide default zoom controls
        attributionControl: false // Hide attribution
      });

      // Use a minimal tile layer
      L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
        attribution: 'OpenStreetMap, CartoDB'
      }).addTo(mapRef.current);

      // Add custom zoom control to top right
      L.control.zoom({
        position: 'topright'
      }).addTo(mapRef.current);

      // Add minimal attribution to bottom right
      L.control.attribution({
        position: 'bottomright',
        prefix: ''
      }).addTo(mapRef.current);
    }

    // Create a LatLngBounds object and fit the map to it
    const latLngBounds = L.latLngBounds(bounds);
    mapRef.current.fitBounds(latLngBounds, {
      padding: [50, 50],
      maxZoom: 16
    });

    // Create custom marker icon
    const icon = L.divIcon({
      className: 'custom-marker',
      html: '<div class="marker-inner"></div>',
      iconSize: [12, 12],
      iconAnchor: [6, 6]
    });

    // Clear existing markers
    if (mapRef.current) {
      mapRef.current.eachLayer((layer) => {
        if (layer instanceof L.Marker) {
          mapRef.current!.removeLayer(layer);
        }
      });
    }

    // Add markers for images with GPS data
    images
      .filter(img => img.metadata.gps !== null)
      .forEach(image => {
        if (image.metadata.gps && mapRef.current) {
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
          marker.addTo(mapRef.current);
        }
      });

    // Add custom styles to the document
    const style = document.createElement('style');
    style.textContent = `
      .custom-marker {
        background: transparent;
      }
      .marker-inner {
        width: 12px;
        height: 12px;
        background: #007AFF;
        border-radius: 50%;
        box-shadow: 0 0 0 2px white;
        transition: transform 0.2s ease;
      }
      .marker-inner:hover {
        transform: scale(1.2);
      }
      .custom-popup-container {
        background: white;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border: none;
      }
      .custom-popup {
        padding: 0;
      }
      .popup-image {
        width: 100%;
        height: 200px;
        object-fit: cover;
        border-radius: 8px 8px 0 0;
      }
      .popup-filename {
        padding: 12px;
        font-size: 14px;
        color: #333;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }
      .leaflet-popup-content-wrapper {
        padding: 0;
        border-radius: 8px;
      }
      .leaflet-popup-content {
        margin: 0;
      }
      .leaflet-popup-tip {
        background: white;
      }
      .leaflet-control-zoom {
        border: none !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1) !important;
      }
      .leaflet-control-zoom a {
        background: white !important;
        color: #333 !important;
        border: none !important;
      }
      .leaflet-control-zoom a:hover {
        background: #f5f5f5 !important;
      }
    `;
    document.head.appendChild(style);

    return () => {
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
      document.head.removeChild(style);
    };
  }, [images, center, bounds]);

  return <div ref={mapContainerRef} className="h-full w-full" />;
};

export default MapClient;