'use client';

import { useEffect, useState } from 'react';
import dynamic from 'next/dynamic';

// Create a NoSSR wrapper component
const NoSSR = ({ children }: { children: React.ReactNode }) => {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    return () => setMounted(false);
  }, []);

  return mounted ? <>{children}</> : null;
};

interface MapComponentProps {
  center: [number, number];
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

const MapComponent = ({ center, images }: MapComponentProps) => {
  const [mapComponents, setMapComponents] = useState<any>(null);
  const [icon, setIcon] = useState<any>(null);

  useEffect(() => {
    // Import all required components on the client side
    Promise.all([
      import('react-leaflet'),
      import('leaflet')
    ]).then(([reactLeaflet, L]) => {
      const { MapContainer, TileLayer, Marker, Popup } = reactLeaflet;
      
      const icon = new L.Icon({
        iconUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png',
        iconRetinaUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon-2x.png',
        shadowUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-shadow.png',
        iconSize: [25, 41],
        iconAnchor: [12, 41],
        popupAnchor: [1, -34],
        shadowSize: [41, 41]
      });

      setMapComponents({ MapContainer, TileLayer, Marker, Popup });
      setIcon(icon);
    });
  }, []);

  if (!mapComponents || !icon) {
    return (
      <div className="h-96 flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
      </div>
    );
  }

  const { MapContainer, TileLayer, Marker, Popup } = mapComponents;

  return (
    <NoSSR>
      <div className="h-full w-full">
        <MapContainer
          key={`${center[0]}-${center[1]}`}
          center={center}
          zoom={13}
          style={{ height: '100%', width: '100%' }}
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          {images.map((image, index) => (
            image.metadata.gps && (
              <Marker
                key={`${image.filename}-${index}`}
                position={[image.metadata.gps.latitude, image.metadata.gps.longitude]}
                icon={icon}
              >
                <Popup>
                  <div className="p-2">
                    <img
                      src={`http://localhost:8000/images/${image.filename}`}
                      alt={image.filename}
                      className="w-48 h-48 object-cover rounded"
                    />
                    <p className="mt-2 text-sm">{image.filename}</p>
                  </div>
                </Popup>
              </Marker>
            )
          ))}
        </MapContainer>
      </div>
    </NoSSR>
  );
};

// Prevent any server-side rendering of the component
export default dynamic(() => Promise.resolve(MapComponent), {
  ssr: false
});