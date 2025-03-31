'use client';

import React, { useState, useEffect } from 'react';
import Image from 'next/image';

interface UniqueFace {
  id: number;
  images: string[];
  face_images: string[];
}

interface Config {
  API_BASE_URL: string;
}

export default function PeopleGrid() {
  const [uniqueFaces, setUniqueFaces] = useState<UniqueFace[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedPerson, setSelectedPerson] = useState<UniqueFace | null>(null);
  const [config, setConfig] = useState<Config>({ API_BASE_URL: 'http://localhost:8000' });

  useEffect(() => {
    // Fetch config from backend
    fetch(`${config.API_BASE_URL}/config`)
      .then(res => res.json())
      .then(data => {
        console.log('Backend config:', data);
        setConfig(data);
      })
      .catch(err => {
        console.error('Failed to fetch backend config:', err);
      });
  }, []);

  useEffect(() => {
    const fetchUniqueFaces = async () => {
      try {
        const response = await fetch(`${config.API_BASE_URL}/unique-faces`);
        if (!response.ok) {
          throw new Error('Failed to fetch unique faces');
        }
        const data = await response.json();
        setUniqueFaces(data.unique_faces || []);
      } catch (error) {
        console.error('Error fetching unique faces:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchUniqueFaces();
  }, [config.API_BASE_URL]);

  // Helper function to get face image URL
  const getFaceImageUrl = (filename: string) => {
    // Remove any leading slashes
    const cleanFilename = filename.replace(/^\/+/, '');
    return `${config.API_BASE_URL}/${cleanFilename}`;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12 custom-scrollbar">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
        <span className="ml-3 text-gray-400">Loading faces...</span>
      </div>
    );
  }

  if (!uniqueFaces || uniqueFaces.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-center space-y-4">
        <p className="text-lg text-white/70">No faces detected yet</p>
        <p className="text-sm text-white/50">Try analyzing some photos in the Photos tab first</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#050505] p-4 sm:p-6 custom-scrollbar">
      {/* Grid of unique faces */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-6 max-w-8xl mx-auto">
        {uniqueFaces.map((person) => (
          <div
            key={person.id}
            className="relative group cursor-pointer rounded-xl overflow-hidden bg-[#0a0a0a] transition-all duration-500 hover:scale-[1.02] hover:shadow-2xl hover:shadow-black/40 border border-white/[0.02]"
            onClick={() => setSelectedPerson(person)}
          >
            {/* Show the first face cutout for this person */}
            <div className="relative w-full pb-[100%]">
              <div className="absolute inset-0">
                <Image
                  src={getFaceImageUrl(person.face_images[0])}
                  alt={`Person ${person.id}`}
                  fill
                  className="object-cover"
                  sizes="(max-width: 640px) 50vw, (max-width: 1024px) 33vw, 20vw"
                  unoptimized={true}
                />
              </div>
            </div>
            
            <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/30 to-transparent opacity-0 group-hover:opacity-100 transition-all duration-300 ease-out">
              <div className="absolute bottom-0 left-0 right-0 p-4 transform translate-y-1 group-hover:translate-y-0 transition-transform duration-300">
                <p className="text-sm font-medium text-white/90 truncate">{person.images.length} photos</p>
                <p className="text-xs text-white/70">Person {person.id}</p>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Modal for showing all images of a person */}
      {selectedPerson && (
        <div className="fixed inset-0 bg-black/95 backdrop-blur-sm flex items-center justify-center p-4 custom-scrollbar">
          <div className="mx-auto max-w-7xl w-full glass-panel shadow-2xl overflow-hidden">
            <div className="flex h-[85vh]">
              {/* Left side - Face Preview */}
              <div className="flex-1 relative flex flex-col p-4">
                {/* Top Bar */}
                <div className="flex items-center justify-between mb-4 p-3 glass-panel">
                  <div className="flex items-center gap-3">
                    <h2 className="text-lg font-medium text-white/90">Person {selectedPerson.id}</h2>
                    <span className="px-2 py-1 rounded-lg text-xs bg-white/[0.05] text-white/70">
                      {selectedPerson.images.length} photos
                    </span>
                  </div>
                </div>
                
                {/* Face Preview */}
                <div className="flex-1 relative flex items-center justify-center glass-panel p-4">
                  <div className="relative w-96 h-96">
                    <Image
                      src={getFaceImageUrl(selectedPerson.face_images[0])}
                      alt="Selected Person"
                      fill
                      className="object-cover rounded-lg"
                      unoptimized={true}
                    />
                  </div>
                </div>
              </div>

              {/* Right side - Appearances */}
              <div className="w-80 border-l border-white/[0.05] custom-scrollbar">
                <div className="p-6 space-y-6">
                  <h3 className="text-lg font-medium text-white/90 mb-4">Photos</h3>
                  <div className="grid gap-4">
                    {selectedPerson.images.map((image, index) => (
                      <div 
                        key={index}
                        className="relative aspect-square rounded-lg overflow-hidden glass-panel"
                      >
                        <Image
                          src={getFaceImageUrl(image)}
                          alt={`Face ${index + 1}`}
                          fill
                          className="object-cover"
                          sizes="320px"
                          unoptimized={true}
                        />
                        <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/30 to-transparent">
                          <div className="absolute bottom-0 left-0 right-0 p-3">
                            <p className="text-sm text-white/90 truncate">{image}</p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}