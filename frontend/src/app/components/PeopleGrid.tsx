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
        console.log('Fetching unique faces from:', `${config.API_BASE_URL}/unique-faces`);
        const response = await fetch(`${config.API_BASE_URL}/unique-faces`);
        if (!response.ok) {
          throw new Error(`Failed to fetch unique faces: ${response.status} ${response.statusText}`);
        }
        const data = await response.json();
        console.log('Received unique faces data:', data);
        setUniqueFaces(data.unique_faces || []);
        console.log('Updated unique faces state:', data.unique_faces?.length || 0, 'faces');
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
    const cleanFilename = filename.replace(/^\/+/, '');
    const url = `${config.API_BASE_URL}/${cleanFilename}`;
    console.log('Face image URL:', url);
    return url;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[300px]">
        <div className="relative">
          <div className="w-12 h-12 rounded-full border-2 border-white/10 animate-ping absolute inset-0" />
          <div className="w-12 h-12 rounded-full border-2 border-t-white/40 animate-spin" />
        </div>
        <span className="ml-4 text-white/60">Loading faces...</span>
      </div>
    );
  }

  if (!uniqueFaces || uniqueFaces.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[300px] py-12 text-center space-y-4">
        <p className="text-lg text-white/70">No faces detected yet</p>
        <p className="text-sm text-white/50">Try analyzing some photos in the Photos tab first</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#050505] p-4 sm:p-6">
      {/* Grid of unique faces */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-6 max-w-8xl mx-auto">
        {uniqueFaces.map((person) => (
          <div
            key={person.id}
            className="relative group cursor-pointer rounded-xl overflow-hidden bg-[#0a0a0a] transition-all duration-500 hover:scale-[1.02] hover:shadow-2xl hover:shadow-black/40 border border-white/[0.02]"
            onClick={() => setSelectedPerson(person)}
          >
            {/* Show the first face cutout for this person */}
            <div className="aspect-square w-full">
              <div className="relative w-full h-full">
                <Image
                  src={getFaceImageUrl(person.face_images[0])}
                  alt={`Person ${person.id}`}
                  fill
                  className="object-cover"
                  sizes="(max-width: 640px) 50vw, (max-width: 768px) 33vw, (max-width: 1024px) 25vw, 20vw"
                  priority={person.id < 10}
                />
              </div>
            </div>
            
            {/* Hover overlay */}
            <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/30 to-transparent opacity-0 group-hover:opacity-100 transition-all duration-300 ease-out">
              <div className="absolute bottom-0 left-0 right-0 p-4">
                <p className="text-sm text-white/90">
                  {person.images.length} photo{person.images.length !== 1 ? 's' : ''}
                </p>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Modal for showing all images of a person */}
      {selectedPerson && (
        <div 
          className="fixed inset-0 bg-black/90 backdrop-blur-sm z-50 flex items-center justify-center p-4"
          onClick={() => setSelectedPerson(null)}
        >
          <div 
            className="bg-[#0a0a0a] rounded-2xl shadow-2xl max-w-6xl w-full max-h-[90vh] overflow-hidden border border-white/[0.05]"
            onClick={e => e.stopPropagation()}
          >
            <div className="flex flex-col h-full">
              {/* Header */}
              <div className="flex items-center justify-between p-6 border-b border-white/[0.05]">
                <div className="flex items-center gap-4">
                  <div className="relative w-12 h-12 rounded-lg overflow-hidden">
                    <Image
                      src={getFaceImageUrl(selectedPerson.face_images[0])}
                      alt={`Person ${selectedPerson.id}`}
                      fill
                      className="object-cover"
                    />
                  </div>
                  <div>
                    <h2 className="text-lg font-medium text-white/90">Person {selectedPerson.id}</h2>
                    <p className="text-sm text-white/60">Found in {selectedPerson.images.length} photos</p>
                  </div>
                </div>
                <button 
                  onClick={() => setSelectedPerson(null)}
                  className="p-2 hover:bg-white/[0.05] rounded-lg transition-colors"
                >
                  <svg className="w-5 h-5 text-white/60" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>

              {/* Content */}
              <div className="flex-1 overflow-y-auto">
                {/* Face cutouts section */}
                <div className="p-6 border-b border-white/[0.05]">
                  <h3 className="text-sm font-medium text-white/60 uppercase tracking-wider mb-4">Face Cutouts</h3>
                  <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-4">
                    {selectedPerson.face_images.map((faceImage, index) => (
                      <div 
                        key={index}
                        className="aspect-square relative rounded-lg overflow-hidden bg-[#0a0a0a] border border-white/[0.05]"
                      >
                        <Image
                          src={getFaceImageUrl(faceImage)}
                          alt={`Face ${index + 1}`}
                          fill
                          className="object-cover"
                          sizes="(max-width: 640px) 50vw, (max-width: 768px) 25vw, (max-width: 1024px) 16.66vw, 12.5vw"
                        />
                      </div>
                    ))}
                  </div>
                </div>

                {/* Original photos section */}
                <div className="p-6">
                  <h3 className="text-sm font-medium text-white/60 uppercase tracking-wider mb-4">Original Photos</h3>
                  <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
                    {selectedPerson.images.map((image, index) => (
                      <div 
                        key={index}
                        className="aspect-video relative rounded-lg overflow-hidden bg-[#0a0a0a] border border-white/[0.05]"
                      >
                        <Image
                          src={`${config.API_BASE_URL}/image/${image}`}
                          alt={`Photo ${index + 1}`}
                          fill
                          className="object-cover"
                          sizes="(max-width: 640px) 100vw, (max-width: 768px) 50vw, 33.33vw"
                        />
                        <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-0 hover:opacity-100 transition-opacity">
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