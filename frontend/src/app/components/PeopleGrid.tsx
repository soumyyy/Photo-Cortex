'use client';

import React, { useState, useEffect } from 'react';
import Image from 'next/image';

interface UniqueFace {
  id: number;
  images: string[];
  face_images: string[];
}

export default function PeopleGrid() {
  const [uniqueFaces, setUniqueFaces] = useState<UniqueFace[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedPerson, setSelectedPerson] = useState<UniqueFace | null>(null);

  useEffect(() => {
    fetchUniqueFaces();
  }, []);

  const fetchUniqueFaces = async () => {
    try {
      const response = await fetch('http://localhost:8000/unique-faces');
      const data = await response.json();
      setUniqueFaces(data.unique_faces || []);
    } catch (error) {
      console.error('Failed to fetch unique faces:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
        <span className="ml-3 text-gray-400">Loading faces...</span>
      </div>
    );
  }

  return (
    <div>
      {/* Grid of unique faces */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
        {uniqueFaces.map((person) => (
          <div
            key={person.id}
            className="relative bg-gray-800 rounded-lg overflow-hidden cursor-pointer group"
            onClick={() => setSelectedPerson(person)}
          >
            {/* Show the first face cutout for this person */}
            <div className="aspect-square relative">
              <Image
                src={`http://localhost:8000/images/${encodeURIComponent(person.face_images[0])}`}
                alt={`Person ${person.id}`}
                fill
                className="object-cover"
                onError={(e) => {
                  console.error(`Failed to load face image: ${person.face_images[0]}`);
                  // If face image fails, try showing the first regular image instead
                  if (person.images.length > 0) {
                    (e.target as HTMLImageElement).src = `http://localhost:8000/images/${encodeURIComponent(person.images[0])}`;
                  }
                }}
              />
            </div>
            
            <div className="absolute bottom-0 left-0 right-0 bg-black/50 p-2 transform translate-y-full group-hover:translate-y-0 transition-transform">
              <p className="text-sm text-white">{person.images.length} photos</p>
            </div>
          </div>
        ))}
      </div>

      {/* Modal for showing all images of a person */}
      {selectedPerson && (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4">
          <div className="bg-gray-900 rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto p-6">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-xl text-white">All Photos ({selectedPerson.images.length})</h3>
              <button
                onClick={() => setSelectedPerson(null)}
                className="text-gray-400 hover:text-white"
              >
                Close
              </button>
            </div>
            
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
              {selectedPerson.images.map((image, index) => (
                <div key={index} className="aspect-square relative bg-gray-800 rounded-lg overflow-hidden">
                  <Image
                    src={`http://localhost:8000/images/${encodeURIComponent(image)}`}
                    alt={`Photo ${index + 1}`}
                    fill
                    className="object-cover"
                    onError={(e) => {
                      console.error(`Failed to load person image: ${image}`);
                    }}
                  />
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}