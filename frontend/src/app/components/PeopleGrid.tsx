'use client';

import React, { useState, useEffect } from 'react';
import Image from 'next/image';
import { PencilIcon, CheckIcon, XMarkIcon } from '@heroicons/react/24/outline';

interface UniqueFace {
  id: number;
  label: string;
  images: string[];  // Array of original image paths
  face_images: number[];  // Array of face IDs (changed from string[])
}

interface Config {
  API_BASE_URL: string;
}

export default function PeopleGrid() {
  const [uniqueFaces, setUniqueFaces] = useState<UniqueFace[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedPerson, setSelectedPerson] = useState<UniqueFace | null>(null);
  const [config, setConfig] = useState<Config>({ API_BASE_URL: 'http://localhost:8000' });
  const [editingPerson, setEditingPerson] = useState<number | null>(null);
  const [newName, setNewName] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

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
        setError('Failed to load people data. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchUniqueFaces();
  }, [config.API_BASE_URL]);

  const getFaceImageUrl = (faceIdentifier: string | number) => {
    const id = faceIdentifier.toString();
    // If it's already a full path (legacy case), use as-is
    if (id.includes('/')) {
      return `${config.API_BASE_URL}/${id.replace(/^\//, '')}`;
    }
    // Otherwise assume it's a face ID and use new format
    return `${config.API_BASE_URL}/images/faces/${id}`;
  };

  // Function to update person name
  const updatePersonName = async (personId: number, name: string) => {
    try {
      setError(null);
      const response = await fetch(`${config.API_BASE_URL}/face-identity/${personId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ label: name }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to update name');
      }
      
      const data = await response.json();
      
      // Update local state
      setUniqueFaces(prevFaces => 
        prevFaces.map(face => 
          face.id === personId ? { ...face, label: name } : face
        )
      );
      
      // If the selected person is being edited, update that too
      if (selectedPerson && selectedPerson.id === personId) {
        setSelectedPerson({ ...selectedPerson, label: name });
      }
      
      setEditingPerson(null);
      setNewName("");
    } catch (error) {
      console.error('Error updating name:', error);
      setError(error instanceof Error ? error.message : 'Failed to update name');
    }
  };

  // Function to handle form submission
  const handleNameSubmit = (e: React.FormEvent, personId: number) => {
    e.preventDefault();
    if (newName.trim()) {
      updatePersonName(personId, newName.trim());
    }
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

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[300px] py-12 text-center space-y-4">
        <p className="text-lg text-red-400">{error}</p>
        <button 
          onClick={() => window.location.reload()} 
          className="px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg text-white/80"
        >
          Try Again
        </button>
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
            onClick={() => {
              if (editingPerson !== person.id) {
                setSelectedPerson(person);
              }
            }}
          >
            {/* Show the first face cutout for this person */}
            <div className="aspect-square w-full">
              <div className="relative w-full h-full">
                <Image
                  src={getFaceImageUrl(person.face_images[0].toString())}
                  alt={person.label || `Person ${person.id}`}
                  fill
                  className="object-cover"
                  sizes="(max-width: 640px) 50vw, (max-width: 768px) 33vw, (max-width: 1024px) 25vw, 20vw"
                  priority={person.id < 10}
                />
              </div>
            </div>
            
            {/* Hover overlay */}
            <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/40 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300">
              <div className="absolute bottom-0 left-0 right-0 p-4">
                {editingPerson === person.id ? (
                  <form 
                    onSubmit={(e) => handleNameSubmit(e, person.id)}
                    onClick={(e) => e.stopPropagation()}
                    className="flex items-center space-x-2"
                  >
                    <input
                      type="text"
                      value={newName}
                      onChange={(e) => setNewName(e.target.value)}
                      className="bg-white/20 rounded px-2 py-1 text-white text-sm w-full outline-none"
                      placeholder="Enter name"
                      autoFocus
                    />
                    <button 
                      type="submit" 
                      className="p-1 bg-white/20 rounded-full hover:bg-white/30"
                    >
                      <CheckIcon className="w-4 h-4 text-white" />
                    </button>
                    <button 
                      type="button" 
                      onClick={(e) => {
                        e.stopPropagation();
                        setEditingPerson(null);
                        setNewName("");
                      }}
                      className="p-1 bg-white/20 rounded-full hover:bg-white/30"
                    >
                      <XMarkIcon className="w-4 h-4 text-white" />
                    </button>
                  </form>
                ) : (
                  <div className="flex items-center justify-between">
                    <h3 className="text-white font-medium truncate">
                      {person.label || `Person ${person.id}`}
                    </h3>
                    <button 
                      onClick={(e) => {
                        e.stopPropagation();
                        setEditingPerson(person.id);
                        setNewName(person.label || "");
                      }}
                      className="p-1 text-white/60 hover:text-white/90 hover:bg-white/10 rounded-full transition-colors"
                    >
                      <PencilIcon className="w-4 h-4" />
                    </button>
                  </div>
                )}
                <p className="text-white/60 text-sm mt-1">{person.images.length} photos</p>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Modal for selected person */}
      {selectedPerson && (
        <div className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-4 overflow-y-auto">
          <div 
            className="bg-[#0a0a0a] rounded-2xl overflow-hidden max-w-6xl w-full max-h-[90vh] border border-white/[0.05] shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex flex-col h-full">
              {/* Header */}
              <div className="flex items-center justify-between p-6 border-b border-white/[0.05]">
                <div className="flex items-center gap-4">
                  <div className="relative w-12 h-12 rounded-lg overflow-hidden">
                    <Image
                      src={getFaceImageUrl(selectedPerson.face_images[0].toString())}
                      alt={selectedPerson.label || `Person ${selectedPerson.id}`}
                      fill
                      className="object-cover"
                    />
                  </div>
                  <div>
                    {editingPerson === selectedPerson.id ? (
                      <form 
                        onSubmit={(e) => handleNameSubmit(e, selectedPerson.id)}
                        className="flex items-center space-x-2"
                      >
                        <input
                          type="text"
                          value={newName}
                          onChange={(e) => setNewName(e.target.value)}
                          className="bg-white/10 rounded px-2 py-1 text-white text-sm outline-none"
                          placeholder="Enter name"
                          autoFocus
                        />
                        <button 
                          type="submit" 
                          className="p-1 bg-white/10 rounded-full hover:bg-white/20"
                        >
                          <CheckIcon className="w-4 h-4 text-white" />
                        </button>
                        <button 
                          type="button" 
                          onClick={() => {
                            setEditingPerson(null);
                            setNewName("");
                          }}
                          className="p-1 bg-white/10 rounded-full hover:bg-white/20"
                        >
                          <XMarkIcon className="w-4 h-4 text-white" />
                        </button>
                      </form>
                    ) : (
                      <div className="flex items-center">
                        <h2 className="text-lg font-medium text-white/90 mr-2">
                          {selectedPerson.label || `Person ${selectedPerson.id}`}
                        </h2>
                        <button 
                          onClick={() => {
                            setEditingPerson(selectedPerson.id);
                            setNewName(selectedPerson.label || "");
                          }}
                          className="p-1 text-white/60 hover:text-white/90 hover:bg-white/10 rounded-full transition-colors"
                        >
                          <PencilIcon className="w-4 h-4" />
                        </button>
                      </div>
                    )}
                    <p className="text-sm text-white/60">Found in {selectedPerson.images.length} photos</p>
                  </div>
                </div>
                <button 
                  onClick={() => setSelectedPerson(null)}
                  className="p-2 hover:bg-white/[0.05] rounded-lg transition-colors"
                >
                  <XMarkIcon className="w-5 h-5 text-white/60" />
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
                          src={getFaceImageUrl(faceImage.toString())}
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