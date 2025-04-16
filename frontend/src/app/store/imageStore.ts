import { create } from 'zustand';
import { ImageAnalysis } from '../types/ImageAnalysis';

interface ImageState {
  images: ImageAnalysis[];
  selectedImage: ImageAnalysis | null;
  loading: boolean;
  error: string | null;
  progress: number;
  processedCount: number;
  totalImages: number;
  loadedImages: Set<string>;
  config: {
    API_BASE_URL: string;
  };
  
  // Actions
  setImages: (images: ImageAnalysis[]) => void;
  setSelectedImage: (image: ImageAnalysis | null) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setProgress: (progress: number) => void;
  setProcessedCount: (count: number) => void;
  setTotalImages: (total: number) => void;
  addLoadedImage: (filename: string) => void;
  setConfig: (config: { API_BASE_URL: string }) => void;
}

export const useImageStore = create<ImageState>((set) => ({
  images: [],
  selectedImage: null,
  loading: false,
  error: null,
  progress: 0,
  processedCount: 0,
  totalImages: 0,
  loadedImages: new Set(),
  config: {
    API_BASE_URL: 'http://localhost:8000'
  },
  
  setImages: (images) => set({ images }),
  setSelectedImage: (image) => set({ selectedImage: image }),
  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error }),
  setProgress: (progress) => set({ progress }),
  setProcessedCount: (count) => set({ processedCount: count }),
  setTotalImages: (total) => set({ totalImages: total }),
  addLoadedImage: (filename) => set((state) => ({
    loadedImages: new Set([...Array.from(state.loadedImages), filename])
  })),
  setConfig: (config) => set({ config })
}));