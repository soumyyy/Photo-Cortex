import { useEffect, useRef } from 'react';
import { useImageStore } from '../store/imageStore';

export const useImageLoader = () => {
  const observerRef = useRef<IntersectionObserver | null>(null);
  const imageRefs = useRef<{ [key: string]: HTMLDivElement | null }>({});
  const { addLoadedImage, config } = useImageStore();

  useEffect(() => {
    if (!config.API_BASE_URL) return;

    observerRef.current = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const img = entry.target as HTMLDivElement;
            const filename = img.dataset.filename;
            if (filename) {
              // Preload the image
              const image = new Image();
              image.src = `${config.API_BASE_URL}/images/${filename}`;
              image.onload = () => {
                addLoadedImage(filename);
              };
              image.onerror = () => {
                console.error(`Failed to load image: ${filename}`);
                // Optionally retry loading after a delay
                setTimeout(() => {
                  if (!image.complete) {
                    image.src = image.src;
                  }
                }, 2000);
              };
            }
          }
        });
      },
      { 
        rootMargin: '100px', // Increased for earlier loading
        threshold: [0, 0.1, 0.5], // Multiple thresholds for smoother loading
      }
    );

    return () => {
      if (observerRef.current) {
        observerRef.current.disconnect();
      }
    };
  }, [addLoadedImage, config.API_BASE_URL]);

  return { imageRefs, observerRef };
};