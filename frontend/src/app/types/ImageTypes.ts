import { useEffect, useRef } from 'react';
import { useImageStore } from '../store/imageStore';

export const useImageLoader = () => {
  const observerRef = useRef<IntersectionObserver | null>(null);
  const imageRefs = useRef<{ [key: string]: HTMLDivElement | null }>({});
  const addLoadedImage = useImageStore(state => state.addLoadedImage);

  useEffect(() => {
    observerRef.current = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const img = entry.target as HTMLDivElement;
            const filename = img.dataset.filename;
            if (filename) {
              addLoadedImage(filename);
            }
          }
        });
      },
      { rootMargin: '50px' }
    );

    return () => {
      if (observerRef.current) {
        observerRef.current.disconnect();
      }
    };
  }, [addLoadedImage]);

  return { imageRefs, observerRef };
};