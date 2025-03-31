'use client';

import React, { ReactNode, useRef, useState, useEffect, useCallback } from 'react';

interface GlassScrollerProps {
  children: ReactNode;
  className?: string;
  maxHeight?: string | number;
  width?: string | number;
  showScrollbar?: boolean;
  fadeTop?: boolean;
  fadeBottom?: boolean;
  scrollbarPosition?: 'right' | 'left';
  scrollbarWidth?: number;
  scrollbarColor?: string;
  scrollbarHoverColor?: string;
  scrollbarTrackColor?: string;
  scrollbarBorderRadius?: number;
  onScroll?: (event: React.UIEvent<HTMLDivElement>) => void;
}

const GlassScroller: React.FC<GlassScrollerProps> = ({
  children,
  className = '',
  maxHeight = '100%',
  width = '100%',
  showScrollbar = true,
  fadeTop = true,
  fadeBottom = true,
  scrollbarPosition = 'right',
  scrollbarWidth = 4,
  scrollbarColor = 'rgba(255, 255, 255, 0.3)',
  scrollbarHoverColor = 'rgba(255, 255, 255, 0.5)',
  scrollbarTrackColor = 'rgba(255, 255, 255, 0.05)',
  scrollbarBorderRadius = 10,
  onScroll
}) => {
  const scrollerRef = useRef<HTMLDivElement>(null);
  const [showTopFade, setShowTopFade] = useState(false);
  const [showBottomFade, setShowBottomFade] = useState(false);
  const [isScrolling, setIsScrolling] = useState(false);
  const [scrollTimeout, setScrollTimeout] = useState<NodeJS.Timeout | null>(null);

  // Calculate if fades should be shown based on scroll position
  const updateFades = useCallback(() => {
    if (!scrollerRef.current) return;
    
    const { scrollTop, scrollHeight, clientHeight } = scrollerRef.current;
    
    // Show top fade if scrolled down
    setShowTopFade(fadeTop && scrollTop > 0);
    
    // Show bottom fade if there's more content to scroll to
    setShowBottomFade(fadeBottom && scrollTop + clientHeight < scrollHeight - 5);
  }, [fadeTop, fadeBottom]);

  // Handle scroll event
  const handleScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
    updateFades();
    
    // Show scrollbar during scrolling
    setIsScrolling(true);
    
    // Clear previous timeout
    if (scrollTimeout) {
      clearTimeout(scrollTimeout);
    }
    
    // Hide scrollbar after scrolling stops
    const timeout = setTimeout(() => {
      setIsScrolling(false);
    }, 1000);
    
    setScrollTimeout(timeout as unknown as NodeJS.Timeout);
    
    // Call user-provided onScroll handler
    if (onScroll) {
      onScroll(e);
    }
  }, [updateFades, onScroll, scrollTimeout]);

  // Initial check for fades
  useEffect(() => {
    updateFades();
    
    // Add resize listener to update fades when container size changes
    const handleResize = () => {
      updateFades();
    };
    
    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
      if (scrollTimeout) {
        clearTimeout(scrollTimeout);
      }
    };
  }, [updateFades, scrollTimeout]);

  // Generate dynamic styles for scrollbar
  const scrollbarStyles = `
    .glass-scroller::-webkit-scrollbar {
      width: ${scrollbarWidth}px;
      height: ${scrollbarWidth}px;
    }
    
    .glass-scroller::-webkit-scrollbar-track {
      background: ${scrollbarTrackColor};
      border-radius: ${scrollbarBorderRadius}px;
    }
    
    .glass-scroller::-webkit-scrollbar-thumb {
      background: ${scrollbarColor};
      border-radius: ${scrollbarBorderRadius}px;
      transition: background 0.3s ease;
    }
    
    .glass-scroller::-webkit-scrollbar-thumb:hover {
      background: ${scrollbarHoverColor};
    }
    
    .glass-scroller::-webkit-scrollbar-corner {
      background: transparent;
    }
    
    .glass-scroller.scrolling::-webkit-scrollbar-thumb {
      background: ${scrollbarHoverColor};
    }
    
    .glass-scroller {
      scrollbar-width: thin;
      scrollbar-color: ${scrollbarColor} ${scrollbarTrackColor};
    }
  `;

  return (
    <div 
      className={`glass-scroller-container relative ${className}`}
      style={{ 
        width, 
        maxHeight: typeof maxHeight === 'number' ? `${maxHeight}px` : maxHeight,
        height: '100%'
      }}
    >
      {/* Inject custom scrollbar styles */}
      <style jsx>{scrollbarStyles}</style>
      
      {/* Top fade gradient */}
      {showTopFade && (
        <div 
          className="absolute top-0 left-0 right-0 h-12 z-10 pointer-events-none"
          style={{
            background: 'linear-gradient(to bottom, rgba(0, 0, 0, 0.3), transparent)',
            opacity: showTopFade ? 1 : 0,
            transition: 'opacity 0.3s ease'
          }}
        />
      )}
      
      {/* Bottom fade gradient */}
      {showBottomFade && (
        <div 
          className="absolute bottom-0 left-0 right-0 h-12 z-10 pointer-events-none"
          style={{
            background: 'linear-gradient(to top, rgba(0, 0, 0, 0.3), transparent)',
            opacity: showBottomFade ? 1 : 0,
            transition: 'opacity 0.3s ease'
          }}
        />
      )}
      
      {/* Scrollable content */}
      <div
        ref={scrollerRef}
        className={`glass-scroller overflow-y-auto h-full ${isScrolling ? 'scrolling' : ''} ${!showScrollbar ? 'hide-scrollbar' : ''}`}
        style={{
          padding: `${scrollbarWidth}px`,
          paddingRight: scrollbarPosition === 'right' ? `${scrollbarWidth * 2}px` : `${scrollbarWidth}px`,
          paddingLeft: scrollbarPosition === 'left' ? `${scrollbarWidth * 2}px` : `${scrollbarWidth}px`,
          background: 'rgba(0, 0, 0, 0.1)',
          backdropFilter: 'blur(8px)',
          borderRadius: '8px',
          border: '1px solid rgba(255, 255, 255, 0.05)',
          boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1), inset 0 1px 0 rgba(255, 255, 255, 0.05)'
        }}
        onScroll={handleScroll}
      >
        {children}
      </div>
    </div>
  );
};

export default GlassScroller;
