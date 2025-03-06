import './globals.css'
import 'leaflet/dist/leaflet.css'
import type { Metadata } from 'next'
import { Inter } from 'next/font/google'

const inter = Inter({
  subsets: ['latin'],
  display: 'swap',
  preload: true,
  adjustFontFallback: true,
  variable: '--font-inter',
})

export const metadata: Metadata = {
  title: 'Photo Cortex',
  description: 'AI-powered photo management',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <head>
        <link 
          rel="stylesheet" 
          href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
          integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
          crossOrigin=""
        />
        <script 
          src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
          integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
          crossOrigin=""
        ></script>
        <title>PhotoCortex</title>
        <meta name="theme-color" content="#050505" />
      </head>
      <body className={`${inter.variable} font-sans min-h-screen bg-[#050505] text-white/90 antialiased`}>
        <div className="fixed inset-0 bg-gradient-to-br from-black/10 via-black/5 to-black/20 pointer-events-none" />
        <main className="relative">{children}</main>
      </body>
    </html>
  )
}