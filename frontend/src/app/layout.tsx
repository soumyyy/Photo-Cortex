import './globals.css'
import 'leaflet/dist/leaflet.css'
import type { Metadata } from 'next'
import { Inter } from 'next/font/google'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Image Analysis',
  description: 'Analyze images with AI',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <head>
        <title>Image GPS Map</title>
      </head>
      <body className={inter.className}>{children}</body>
    </html>
  )
}