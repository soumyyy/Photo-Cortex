import './globals.css'
import 'leaflet/dist/leaflet.css'
import type { Metadata } from 'next'
import { Inter } from 'next/font/google'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'PhotoCortex',
  description: 'Analyze images with AI',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <head>
        <title>PhotoCortex</title>
        <meta name="theme-color" content="#050505" />
      </head>
      <body className={`${inter.className} min-h-screen bg-[#050505] text-white/90 antialiased`}>
        <div className="fixed inset-0 bg-gradient-to-br from-black/10 via-black/5 to-black/20 pointer-events-none" />
        <main className="relative">{children}</main>
      </body>
    </html>
  )
}