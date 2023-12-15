import RootStyleRegistry from './emotion';
import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'NerfBaselines',
  description: 'Reproducible evaluation of NeRF methods',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <head>
      </head>
      <body className={`main-container ${inter.className}`}>
        <RootStyleRegistry>{children}</RootStyleRegistry>
      </body>
    </html>
  )
}