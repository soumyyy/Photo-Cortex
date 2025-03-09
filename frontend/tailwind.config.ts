import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './src/**/*.{js,ts,jsx,tsx,mdx}',  // Include all source files
  ],
  theme: {
    extend: {
      colors: {
        background: '#050505',
      },
      fontFamily: {
        sans: ['var(--font-inter)', 'sans-serif'],
      },
      maxWidth: {
        '8xl': '88rem',
      },
    },
  },
  plugins: [],
}

export default config