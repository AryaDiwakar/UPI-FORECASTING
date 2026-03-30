/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'fintech': {
          '900': '#0a0f1a',
          '800': '#111827',
          '700': '#1f2937',
          '600': '#374151',
          '500': '#4b5563',
          '400': '#6b7280',
          '300': '#9ca3af',
          '200': '#d1d5db',
          '100': '#f3f4f6',
          'accent': '#10b981',
          'accent-hover': '#059669',
        }
      }
    },
  },
  plugins: [],
}
