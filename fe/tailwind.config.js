/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: 'hsl(0 0% 100%)',
        foreground: 'hsl(0 0% 3.9%)',
        card: 'hsl(0 0% 100%)',
        'card-foreground': 'hsl(0 0% 3.9%)',
        border: 'hsl(0 0% 89.8%)',
        input: 'hsl(0 0% 89.8%)',
        primary: 'hsl(0 0% 9%)',
        'primary-foreground': 'hsl(0 0% 98%)',
        secondary: 'hsl(0 0% 96.1%)',
        'secondary-foreground': 'hsl(0 0% 9%)',
        muted: 'hsl(0 0% 96.1%)',
        'muted-foreground': 'hsl(0 0% 45.1%)',
        accent: 'hsl(0 0% 96.1%)',
        'accent-foreground': 'hsl(0 0% 9%)',
      },
    },
  },
  plugins: [],
}

