import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#0A0E17",
        foreground: "#F8FAFC",
        surface: "#101625",
        "surface-raised": "#1A2338",
        border: "#202D45",
        primary: "#00F2FE", // Electric Cyan
        secondary: "#8B9DC3",
        bullish: "#00E676",
        bearish: "#FF1744",
        warning: "#FFD700",
      },
      fontFamily: {
        sans: ["var(--font-outfit)"],
        mono: ["var(--font-jetbrains-mono)"],
        numbers: ["var(--font-space)"],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      }
    },
  },
  plugins: [],
};
export default config;
