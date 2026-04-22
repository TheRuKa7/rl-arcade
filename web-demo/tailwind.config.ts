import type { Config } from "tailwindcss";

export default {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        background: "hsl(240 10% 3.9%)",
        surface: "hsl(240 10% 8%)",
        border: "hsl(240 5% 20%)",
        foreground: "hsl(0 0% 98%)",
        muted: "hsl(240 4% 60%)",
        primary: "hsl(200 85% 60%)",
        success: "hsl(142 71% 45%)",
        danger: "hsl(0 84% 60%)",
      },
    },
  },
} satisfies Config;
