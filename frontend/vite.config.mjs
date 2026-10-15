import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // Keep the REACT_APP_ prefix so existing frontend/.env files keep working.
  envPrefix: ["VITE_", "REACT_APP_"],
  server: {
    // The backend's default DROPLET_CORS_ORIGINS is http://localhost:3000.
    port: 3000,
    strictPort: true,
  },
  build: {
    // The Dockerfile and the backend's webui autodetect both expect build/.
    outDir: "build",
  },
});
