import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/tryon": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000",
      "/system": "http://127.0.0.1:8000",
      "/metrics": "http://127.0.0.1:8000",
      "/artifacts": "http://127.0.0.1:8000"
    }
  }
});
