import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import fs from "node:fs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const FLASK_TARGET = process.env.FLASK_URL ?? "http://localhost:5000";

// Serve /static/* directly from app/static/ so image assets load in dev
// even when Flask isn't running. Runs BEFORE the proxy rule below.
function staticFallbackPlugin(): Plugin {
  const root = resolve(__dirname, "../static");
  const mime: Record<string, string> = {
    jpg: "image/jpeg",
    jpeg: "image/jpeg",
    png: "image/png",
    webp: "image/webp",
    gif: "image/gif",
    svg: "image/svg+xml",
    mp4: "video/mp4",
    webm: "video/webm",
  };
  return {
    name: "forma-static-fallback",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!req.url?.startsWith("/static/")) return next();
        const urlPath = decodeURIComponent(req.url.replace(/^\/static/, "").split("?")[0]);
        const abs = resolve(root, "." + urlPath);
        if (!abs.startsWith(root)) return next();
        fs.stat(abs, (err, stat) => {
          if (err || !stat.isFile()) return next();
          const ext = abs.split(".").pop()?.toLowerCase() ?? "";
          res.setHeader("Content-Type", mime[ext] ?? "application/octet-stream");
          res.setHeader("Cache-Control", "public, max-age=3600");
          fs.createReadStream(abs).pipe(res);
        });
      });
    },
  };
}

export default defineConfig({
  plugins: [staticFallbackPlugin(), react(), tailwindcss()],
  resolve: {
    alias: {
      "@": resolve(__dirname, "src"),
    },
  },
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": { target: FLASK_TARGET, changeOrigin: true },
      "/socket.io": {
        target: FLASK_TARGET,
        changeOrigin: true,
        ws: true,
      },
      "/legacy": { target: FLASK_TARGET, changeOrigin: true },
    },
  },
  build: {
    outDir: resolve(__dirname, "../static/dist"),
    emptyOutDir: true,
    assetsDir: "assets",
    sourcemap: false,
    target: "es2022",
  },
});
