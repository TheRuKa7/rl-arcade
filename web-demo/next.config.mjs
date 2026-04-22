import CopyPlugin from "copy-webpack-plugin";
import path from "node:path";

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // onnxruntime-web ships .wasm files that need to be served from /public.
  webpack: (config, { isServer }) => {
    if (!isServer) {
      config.plugins.push(
        new CopyPlugin({
          patterns: [
            {
              from: path.resolve("node_modules/onnxruntime-web/dist"),
              to: path.resolve(".next/static/chunks/pages"),
              filter: (r) => r.endsWith(".wasm") || r.endsWith(".mjs"),
            },
            {
              from: path.resolve("node_modules/onnxruntime-web/dist"),
              to: path.resolve("public/ort"),
              filter: (r) => r.endsWith(".wasm") || r.endsWith(".mjs"),
              noErrorOnMissing: true,
            },
          ],
        }),
      );
    }
    return config;
  },
};

export default nextConfig;
