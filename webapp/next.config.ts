import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async redirects() {
    return [
      // The original FIREKit Hub dashboard lives in public/dashboard with
      // relative asset paths, so it must be served as /dashboard/index.html.
      {
        source: "/dashboard",
        destination: "/dashboard/index.html",
        permanent: false,
      },
    ];
  },
};

export default nextConfig;
