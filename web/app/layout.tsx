import type { Metadata, Viewport } from "next";
import "./globals.css";
import Sidebar from "@/components/Sidebar";

export const metadata: Metadata = {
  title: "ATLAS Dashboard",
  description: "AI Trading Lab for Agent Strategy -- 4 LLMs x 2 Architectures Benchmark Dashboard",
  icons: { icon: "/favicon.svg" },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#0f1117",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ko" className="dark">
      <body className="flex min-h-screen">
        <Sidebar />
        <main className="flex-1 lg:ml-64 p-4 pt-16 lg:pt-6 lg:p-6 overflow-auto">
          {children}
        </main>
      </body>
    </html>
  );
}
