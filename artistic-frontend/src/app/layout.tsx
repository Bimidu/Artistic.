import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ASD Detection System",
  description: "Advanced machine learning system for analyzing speech patterns to support autism spectrum disorder detection using multi-modal feature extraction for children",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="bg-white min-h-screen font-sans antialiased">
        {children}
      </body>
    </html>
  );
}
