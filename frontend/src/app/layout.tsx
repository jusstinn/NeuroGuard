import type { Metadata } from "next";
import "./globals.css";
import { Providers } from "./providers";

export const metadata: Metadata = {
  title: "NeuroGuard | AI Safety Evaluation Platform",
  description: "Next-gen evaluation platform for detecting sandbagging, sycophancy, dark patterns, and malicious plasticity in fine-tuned LLMs",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="font-sans bg-neuro-grid min-h-screen antialiased">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
