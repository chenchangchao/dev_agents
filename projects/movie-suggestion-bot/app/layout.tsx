import type { Metadata } from "next";
import "./globals.css";
import { CopilotKit } from "@copilotkit/react-core";

export const metadata: Metadata = {
  title: "Movie Suggestion Bot",
  description: "A bot that recommends movies based on genre and mood.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link
          rel="icon"
          href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>📽️</text></svg>"
        />
      </head>
      <body>
        <CopilotKit
          runtimeUrl="/api/copilotkit"
          forwardedParameters={{ temperature: 0.4 }}
        >
          {children}
        </CopilotKit>
      </body>
    </html>
  );
}
