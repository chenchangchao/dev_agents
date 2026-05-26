

## Create a Movie Suggestion Bot
- This tutorial will guide you through setting up a movie suggestion bot that uses natural language to detect your mood and genre preferences to suggest movies accordingly.

## Setup

```bash
bun create next-app@latest movie-suggestion-bot
cd movie-suggestion-bot
bunx --bun shadcn@latest init
bunx --bun shadcn@latest add card badge button
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## resources

- https://www.omdbapi.com/
- https://www.omdbapi.com/apikey.aspx