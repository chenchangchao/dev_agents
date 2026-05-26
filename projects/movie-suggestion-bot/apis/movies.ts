"use server";

import { Movie } from "@/types";

type OmdbSearchResponse =
  | {
      Response: "True";
      Search: Movie[];
    }
  | {
      Response: "False";
      Error: string;
    };

type OmdbTitleResponse =
  | (Movie & {
      Response: "True";
    })
  | {
      Response: "False";
      Error: string;
    };

function getCandidateTitleQueries(query: string): string[] {
  const normalizedQuery = query.trim();
  const quotedTitles = Array.from(
    normalizedQuery.matchAll(/["“”']([^"“”']{2,})["“”']/g),
    (match) => match[1].trim()
  );
  const delimitedTitles = normalizedQuery
    .split(/\n|,|，|;|；|\||、/)
    .map((item) => item.replace(/^\d+[.)、-]?\s*/, "").trim())
    .filter((item) => item.length > 1);

  const candidates = quotedTitles.length > 0 ? quotedTitles : delimitedTitles;
  const uniqueCandidates = Array.from(new Set(candidates));

  return (uniqueCandidates.length > 0 ? uniqueCandidates : [normalizedQuery])
    .filter(Boolean)
    .slice(0, 10);
}

async function requestOmdb<T>(
  params: Record<string, string>,
  apiKey: string
): Promise<T | null> {
  const searchParams = new URLSearchParams({
    apikey: apiKey,
    ...params,
  });

  try {
    const response = await fetch(`https://www.omdbapi.com/?${searchParams}`, {
      cache: "no-store",
    });

    if (!response.ok) {
      console.warn(`OMDB request failed with status ${response.status}`);
      return null;
    }

    return (await response.json()) as T;
  } catch (error) {
    console.error(
      "Error fetching movies:",
      error instanceof Error ? error.message : error
    );
    return null;
  }
}

export async function fetchMovies({
  query,
}: {
  query: string;
}): Promise<Movie[]> {
  const apiKey = process.env.OMDB_API_KEY;

  if (!apiKey) {
    console.warn("OMDB_API_KEY is not configured");
    return [];
  }

  const moviesById = new Map<string, Movie>();
  const titleQueries = getCandidateTitleQueries(query);

  for (const title of titleQueries) {
    const result = await requestOmdb<OmdbTitleResponse>({ t: title }, apiKey);

    if (result?.Response === "True") {
      moviesById.set(result.imdbID, {
        Title: result.Title,
        Year: result.Year,
        imdbID: result.imdbID,
        Poster: result.Poster,
      });
    } else if (result?.Response === "False") {
      console.warn(`OMDB title lookup failed for "${title}": ${result.Error}`);
    }
  }

  if (moviesById.size === 0) {
    for (const title of titleQueries) {
      if (moviesById.size >= 8) {
        break;
      }

      const result = await requestOmdb<OmdbSearchResponse>({ s: title }, apiKey);
      if (result?.Response === "True") {
        for (const movie of result.Search) {
          moviesById.set(movie.imdbID, movie);
        }
      }
    }
  }

  return Array.from(moviesById.values()).slice(0, 8);
}
