import "dotenv/config";

type OmdbSearchResponse =
  | {
      Response: "True";
      Search: Array<{
        Title: string;
        Year: string;
        imdbID: string;
        Type: string;
      }>;
      totalResults: string;
    }
  | {
      Response: "False";
      Error: string;
    };

type OmdbTitleResponse =
  | {
      Response: "True";
      Title: string;
      Year: string;
      imdbID: string;
      Runtime: string;
      Genre: string;
    }
  | {
      Response: "False";
      Error: string;
    };

const apiKey = process.env.OMDB_API_KEY;

if (!apiKey) {
  console.error("OMDB_API_KEY is not configured");
  process.exit(1);
}

const omdbApiKey = apiKey;

async function requestOmdb<T>(params: Record<string, string>): Promise<T> {
  const searchParams = new URLSearchParams({
    apikey: omdbApiKey,
    ...params,
  });

  const response = await fetch(`https://www.omdbapi.com/?${searchParams}`);

  if (!response.ok) {
    throw new Error(`OMDB HTTP request failed: ${response.status}`);
  }

  return (await response.json()) as T;
}

async function main() {
  console.log("Testing OMDB API connectivity...");

  const searchResult = await requestOmdb<OmdbSearchResponse>({
    s: "about time",
  });

  if (searchResult.Response !== "True") {
    throw new Error(`OMDB search failed: ${searchResult.Error}`);
  }

  console.log(
    `Search OK: ${searchResult.Search.length} results, total=${searchResult.totalResults}`
  );
  console.log(
    `First result: ${searchResult.Search[0].Title} (${searchResult.Search[0].Year})`
  );

  const titleResult = await requestOmdb<OmdbTitleResponse>({
    t: "About Time",
  });

  if (titleResult.Response !== "True") {
    throw new Error(`OMDB title lookup failed: ${titleResult.Error}`);
  }

  console.log(
    `Title OK: ${titleResult.Title} (${titleResult.Year}), ${titleResult.Runtime}, ${titleResult.Genre}`
  );
  console.log("OMDB API connectivity test passed.");
}

main().catch((error) => {
  console.error(
    "OMDB API connectivity test failed:",
    error instanceof Error ? error.message : error
  );
  process.exit(1);
});
