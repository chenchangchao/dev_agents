"use client";

import { Movie } from "@/types/index";
import { fetchMovies } from "@/apis/movies";
import { CopilotChat } from "@copilotkit/react-ui";
import MovieCard from "@/app/_components/movie-card";
import { useCopilotAction } from "@copilotkit/react-core";
import NoMoviesCard from "@/app/_components/no-movies-card";
import { Spinner } from "@/components/ui-expansions/spinner";

import "@copilotkit/react-ui/styles.css";

export default function Home() {
	useCopilotAction({
		name: "fetchMovies",
		description:
			"Hydrates movie recommendations with OMDB metadata. Before calling this tool, infer 6-10 real, well-known movie titles that match the user's mood, genre, language, era, and constraints. Pass those titles in English, separated by pipes. Do not pass raw mood words such as 'healing heartbreak warm'.",
		parameters: [
			{
				name: "query",
				type: "string",
				description:
					"6-10 real English movie titles separated by pipes, for example: About Time | Her | Amelie",
				required: true,
			},
		],
		handler: fetchMovies,
		render: ({ status, result }) => {
			if (status === "executing" || status === "inProgress") {
				return <Spinner size="large" />;
			} else if (status === "complete" && result.length > 0) {
				return (
					<div className="grid grid-cols-4 gap-4">
						{result.map((movie: Movie) => <MovieCard key={movie.imdbID} movie={movie} />)}
					</div>
				);
			} else {
				return <NoMoviesCard />
			}
		},
	});

	return (
		<div className="w-full h-screen">
			<CopilotChat
				className="w-full h-full"
				labels={{
					title: "Movie Suggestion Bot",
					initial: "Hello! 👋 What type of movie are you in the mood for?",
				}}
				instructions="When the user asks for movie recommendations, infer suitable real movie titles from their mood, genre, language, era, and constraints. Then call fetchMovies with 6-10 English movie titles separated by pipes so the app can render cards. If movie cards are returned, keep the text response brief and let the cards carry the recommendations."
			/>
		</div>
	);
}
