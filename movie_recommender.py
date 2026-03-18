"""Content-based movie recommendation system using titles and credits datasets.

The script is intentionally flexible because coursework datasets often appear in
one of two formats:
1. TMDB-style files where `genres`, `cast`, and `crew` are already stored as
   JSON-like lists.
2. Row-based credits files where each person-credit is stored as a separate row
   with columns such as `name`, `character`, and `role`, while the titles file
   uses `description` instead of `overview`.

Both variants are normalized into a common schema before the recommendation
pipeline is applied.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


FINAL_COLUMNS: Sequence[str] = ("id", "title", "overview", "genres", "cast", "crew")
TITLE_ALIASES = {"overview": ("overview", "description"), "title": ("title", "name")}


def load_datasets(titles_path: str | Path, credits_path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the titles and credits datasets using pandas."""
    titles = pd.read_csv(titles_path)
    credits = pd.read_csv(credits_path)
    return titles, credits


def first_available_column(frame: pd.DataFrame, candidates: Sequence[str]) -> pd.Series:
    """Return the first matching column from a dataframe."""
    for column in candidates:
        if column in frame.columns:
            return frame[column]
    raise KeyError(f"None of the expected columns were found: {list(candidates)}")


def normalize_titles_dataset(titles: pd.DataFrame) -> pd.DataFrame:
    """Normalize the titles dataset to provide id, title, overview, and genres."""
    if "id" not in titles.columns:
        raise KeyError("The titles dataset must contain an 'id' column.")

    normalized = pd.DataFrame({"id": titles["id"]})
    normalized["title"] = first_available_column(titles, TITLE_ALIASES["title"])
    normalized["overview"] = first_available_column(titles, TITLE_ALIASES["overview"])

    if "genres" in titles.columns:
        normalized["genres"] = titles["genres"]
    elif "genre" in titles.columns:
        normalized["genres"] = titles["genre"]
    else:
        raise KeyError("The titles dataset must contain either 'genres' or 'genre'.")

    return normalized


def aggregate_credits_dataset(credits: pd.DataFrame) -> pd.DataFrame:
    """Aggregate row-based credits data into cast and crew lists by movie id."""
    required = {"id", "name"}
    if not required.issubset(credits.columns):
        raise KeyError("Row-based credits data must contain at least 'id' and 'name' columns.")

    working = credits.copy()
    working["role"] = working["role"].fillna("") if "role" in working.columns else ""
    working["character"] = working["character"].fillna("") if "character" in working.columns else ""

    def is_cast_row(row: pd.Series) -> bool:
        role_value = str(row.get("role", "")).strip().lower()
        character_value = str(row.get("character", "")).strip()
        return bool(character_value) or role_value in {"actor", "actress", "cast", "self"}

    def is_director_row(row: pd.Series) -> bool:
        return str(row.get("role", "")).strip().lower() == "director"

    cast_rows = working[working.apply(is_cast_row, axis=1)]
    crew_rows = working[~working.apply(is_cast_row, axis=1)]

    cast_agg = (
        cast_rows.groupby("id", sort=False)["name"]
        .apply(lambda names: [{"name": name} for name in names.dropna().tolist()])
        .rename("cast")
    )

    crew_agg = (
        crew_rows.groupby("id", sort=False)
        .apply(
            lambda frame: [
                {
                    "name": row["name"],
                    "job": row["role"] if str(row.get("role", "")).strip() else "Crew",
                }
                for _, row in frame.iterrows()
                if pd.notna(row.get("name"))
            ]
        )
        .rename("crew")
    )

    # Preserve director information even if the dataset contains only cast-like rows.
    director_rows = working[working.apply(is_director_row, axis=1)]
    if not director_rows.empty:
        director_agg = (
            director_rows.groupby("id", sort=False)["name"]
            .apply(lambda names: [{"name": name, "job": "Director"} for name in names.dropna().tolist()])
            .rename("director_entries")
        )
    else:
        director_agg = pd.Series(dtype=object, name="director_entries")

    credits_agg = pd.concat([cast_agg, crew_agg, director_agg], axis=1).reset_index()
    credits_agg["cast"] = credits_agg["cast"].apply(lambda value: value if isinstance(value, list) else [])
    credits_agg["crew"] = credits_agg["crew"].apply(lambda value: value if isinstance(value, list) else [])
    if "director_entries" in credits_agg.columns:
        credits_agg["crew"] = credits_agg.apply(
            lambda row: row["crew"] + (row["director_entries"] if isinstance(row["director_entries"], list) else []),
            axis=1,
        )
        credits_agg.drop(columns=["director_entries"], inplace=True)

    return credits_agg[["id", "cast", "crew"]]


def normalize_credits_dataset(credits: pd.DataFrame) -> pd.DataFrame:
    """Normalize credits data to provide `id`, `cast`, and `crew`."""
    if "id" not in credits.columns:
        raise KeyError("The credits dataset must contain an 'id' column.")

    if {"cast", "crew"}.issubset(credits.columns):
        return credits[["id", "cast", "crew"]].copy()

    return aggregate_credits_dataset(credits)


def merge_datasets(titles: pd.DataFrame, credits: pd.DataFrame) -> pd.DataFrame:
    """Merge normalized titles and credits datasets on `id`."""
    normalized_titles = normalize_titles_dataset(titles)
    normalized_credits = normalize_credits_dataset(credits)

    movies = normalized_titles.merge(normalized_credits, on="id", how="inner")
    missing_columns = [column for column in FINAL_COLUMNS if column not in movies.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns after normalization and merge: {missing_columns}")

    movies = movies.loc[:, FINAL_COLUMNS].copy()
    movies.dropna(subset=["id", "title", "overview", "genres"], inplace=True)
    movies["cast"] = movies["cast"].apply(lambda value: value if value is not None else [])
    movies["crew"] = movies["crew"].apply(lambda value: value if value is not None else [])
    return movies


def parse_json_like(value) -> List:
    """Safely parse stringified JSON-like lists from the dataset."""
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return [item.strip() for item in text.split(",") if item.strip()]

    if isinstance(parsed, list):
        return parsed
    return [parsed]


def extract_names(items: Iterable, limit: int | None = None) -> List[str]:
    """Extract names from a list of dictionaries or plain strings."""
    names: List[str] = []
    for item in items:
        if isinstance(item, dict):
            value = item.get("name") or item.get("title")
        else:
            value = str(item)

        if value:
            names.append(str(value))
            if limit is not None and len(names) >= limit:
                break
    return names


def extract_director(items: Iterable) -> List[str]:
    """Extract the director's name from a crew list."""
    for item in items:
        if isinstance(item, dict) and str(item.get("job", "")).strip().lower() == "director" and item.get("name"):
            return [str(item["name"])]
    return []


def normalize_tokens(tokens: Iterable[str]) -> List[str]:
    """Remove spaces from multi-word names and convert each token to lowercase."""
    return [str(token).replace(" ", "").lower() for token in tokens if str(token).strip()]


def preprocess_movies(movies: pd.DataFrame) -> pd.DataFrame:
    """Perform feature engineering and create the `tags` column."""
    processed = movies.copy()

    processed["genres"] = processed["genres"].apply(parse_json_like).apply(extract_names)
    processed["cast"] = processed["cast"].apply(parse_json_like).apply(lambda items: extract_names(items, limit=3))
    processed["crew"] = processed["crew"].apply(parse_json_like).apply(extract_director)
    processed["overview"] = processed["overview"].fillna("").apply(lambda text: str(text).split())

    for column in ("overview", "genres", "cast", "crew"):
        processed[column] = processed[column].apply(normalize_tokens)

    processed["tags"] = processed["overview"] + processed["genres"] + processed["cast"] + processed["crew"]
    processed["tags"] = processed["tags"].apply(lambda tokens: " ".join(tokens))
    processed = processed[processed["tags"].str.strip() != ""].reset_index(drop=True)

    return processed[["id", "title", "tags"]]


def build_similarity_matrix(movies: pd.DataFrame):
    """Vectorize tags and compute the cosine similarity matrix."""
    if movies.empty:
        raise ValueError("No movies are available after preprocessing.")

    vectorizer = CountVectorizer(max_features=5000, stop_words="english")
    vectors = vectorizer.fit_transform(movies["tags"]).toarray()
    similarity = cosine_similarity(vectors)
    return vectorizer, similarity


def recommend(movie_name: str, movies: pd.DataFrame, similarity) -> List[str]:
    """Return the top 5 similar movie titles for the provided movie name."""
    matches = movies[movies["title"].str.lower() == movie_name.lower()]
    if matches.empty:
        raise ValueError(f"Movie '{movie_name}' not found in the dataset.")

    movie_index = matches.index[0]
    distances = list(enumerate(similarity[movie_index]))
    ranked = sorted(distances, key=lambda item: item[1], reverse=True)

    recommendations: List[str] = []
    for index, _score in ranked[1:]:
        recommendations.append(movies.iloc[index]["title"])
        if len(recommendations) == 5:
            break
    return recommendations


def build_recommender(titles_path: str | Path = "titles.csv", credits_path: str | Path = "credits.csv"):
    """Create the processed movies dataframe and similarity matrix."""
    titles, credits = load_datasets(titles_path, credits_path)
    merged_movies = merge_datasets(titles, credits)
    processed_movies = preprocess_movies(merged_movies)
    _vectorizer, similarity = build_similarity_matrix(processed_movies)
    return processed_movies, similarity


def main() -> None:
    """Run the movie recommendation workflow and print a sample output."""
    try:
        movies, similarity = build_recommender("titles.csv", "credits.csv")
        movie_name = "Avatar"
        recommendations = recommend(movie_name, movies, similarity)

        print(f"Top 5 movies similar to '{movie_name}':")
        for title in recommendations:
            print(title)
    except FileNotFoundError as error:
        print(f"Dataset file not found: {error}")
        print("Place 'titles.csv' and 'credits.csv' in the same directory as this script.")
    except Exception as error:  # pragma: no cover - helpful for quick debugging in assignments
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
