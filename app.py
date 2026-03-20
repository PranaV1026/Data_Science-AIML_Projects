import ast
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PLACEHOLDER_POSTER = "https://placehold.co/300x450/111111/E50914?text=Movie+Poster"
TITLE_ALIASES = {"overview": ("overview", "description"), "title": ("title", "name")}


st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="wide")


def apply_netflix_theme() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(180deg, #0b0b0b 0%, #141414 45%, #1b1b1b 100%);
                color: #f5f5f5;
            }
            .main-title {
                font-size: 3rem;
                font-weight: 800;
                color: #E50914;
                margin-bottom: 0;
            }
            .subtitle {
                font-size: 1.15rem;
                color: #d6d6d6;
                margin-top: 0.25rem;
                margin-bottom: 2rem;
            }
            .movie-card {
                background: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 18px;
                padding: 14px;
                text-align: center;
                min-height: 100%;
                box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
            }
            .movie-title {
                color: #ffffff;
                font-size: 1rem;
                font-weight: 700;
                margin-top: 0.85rem;
                line-height: 1.4;
            }
            .footer {
                text-align: center;
                color: #b3b3b3;
                margin-top: 2.5rem;
                padding: 1rem 0;
                border-top: 1px solid rgba(255,255,255,0.08);
            }
            .stButton>button {
                background: #E50914;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 0.65rem 1.5rem;
                font-weight: 700;
                width: 100%;
            }
            .stButton>button:hover {
                background: #b20710;
                color: white;
            }
            [data-testid="stSidebar"] {
                background: #111111;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_datasets(titles_path: str = "titles.csv", credits_path: str = "credits.csv") -> tuple[pd.DataFrame, pd.DataFrame]:
    titles = pd.read_csv(Path(titles_path))
    credits = pd.read_csv(Path(credits_path))
    return titles, credits


def first_available_column(frame: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    for column in candidates:
        if column in frame.columns:
            return frame[column]
    raise KeyError(f"Expected one of these columns: {list(candidates)}")


def normalize_titles_dataset(titles: pd.DataFrame) -> pd.DataFrame:
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
    working = credits.copy()
    if "role" not in working.columns:
        working["role"] = ""
    if "character" not in working.columns:
        working["character"] = ""

    working["role"] = working["role"].fillna("")
    working["character"] = working["character"].fillna("")

    def is_cast_row(row: pd.Series) -> bool:
        role_value = str(row.get("role", "")).strip().lower()
        character_value = str(row.get("character", "")).strip()
        return bool(character_value) or role_value in {"actor", "actress", "cast", "self"}

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

    aggregated = pd.concat([cast_agg, crew_agg], axis=1).reset_index()
    aggregated["cast"] = aggregated["cast"].apply(lambda value: value if isinstance(value, list) else [])
    aggregated["crew"] = aggregated["crew"].apply(lambda value: value if isinstance(value, list) else [])
    return aggregated[["id", "cast", "crew"]]


def normalize_credits_dataset(credits: pd.DataFrame) -> pd.DataFrame:
    if {"cast", "crew"}.issubset(credits.columns):
        return credits[["id", "cast", "crew"]].copy()
    return aggregate_credits_dataset(credits)


def parse_json_like(value) -> List:
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return [item.strip() for item in text.split(",") if item.strip()]

    if isinstance(parsed, list):
        return parsed
    return [parsed]


def extract_names(items: Iterable, limit: int | None = None) -> List[str]:
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
    for item in items:
        if isinstance(item, dict) and str(item.get("job", "")).strip().lower() == "director":
            if item.get("name"):
                return [str(item["name"])]
    return []


def normalize_tokens(tokens: Iterable[str]) -> List[str]:
    return [str(token).replace(" ", "").lower() for token in tokens if str(token).strip()]


@st.cache_data(show_spinner=False)
def build_recommender(titles_path: str = "titles.csv", credits_path: str = "credits.csv"):
    titles, credits = load_datasets(titles_path, credits_path)
    titles = normalize_titles_dataset(titles)
    credits = normalize_credits_dataset(credits)

    movies = titles.merge(credits, on="id", how="inner")
    movies = movies[["title", "overview", "genres", "cast", "crew"]].dropna().copy()

    movies["genres"] = movies["genres"].apply(parse_json_like).apply(extract_names)
    movies["cast"] = movies["cast"].apply(parse_json_like).apply(lambda items: extract_names(items, limit=3))
    movies["crew"] = movies["crew"].apply(parse_json_like).apply(extract_director)
    movies["overview"] = movies["overview"].apply(lambda text: str(text).split())

    for column in ["overview", "genres", "cast", "crew"]:
        movies[column] = movies[column].apply(normalize_tokens)

    movies["tags"] = movies["overview"] + movies["genres"] + movies["cast"] + movies["crew"]
    movies["tags"] = movies["tags"].apply(lambda tokens: " ".join(tokens))
    movies = movies[movies["tags"].str.strip() != ""].reset_index(drop=True)

    vectorizer = CountVectorizer(max_features=5000, stop_words="english")
    vectors = vectorizer.fit_transform(movies["tags"]).toarray()
    similarity = cosine_similarity(vectors)
    return movies, similarity


def recommend(movie_name: str, movies: pd.DataFrame, similarity) -> List[str]:
    movie_name = movie_name.strip().lower()
    if not movie_name:
        raise ValueError("Please enter or select a movie name.")

    exact_match = movies[movies["title"].str.lower() == movie_name]
    if exact_match.empty:
        partial_match = movies[movies["title"].str.lower().str.contains(movie_name, na=False)]
        if partial_match.empty:
            raise ValueError("Movie not found. Please try another title from the dataset.")
        movie_index = partial_match.index[0]
    else:
        movie_index = exact_match.index[0]

    distances = list(enumerate(similarity[movie_index]))
    ranked_movies = sorted(distances, key=lambda item: item[1], reverse=True)[1:6]
    return [movies.iloc[index]["title"] for index, _ in ranked_movies]


apply_netflix_theme()

st.markdown('<div class="main-title">🎬 Movie Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Get similar movies instantly</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("About Project")
    st.write(
        "This content-based movie recommender uses movie metadata like overview, genres, cast, and director "
        "to suggest similar titles without calling any external movie API."
    )
    st.header("Tech Stack")
    st.markdown(
        "- Python\n"
        "- Pandas\n"
        "- Scikit-learn\n"
        "- Streamlit\n"
        "- CountVectorizer\n"
        "- Cosine Similarity"
    )

try:
    movies, similarity = build_recommender()
    movie_list = sorted(movies["title"].astype(str).unique().tolist())

    selected_movie = st.selectbox(
        "Search or select a movie",
        options=movie_list,
        index=None,
        placeholder="Type to search movies...",
    )

    manual_movie = st.text_input("Or enter movie name manually")

    if st.button("Recommend"):
        movie_query = manual_movie if manual_movie.strip() else (selected_movie or "")
        with st.spinner("Finding similar movies for you..."):
            recommendations = recommend(movie_query, movies, similarity)

        st.subheader("Recommended for you")
        columns = st.columns(5)
        for column, title in zip(columns, recommendations):
            with column:
                st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                st.image(PLACEHOLDER_POSTER, use_container_width=True)
                st.markdown(f'<div class="movie-title">{title}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
except FileNotFoundError:
    st.error("titles.csv or credits.csv was not found. Please place both files in the project folder.")
except KeyError as error:
    st.error(f"Dataset schema issue: {error}")
except ValueError as error:
    st.error(str(error))
except Exception as error:
    st.error(f"Something went wrong: {error}")

st.markdown('<div class="footer">Built with ❤️ using Machine Learning</div>', unsafe_allow_html=True)
