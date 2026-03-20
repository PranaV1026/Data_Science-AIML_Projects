import ast
import html
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


POSTER_URL = "https://placehold.co/320x480/111111/E50914?text=NETFLIX+STYLE"
TITLE_ALIASES = {"overview": ("overview", "description"), "title": ("title", "name")}
FAKE_FILTERS = ["2025-2020", "2019-2015", "2014-2010", "2009-2005", "Classics"]


st.set_page_config(page_title="🎬 Movie Recommendation System", page_icon="🎬", layout="wide")


def inject_css() -> None:
    st.markdown(
        """
        <style>
            :root {
                --bg: #0f0f0f;
                --panel: #171717;
                --panel-soft: rgba(255, 255, 255, 0.04);
                --text: #f5f5f1;
                --muted: #b3b3b3;
                --accent: #e50914;
                --accent-dark: #b20710;
                --border: rgba(255, 255, 255, 0.08);
            }
            .stApp {
                background:
                    radial-gradient(circle at top, rgba(229, 9, 20, 0.18), transparent 22%),
                    linear-gradient(180deg, #0f0f0f 0%, #111111 35%, #0f0f0f 100%);
                color: var(--text);
            }
            header[data-testid="stHeader"],
            #MainMenu,
            footer {
                visibility: hidden;
            }
            section[data-testid="stSidebar"] {
                background: #111111;
                border-right: 1px solid var(--border);
            }
            div.block-container {
                padding-top: 1.4rem;
                padding-bottom: 2rem;
                max-width: 1380px;
            }
            .headline {
                font-size: 3.4rem;
                font-weight: 900;
                letter-spacing: -0.04em;
                color: #ffffff;
                margin-bottom: 0.35rem;
                text-align: center;
            }
            .headline span {
                color: var(--accent);
            }
            .subheadline {
                color: var(--muted);
                font-size: 1.08rem;
                text-align: center;
                margin-bottom: 1.5rem;
            }
            .filters-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.75rem;
                justify-content: center;
                margin: 0.5rem 0 1.75rem;
            }
            .filter-chip {
                background: rgba(255,255,255,0.06);
                color: #f7f7f7;
                border: 1px solid rgba(255,255,255,0.08);
                padding: 0.65rem 1rem;
                border-radius: 999px;
                font-size: 0.92rem;
                font-weight: 700;
                transition: all 0.25s ease;
            }
            .filter-chip:hover {
                transform: translateY(-2px);
                border-color: rgba(229,9,20,0.45);
                box-shadow: 0 10px 22px rgba(229,9,20,0.18);
            }
            .section-title {
                font-size: 1.45rem;
                font-weight: 800;
                color: white;
                margin: 0.2rem 0 1rem;
            }
            .hero-row,
            .movie-row {
                display: flex;
                gap: 1rem;
                overflow-x: auto;
                padding-bottom: 0.35rem;
                scroll-behavior: smooth;
            }
            .hero-row::-webkit-scrollbar,
            .movie-row::-webkit-scrollbar {
                display: none;
            }
            .hero-card,
            .movie-card {
                position: relative;
                flex: 0 0 auto;
                border-radius: 18px;
                overflow: hidden;
                background: #151515;
                border: 1px solid rgba(255,255,255,0.08);
                transition: transform 0.35s ease, box-shadow 0.35s ease;
                box-shadow: 0 14px 36px rgba(0, 0, 0, 0.35);
            }
            .hero-card {
                width: 250px;
                height: 360px;
            }
            .movie-card {
                width: 220px;
                height: 330px;
            }
            .hero-card:hover,
            .movie-card:hover {
                transform: translateY(-8px) scale(1.04);
                box-shadow: 0 22px 45px rgba(0, 0, 0, 0.48);
            }
            .hero-card img,
            .movie-card img {
                width: 100%;
                height: 100%;
                object-fit: cover;
                display: block;
            }
            .overlay {
                position: absolute;
                inset: 0;
                background: linear-gradient(180deg, rgba(0,0,0,0.02) 20%, rgba(0,0,0,0.88) 100%);
                display: flex;
                align-items: end;
                padding: 1rem;
            }
            .card-title {
                color: #fff;
                font-weight: 800;
                font-size: 1rem;
                line-height: 1.3;
                text-shadow: 0 2px 16px rgba(0,0,0,0.7);
            }
            .hero-shell {
                background: linear-gradient(135deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 26px;
                padding: 1.2rem;
                margin-bottom: 1.75rem;
                backdrop-filter: blur(8px);
            }
            .search-shell {
                background: rgba(255,255,255,0.03);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 24px;
                padding: 1.2rem 1.25rem 0.8rem;
                margin: 0 auto 1.8rem;
                max-width: 960px;
            }
            .search-caption {
                text-align: center;
                color: var(--muted);
                margin-bottom: 0.9rem;
                font-size: 0.98rem;
            }
            [data-testid="stTextInputRootElement"] input {
                background: #161616 !important;
                color: white !important;
                border: 1px solid rgba(255,255,255,0.08) !important;
                border-radius: 999px !important;
                min-height: 3.2rem !important;
                padding-left: 1rem !important;
            }
            [data-testid="stTextInputRootElement"] input:focus {
                border-color: rgba(229,9,20,0.65) !important;
                box-shadow: 0 0 0 1px rgba(229,9,20,0.2) !important;
            }
            .stButton > button {
                width: 100%;
                min-height: 3.2rem;
                border-radius: 999px;
                background: linear-gradient(180deg, var(--accent) 0%, var(--accent-dark) 100%);
                border: none;
                color: #fff;
                font-weight: 800;
                letter-spacing: 0.01em;
                box-shadow: 0 12px 24px rgba(229,9,20,0.26);
            }
            .stButton > button:hover {
                transform: translateY(-1px);
                background: linear-gradient(180deg, #f40d19 0%, var(--accent-dark) 100%);
                color: #fff;
            }
            .footer {
                text-align: center;
                color: var(--muted);
                padding: 1.25rem 0 0.35rem;
                margin-top: 2rem;
                border-top: 1px solid rgba(255,255,255,0.06);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_data(titles_path: str = "titles.csv", credits_path: str = "credits.csv") -> Tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(Path(titles_path)), pd.read_csv(Path(credits_path))


def first_available_column(frame: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    for column in candidates:
        if column in frame.columns:
            return frame[column]
    raise KeyError(f"Expected one of these columns: {list(candidates)}")


def normalize_titles(titles: pd.DataFrame) -> pd.DataFrame:
    if "id" not in titles.columns:
        raise KeyError("titles.csv must contain an 'id' column.")

    normalized = pd.DataFrame({"id": titles["id"]})
    normalized["title"] = first_available_column(titles, TITLE_ALIASES["title"])
    normalized["overview"] = first_available_column(titles, TITLE_ALIASES["overview"])

    if "genres" in titles.columns:
        normalized["genres"] = titles["genres"]
    elif "genre" in titles.columns:
        normalized["genres"] = titles["genre"]
    else:
        raise KeyError("titles.csv must contain either 'genres' or 'genre'.")

    return normalized


def aggregate_credits(credits: pd.DataFrame) -> pd.DataFrame:
    if "id" not in credits.columns or "name" not in credits.columns:
        raise KeyError("credits.csv must contain 'id' and 'name' columns when cast/crew columns are absent.")

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
        .apply(lambda names: [{"name": value} for value in names.dropna().tolist()])
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

    normalized = pd.concat([cast_agg, crew_agg], axis=1).reset_index()
    normalized["cast"] = normalized["cast"].apply(lambda value: value if isinstance(value, list) else [])
    normalized["crew"] = normalized["crew"].apply(lambda value: value if isinstance(value, list) else [])
    return normalized[["id", "cast", "crew"]]


def normalize_credits(credits: pd.DataFrame) -> pd.DataFrame:
    if "id" not in credits.columns:
        raise KeyError("credits.csv must contain an 'id' column.")
    if {"cast", "crew"}.issubset(credits.columns):
        return credits[["id", "cast", "crew"]].copy()
    return aggregate_credits(credits)


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
    except (SyntaxError, ValueError):
        return [token.strip() for token in text.split(",") if token.strip()]

    return parsed if isinstance(parsed, list) else [parsed]


def extract_names(items: Iterable, limit: int | None = None) -> List[str]:
    extracted: List[str] = []
    for item in items:
        if isinstance(item, dict):
            value = item.get("name") or item.get("title")
        else:
            value = str(item)
        if value:
            extracted.append(str(value))
        if limit is not None and len(extracted) >= limit:
            break
    return extracted


def extract_director(items: Iterable) -> List[str]:
    for item in items:
        if isinstance(item, dict) and str(item.get("job", "")).strip().lower() == "director" and item.get("name"):
            return [str(item["name"])]
    return []


def normalize_tokens(tokens: Iterable[str]) -> List[str]:
    return [str(token).replace(" ", "").lower() for token in tokens if str(token).strip()]


@st.cache_data(show_spinner=False)
def preprocess() -> Tuple[pd.DataFrame, object]:
    titles, credits = load_data()
    titles = normalize_titles(titles)
    credits = normalize_credits(credits)

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
    normalized_name = movie_name.strip().lower()
    if not normalized_name:
        raise ValueError("Please enter a movie title.")

    matches = movies[movies["title"].astype(str).str.lower() == normalized_name]
    if matches.empty:
        matches = movies[movies["title"].astype(str).str.lower().str.contains(normalized_name, na=False)]
    if matches.empty:
        raise ValueError("Movie not found. Try another title from the catalog.")

    movie_index = matches.index[0]
    ranked = sorted(enumerate(similarity[movie_index]), key=lambda item: item[1], reverse=True)
    return [movies.iloc[index]["title"] for index, _ in ranked[1:6]]


def build_movie_row(titles: List[str], card_class: str = "movie-card") -> str:
    cards = []
    for title in titles:
        safe_title = html.escape(str(title))
        cards.append(
            f"""
            <div class="{card_class}">
                <img src="{POSTER_URL}" alt="{safe_title}">
                <div class="overlay">
                    <div class="card-title">{safe_title}</div>
                </div>
            </div>
            """
        )
    row_class = "hero-row" if card_class == "hero-card" else "movie-row"
    return f'<div class="{row_class}">' + "".join(cards) + "</div>"


def render_filters() -> None:
    chips = "".join(f'<div class="filter-chip">{html.escape(label)}</div>' for label in FAKE_FILTERS)
    st.markdown(f'<div class="filters-row">{chips}</div>', unsafe_allow_html=True)


def render_hero(movies: pd.DataFrame) -> None:
    spotlight_titles = movies["title"].astype(str).head(12).tolist()
    st.markdown('<div class="hero-shell"><div class="section-title">Trending Now</div>', unsafe_allow_html=True)
    st.markdown(build_movie_row(spotlight_titles, card_class="hero-card"), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_recommendations(titles: List[str]) -> None:
    st.markdown('<div class="section-title">Recommended for You</div>', unsafe_allow_html=True)
    st.markdown(build_movie_row(titles, card_class="movie-card"), unsafe_allow_html=True)


def main() -> None:
    inject_css()

    st.markdown('<div class="headline"><span>🎬</span> Movie Recommendation System</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheadline">Get similar movies instantly</div>', unsafe_allow_html=True)
    render_filters()

    with st.sidebar:
        st.header("About Project")
        st.write(
            "A Netflix-style content-based recommendation app built using local movie metadata only. "
            "It combines genres, cast, director, and overview text to find similar movies."
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
        movies, similarity = preprocess()
        render_hero(movies)

        st.markdown('<div class="search-shell">', unsafe_allow_html=True)
        st.markdown('<div class="search-caption">Search the catalog and discover your next binge-worthy title.</div>', unsafe_allow_html=True)
        left, center, right = st.columns([1.2, 5, 1.4])
        with center:
            search_query = st.text_input(
                label="Movie Search",
                placeholder="What's your taste?",
                label_visibility="collapsed",
            )
        with right:
            trigger = st.button("Recommend")
        st.markdown('</div>', unsafe_allow_html=True)

        if trigger:
            with st.spinner("Curating a premium row of recommendations..."):
                recommendations = recommend(search_query, movies, similarity)
            render_recommendations(recommendations)
        else:
            starter_titles = movies["title"].astype(str).iloc[12:17].tolist()
            if starter_titles:
                render_recommendations(starter_titles)

    except FileNotFoundError:
        st.error("titles.csv or credits.csv was not found. Please place both files in the project folder.")
    except (KeyError, ValueError) as error:
        st.error(str(error))
    except Exception as error:
        st.error(f"Something went wrong: {error}")

    st.markdown('<div class="footer">Built with ❤️ using Machine Learning</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
