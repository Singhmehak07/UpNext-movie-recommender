import os
import urllib.parse
from html import escape
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
import pandas as pd
import scipy.sparse
import requests
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_searchbox import st_searchbox

# 1. Page configuration with consistent branding
st.set_page_config(page_title="UpNext", page_icon="🍿", layout="wide")

# Retrieve TMDB API key securely
TMDB_API_KEY = None
if "TMDB_API_KEY" in st.secrets:
    TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
elif os.environ.get("TMDB_API_KEY"):
    TMDB_API_KEY = os.environ.get("TMDB_API_KEY")

# Display CSS style matching original cinematic dark theme
st.markdown("""
<style>
.stApp { background:#141414; }

/* Hide Streamlit platform controls */
header[data-testid="stHeader"] {
    display: none !important;
    height: 0 !important;
}
[data-testid="stToolbar"],
[data-testid="stAppToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stDeployButton"] {
    display: none !important;
    visibility: hidden !important;
}
#MainMenu {
    display: none !important;
    visibility: hidden !important;
}
footer {
    display: none !important;
    visibility: hidden !important;
}
/* Remove space left by the hidden Streamlit header */
[data-testid="stAppViewContainer"] > .main {
    padding-top: 0 !important;
}
.block-container {
    padding-top: 2rem !important;
}

.hero-row  { display:flex; align-items:center; gap:14px; margin-bottom:6px; }
.hero-icon { font-size:3rem; line-height:1; }
.hero-title {
    font-size:3.6rem; font-weight:900; letter-spacing:-0.03em; line-height:1;
    color:#E50914;
}
.hero-sub { font-size:17px; color:rgba(255,255,255,.6); margin-bottom:26px; }

.controls { display:flex; gap:16px; margin-bottom:26px; }
.btn {
    background:#E50914; border:none; border-radius:10px; padding:0 34px;
    font-size:15px; font-weight:700; color:#fff; cursor:pointer;
    transition:background .2s ease, transform .1s ease;
}
.btn:hover  { background:#f6121d; }
.btn:active { transform:scale(.97); }

.banner {
    position:relative; height:230px; border-radius:16px; overflow:hidden;
    margin-bottom:30px; border:none;
    background-size:cover; background-position:center;
}
.banner-overlay {
    position:absolute; inset:0; padding:28px 32px;
    background:rgba(20,20,20,.55);
    display:flex; flex-direction:column; justify-content:flex-end;
}
.eyebrow  { color:#E50914; font-size:13px; font-weight:800; letter-spacing:.14em; }
.banner-title { font-size:34px; font-weight:800; color:#fff; margin-top:6px; }

.section  { font-size:24px; font-weight:800; color:#fff; margin:10px 0 20px; }

.grid { display:grid; grid-template-columns:repeat(5,1fr); gap:20px; }
.card {
    background:#1c1c1c; border:1px solid rgba(255,255,255,.1);
    border-radius:14px; padding:12px;
    transition:transform .25s ease, border-color .25s ease;
}
.card:hover { transform:translateY(-6px); border-color:#E50914; }

.poster { aspect-ratio:2/3; border-radius:10px; margin-bottom:12px; overflow:hidden; }
.poster img {
    width:100%; height:100%; object-fit:cover; display:block;
    transition:transform .4s ease;
}
.card:hover .poster img { transform:scale(1.09); }
.poster-ph {
    aspect-ratio:2/3; border-radius:10px; margin-bottom:12px; background:#2a2a2a;
    display:flex; align-items:center; justify-content:center; font-size:40px;
}

.card-head { display:flex; justify-content:space-between; align-items:center; padding:0 4px; }
.rank  { font-size:18px; font-weight:800; color:#E50914; }
.badge { padding:3px 10px; border-radius:999px; font-size:12px; font-weight:800;
         background:rgba(255,255,255,.12); color:#fff; }
.m-title { font-size:15px; font-weight:700; color:#fff; margin:12px 4px 10px; line-height:1.3; }

.bar  { background:rgba(255,255,255,.15); border-radius:999px; height:6px; margin:0 4px; }
.fill { background:#ffffff; height:100%; border-radius:999px;
        box-shadow:0 0 10px rgba(255,255,255,.85); }
.pct  { color:rgba(255,255,255,.55); font-size:12px; margin:7px 4px 4px; display:block; }

.footer { text-align:center; color:rgba(255,255,255,.4); font-size:13px; margin-top:44px; }

/* =========================================================
   Portrait mobile layout via Container Queries
   Desktop remains unchanged above this container size.
   ========================================================= */
[data-testid="stMain"] {
    container-type: inline-size;
    container-name: upnext-main;
}

@container upnext-main (max-width: 768px) {

    /* Natural vertical scrolling with no sideways overflow */
    html,
    body,
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        width: 100% !important;
        max-width: 100% !important;
        overflow-x: hidden !important;
        overflow-y: auto !important;
    }

    .block-container {
        width: 100% !important;
        max-width: 100% !important;
        padding: 1.25rem 1rem 2rem !important;
        box-sizing: border-box !important;
    }

    .hero-row {
        gap: 10px;
        margin-bottom: 7px;
    }

    .hero-icon {
        font-size: 2.1rem;
        line-height: 1;
    }

    .hero-title {
        font-size: 2.6rem;
        line-height: 1;
        letter-spacing: -0.04em;
    }

    .hero-sub {
        max-width: 330px;
        margin-bottom: 20px;
        font-size: 14px;
        line-height: 1.5;
    }

    /* Full-width live search */
    [data-testid="stCustomComponentV1"],
    [data-testid="stCustomComponentV1"] iframe {
        display: block !important;
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
    }

    /* Full-width touch-friendly button */
    div.stButton,
    div.stButton > button {
        width: 100% !important;
    }

    div.stButton > button {
        min-height: 46px !important;
        border-radius: 10px !important;
        font-size: 15px !important;
        touch-action: manipulation;
    }

    .banner {
        width: 100%;
        height: 180px;
        margin-top: 18px;
        margin-bottom: 24px;
        border-radius: 12px;
        background-position: center;
        box-sizing: border-box;
    }

    .banner-overlay {
        padding: 20px;
    }

    .eyebrow {
        font-size: 10px;
        letter-spacing: 0.12em;
    }

    .banner-title {
        max-width: 100%;
        margin-top: 6px;
        font-size: 25px;
        line-height: 1.15;
        overflow-wrap: anywhere;
    }

    .section {
        margin: 6px 0 16px;
        font-size: 21px;
        line-height: 1.25;
    }

    /* Vertical recommendation feed */
    .grid {
        display: grid !important;
        grid-template-columns: 1fr !important;
        gap: 14px !important;
        width: 100% !important;
        max-width: 100% !important;
    }

    /* Poster-left compact mobile card */
    .card {
        display: grid !important;
        grid-template-columns: 105px minmax(0, 1fr) !important;
        grid-template-rows: auto auto auto auto !important;
        column-gap: 14px;
        row-gap: 0;
        align-items: start;
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;
        padding: 10px;
        border-radius: 12px;
        box-sizing: border-box;
    }

    .poster,
    .poster-ph {
        grid-column: 1;
        grid-row: 1 / 5;
        width: 105px;
        max-width: 105px;
        aspect-ratio: 2 / 3;
        margin: 0;
        border-radius: 8px;
        overflow: hidden;
    }

    .poster img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }

    .poster-ph {
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 30px;
    }

    .card-head {
        grid-column: 2;
        grid-row: 1;
        width: 100%;
        min-width: 0;
        padding: 2px 0 0;
    }

    .rank {
        font-size: 17px;
    }

    .badge {
        flex-shrink: 0;
        padding: 3px 8px;
        font-size: 11px;
    }

    .m-title {
        grid-column: 2;
        grid-row: 2;
        min-width: 0;
        margin: 14px 0 12px;
        font-size: 16px;
        line-height: 1.35;
        overflow-wrap: anywhere;
        word-break: normal;
    }

    .bar {
        grid-column: 2;
        grid-row: 3;
        width: 100%;
        min-width: 0;
        margin: 0;
        align-self: end;
    }

    .pct {
        grid-column: 2;
        grid-row: 4;
        margin: 7px 0 0;
        font-size: 12px;
    }

    /* Avoid sticky hover effects on touch screens */
    .card:hover {
        transform: none;
    }

    .card:hover .poster img {
        transform: none;
    }

    .footer {
        margin-top: 30px;
        padding: 0 12px;
        font-size: 11px;
        line-height: 1.6;
    }
}

/* Extra-small portrait phones */
@container upnext-main (max-width: 420px) {

    .block-container {
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
    }

    .hero-title {
        font-size: 2.35rem;
    }

    .hero-icon {
        font-size: 1.9rem;
    }

    .banner {
        height: 160px;
    }

    .banner-overlay {
        padding: 16px;
    }

    .banner-title {
        font-size: 22px;
    }

    .card {
        grid-template-columns: 92px minmax(0, 1fr) !important;
        column-gap: 12px;
        padding: 9px;
    }

    .poster,
    .poster-ph {
        width: 92px;
        max-width: 92px;
    }

    .m-title {
        margin-top: 11px;
        margin-bottom: 10px;
        font-size: 15px;
    }
}
</style>
""", unsafe_allow_html=True)

# Safe HTML string and URL helpers
def safe_html_text(text):
    """Safely escapes text to prevent HTML and script injection."""
    if not isinstance(text, str):
        text = str(text)
    return escape(text)

def validate_image_url(url):
    """Validates that a URL is a legitimate HTTP/HTTPS TMDB image link to prevent attribute injection."""
    if not url:
        return None
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return None
        # Only allow images loaded from the TMDB image host
        if parsed.netloc != "image.tmdb.org":
            return None
        return escape(url)
    except Exception:
        return None

# Load dataset securely from actual committed artifacts
@st.cache_data
def load_data():
    movies_path = "movies.pkl"
    vectors_path = "vectors.npz"
    if not os.path.exists(movies_path):
        raise FileNotFoundError(
            f"Required movie database '{movies_path}' is missing."
        )
    if not os.path.exists(vectors_path):
        raise FileNotFoundError(
            f"Required sparse vectors file '{vectors_path}' is missing."
        )
    try:
        # SECURITY WARNING: Pickle files should only be loaded from trusted sources!
        movies_df = pd.read_pickle(movies_path)
    except Exception as exc:
        raise IOError(
            f"Failed to load '{movies_path}'."
        ) from exc
    try:
        vectors = scipy.sparse.load_npz(vectors_path)
    except Exception as exc:
        raise IOError(
            f"Failed to load '{vectors_path}'."
        ) from exc
    if len(movies_df) != vectors.shape[0]:
        raise ValueError(
            f"Shape mismatch: movies database has "
            f"{len(movies_df)} rows, but the vector matrix has "
            f"{vectors.shape[0]} rows."
        )
    return movies_df, vectors

# Attempt loading the files and display error if files are missing/damaged
movies = None
vectors = None
data_loaded = False

try:
    movies, vectors = load_data()
    data_loaded = True
except Exception as e:
    st.error(f"⚠️ **Data Initialization Error:** {str(e)}")
    st.info("Please verify that all project data files (`movies.pkl` and `vectors.npz`) are present in the repository root directory.")

# Cached, deduplicated, sorted list of movie titles
@st.cache_data
def get_all_titles(movie_titles):
    return sorted(set(movie_titles))

if data_loaded:
    TITLES = get_all_titles(tuple(movies["title"].dropna().tolist()))

# Live-search callback for streamlit-searchbox
def search_titles(search_term):
    if not search_term:
        return []
    query = search_term.strip().lower()
    if not query:
        return []
    starts_with = [
        title for title in TITLES
        if title.lower().startswith(query)
    ]
    contains = [
        title for title in TITLES
        if query in title.lower()
        and not title.lower().startswith(query)
    ]
    return (starts_with + contains)[:50]

# Caching searches to TMDB API to save rate limits and speed up responses
@st.cache_data(show_spinner=False)
def _search_tmdb(title, release_year=None):
    if not TMDB_API_KEY:
        return None
        
    params = {
        "api_key": TMDB_API_KEY,
        "query": title,
    }
    
    # Send both title and year to TMDB to optimize matching if year exists
    if release_year:
        try:
            year_val = int(release_year)
            if 1800 < year_val < 2100:
                params["year"] = year_val
        except (ValueError, TypeError):
            pass
            
    # Retry logic up to 2 times
    for attempt in range(2):
        try:
            r = requests.get(
                "https://api.themoviedb.org/3/search/movie",
                params=params,
                timeout=8,
            )
            r.raise_for_status()
            data = r.json()
            results = data.get("results", [])
            return results[0] if results else None
        except requests.exceptions.RequestException:
            # Silence specific network details; don't log or print secrets/keys in logs
            if attempt == 0:
                continue
            return None
        except ValueError:
            # Handle JSON decoding errors
            if attempt == 0:
                continue
            return None
    return None

def fetch_poster(title, release_year=None):
    hit = _search_tmdb(title, release_year)
    if hit and hit.get("poster_path"):
        return "https://image.tmdb.org/t/p/w500" + hit["poster_path"]
    return None

def fetch_backdrop(title, release_year=None):
    hit = _search_tmdb(title, release_year)
    if hit and hit.get("backdrop_path"):
        return "https://image.tmdb.org/t/p/original" + hit["backdrop_path"]
    return None

def get_movie_year(title):
    """Retrieve the release year for a given title from the dataset to pass to TMDB."""
    if movies is not None:
        matches = movies[movies["title"] == title]
        if not matches.empty:
            if "release_year" in matches.columns:
                val = matches["release_year"].iloc[0]
                if pd.notna(val) and val > 0:
                    return int(val)
    return None

def recommend(title):
    if movies is None or vectors is None:
        st.error("❌ Recommendation system is unavailable due to data loading issues.")
        return pd.DataFrame()
        
    # 1. Handle movie title not found
    matching_movies = movies[movies["title"] == title]
    if matching_movies.empty:
        st.error(f"❌ The movie title '{title}' was not found in the dataset.")
        return pd.DataFrame()
        
    # 2. Handle duplicate movie titles
    # NOTE: If duplicate titles exist, we select the first matched index in the dataset.
    idx = matching_movies.index[0]
    
    # 3. Handle invalid vector index check
    if idx < 0 or idx >= vectors.shape[0]:
        st.error("❌ Database Error: Movie index is out of bounds for the vectors matrix.")
        return pd.DataFrame()
        
    # 4. Calculate cosine similarity
    try:
        sims = cosine_similarity(vectors[idx], vectors).flatten()
    except Exception as e:
        st.error(f"❌ Failed to compute similarity matrix: {str(e)}")
        return pd.DataFrame()
        
    # 5. Find the closest 20 movies (excluding the searched movie itself)
    order = [i for i in sims.argsort()[::-1] if i != idx][:20]
    if not order:
        st.error("❌ No recommendations match the selected movie.")
        return pd.DataFrame()
        
    pool = movies.iloc[order].copy()
    pool["similarity"] = sims[order]
    
    # Normalize ratings to a 0.0 - 1.0 range safely (ratings in dataset are 0-10)
    normalized_ratings = []
    for rating_val in pool["rating"]:
        try:
            if pd.isna(rating_val) or rating_val is None:
                norm = 0.0
            else:
                norm = max(0.0, min(1.0, float(rating_val) / 10.0))
        except (ValueError, TypeError):
            norm = 0.0
        normalized_ratings.append(norm)
    pool["normalized_rating"] = normalized_ratings
    
    # Blended score used ONLY for ranking:
    # 90% cosine similarity + 10% normalized rating nudge to push better-rated movies higher.
    pool["score"] = pool["similarity"] * 0.9 + pool["normalized_rating"] * 0.1
    
    # Sort and return the top 5
    recommendations = pool.sort_values("score", ascending=False).head(5)
    
    if recommendations.empty:
        st.error("❌ Recommendation results are empty.")
        return pd.DataFrame()
        
    return recommendations

# Render UI branding
st.markdown(
    '<div class="hero-row">'
    '<span class="hero-icon">🍿</span>'
    '<span class="hero-title">UpNext</span>'
    '</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-sub">Your next watch, matched by theme and ranked by rating.</div>',
    unsafe_allow_html=True,
)

# Friendly API key validation warning (does not crash the app)
if data_loaded and not TMDB_API_KEY:
    st.warning("⚠️ **TMDB API key is missing:** The recommender works, but movie posters and backdrop banners cannot be fetched. Please configure a `TMDB_API_KEY` in Streamlit secrets or environment variables.")

# Only render search box if database loaded successfully
if data_loaded:
    choice = st_searchbox(
        search_titles,
        placeholder="Search a movie...",
        key="movie_search",
    )
    go = st.button("✨ Recommend", type="primary")
    
    if go and choice:
        recs = recommend(choice)
        
        if not recs.empty:
            choice_year = get_movie_year(choice)
            
            # Fetch backdrop and posters in parallel
            recommendation_rows = list(recs.iterrows())
            
            def fetch_recommendation_poster(row):
                release_year = row.get("release_year")
                return fetch_poster(row["title"], release_year)
                
            with ThreadPoolExecutor(max_workers=6) as executor:
                backdrop_future = executor.submit(
                    fetch_backdrop,
                    choice,
                    choice_year,
                )
                poster_futures = [
                    executor.submit(fetch_recommendation_poster, row)
                    for _, row in recommendation_rows
                ]
                backdrop = backdrop_future.result()
                posters = [future.result() for future in poster_futures]
            
            valid_backdrop = validate_image_url(backdrop)
            bg_css = f"background-image:url('{valid_backdrop}');" if valid_backdrop else ""
            escaped_choice = safe_html_text(choice)
            
            st.markdown(
                f'<div class="banner" style="{bg_css}">'
                f'<div class="banner-overlay">'
                f'<span class="eyebrow">BECAUSE YOU LIKED</span>'
                f'<div class="banner-title">{escaped_choice}</div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
            
            st.markdown('<div class="section">You might also like</div>', unsafe_allow_html=True)
            
            medals = {1: "🥇", 2: "🥈", 3: "🥉"}
            cards = ""
            top_score = float(recs["score"].iloc[0])
            
            for rank, ((_, row), poster_url) in enumerate(
                zip(recommendation_rows, posters),
                start=1,
            ):
                # Calculate match percentage relative to the highest recommendation score
                if top_score > 0:
                    match_percentage = int((float(row["score"]) / top_score) * 100)
                else:
                    match_percentage = 0
                match_percentage = max(0, min(100, match_percentage))
                
                rank_label = medals.get(rank, f"#{rank}")
                escaped_rank_label = safe_html_text(rank_label)
                
                # Format and escape rating
                rating_str = f"{row['rating']:.1f}" if pd.notna(row.get('rating')) else "N/A"
                escaped_rating = safe_html_text(f"★ {rating_str}")
                
                escaped_title = safe_html_text(row["title"])
                escaped_similarity = safe_html_text(f"{match_percentage}% match")
                
                valid_poster = validate_image_url(poster_url)
                
                poster_html = (
                    f'<div class="poster"><img src="{valid_poster}"></div>'
                    if valid_poster else '<div class="poster-ph">🎬 </div>'
                )
                
                cards += (
                    f'<div class="card{" hl" if rank == 1 else ""}">'
                    f'{poster_html}'
                    f'<div class="card-head"><span class="rank">{escaped_rank_label}</span>'
                    f'<span class="badge">{escaped_rating}</span></div>'
                    f'<div class="m-title">{escaped_title}</div>'
                    f'<div class="bar"><div class="fill" style="width:{match_percentage}%"></div></div>'
                    f'<span class="pct">{escaped_similarity}</span>'
                    f'</div>'
                )
                
            st.markdown(f'<div class="grid">{cards}</div>', unsafe_allow_html=True)

# Render credit footer
if data_loaded:
    total_titles_str = safe_html_text(f"{len(movies):,}")
    st.markdown(
        f'<p class="footer">🍿 Powered by the Netflix catalog through 2025 · '
        f'{total_titles_str} titles · Posters via TMDB</p>',
        unsafe_allow_html=True,
    )
