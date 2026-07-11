<p align="center">
  <img src="assets/upnext-banner.svg" alt="UpNext — find your next watch" width="100%" />
</p>

<p align="center">
  <strong>A content-based movie recommender that matches titles by theme and ranks them with a small quality boost.</strong>
</p>

<p align="center">
  <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit" /></a>
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.9+" />
  <img src="https://img.shields.io/badge/Model-TF--IDF-E50914?style=flat-square" alt="TF-IDF model" />
  <img src="https://img.shields.io/badge/Catalog-through%202025-181818?style=flat-square" alt="Catalog through 2025" />
</p>

<p align="center">
  <a href="https://nextwa.streamlit.app/"><strong>Launch the live app →</strong></a>
  ·
  <a href="#-how-it-works">How it works</a>
  ·
  <a href="#-run-locally">Run locally</a>
</p>


---

## Demo

<p align="center">
  <img src="demo.png" alt="UpNext application demo" width="100%" />
</p>


## About UpNext

UpNext is a content-based recommendation system built around a processed Netflix catalog through 2025. Instead of relying on viewing history or user profiles, it compares the metadata of each title—including genres, cast, director, and plot description—to find movies with similar themes.

After retrieving the closest matches, UpNext applies a small rating-based boost and presents the top five recommendations in a cinematic Streamlit interface with live TMDB artwork.

## Features

- **Theme-based recommendations** — matches titles using genres, cast, director, and plot information.
- **Quality-aware ranking** — combines 90% content similarity with a 10% normalized rating boost.
- **Fast single-bar search** — shows a small set of matching titles as the user types instead of loading the full catalog into the browser.
- **Live movie artwork** — retrieves posters and backdrop images from TMDB.
- **Parallel image fetching** — requests recommendation artwork concurrently to reduce waiting time.
- **Ranked top-five results** — displays ratings, medals, and relative match bars in a responsive card layout.
- **Graceful fallbacks** — uses a clean placeholder when TMDB artwork is unavailable.
- **Secure configuration** — reads the TMDB key from Streamlit secrets rather than storing it in the repository.

## How it works

```text
Raw catalog
    ↓
Clean and normalize metadata
    ↓
Combine genre, cast, director, and description into tags
    ↓
Convert tags into TF-IDF vectors
    ↓
Calculate cosine similarity against the selected movie
    ↓
Keep the 20 closest candidates
    ↓
Apply a small rating boost
    ↓
Return the top 5 recommendations
```

### Ranking formula

```text
final_score = (cosine_similarity × 0.90) + (normalized_rating × 0.10)
```

The model is intentionally weighted toward relevance: similarity contributes **90%** of the final score, while rating contributes **10%** as a small quality signal.

The match bars shown in the interface are relative to the highest-ranked recommendation: the first result is displayed as 100%, and the remaining results are scaled against its final score. They represent metadata alignment—not a guaranteed enjoyment probability.

## Tech stack

| Purpose | Technology |
|---|---|
| App and interface | [Streamlit](https://streamlit.io/) |
| Live search component | `streamlit-searchbox` |
| Data processing | [pandas](https://pandas.pydata.org/) |
| TF-IDF and cosine similarity | [scikit-learn](https://scikit-learn.org/) |
| Sparse matrix storage | [SciPy](https://scipy.org/) |
| API requests | [Requests](https://requests.readthedocs.io/) |
| Posters and backdrops | [TMDB API](https://www.themoviedb.org/) |

## Repository structure

```text
NextWatch-movie-recommender/
├── assets/
│   └── upnext-banner.svg   # animated README banner
├── app.py                  # Streamlit application
├── movies.pkl              # cleaned movie metadata
├── vectors.npz             # precomputed sparse TF-IDF matrix
├── requirements.txt        # Python dependencies
├── README.md               # project documentation
├── demo.png                # app screenshot you will add
└── .gitignore              # files excluded from Git
```

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/Singhmehak07/NextWatch-movie-recommender.git
cd NextWatch-movie-recommender
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your TMDB API key

Create `.streamlit/secrets.toml` locally:

```toml
TMDB_API_KEY = "your_tmdb_v3_api_key"
```

The secrets file must remain excluded by `.gitignore`. Never commit a real API key.

### 5. Start the app

```bash
streamlit run app.py
```

Streamlit will normally open the app at `http://localhost:8501`.

## Deploy on Streamlit Community Cloud

1. Push the project to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create an app from `Singhmehak07/NextWatch-movie-recommender`.
4. Set the main file path to `app.py`.
5. Open **App settings → Secrets** and add:

```toml
TMDB_API_KEY = "your_tmdb_v3_api_key"
```

6. Deploy, then replace the placeholder live-demo URL at the top of this README.

## Dataset

The recommendation data comes from a processed Netflix catalog covering titles through 2025. During preprocessing, descriptive fields were cleaned and combined into a single tag representation used by TF-IDF.

The deployed app loads precomputed artifacts (`movies.pkl` and `vectors.npz`) so it does not need to rebuild the vectorizer whenever it starts.

## Known limitations

- Recommendation quality depends on the richness and accuracy of the available metadata.
- The system is content-based and does not learn from individual viewing behavior.
- Duplicate or highly ambiguous titles may resolve to the first matching dataset row.
- TMDB title searches can occasionally return incorrect artwork for short or ambiguous movie names.
- Match percentages describe relative metadata similarity, not the probability that someone will enjoy a title.
- Titles released after the dataset's 2025 cutoff are not included.

## Future improvements

- Add year-aware TMDB matching for more accurate posters.
- Introduce fuzzy title search for misspellings.
- Expand and evaluate the TV-show catalog.
- Add offline relevance metrics and user feedback evaluation.
- Support personalized recommendations using interaction data.

## Credits

Movie posters and backdrop images are provided by [The Movie Database (TMDB)](https://www.themoviedb.org/).

> This product uses the TMDB API but is not endorsed or certified by TMDB.

## Author

**Mehakpreet Singh**  
Building projects while learning machine learning.
---

<p align="center">
  If you found this project useful, consider giving the repository a star.
</p>
