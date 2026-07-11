# 🍿 UpNext

> Your next watch, matched by theme and ranked by rating.

UpNext is a content-based movie recommendation system that recommends movies using their content and metadata, not user viewing history. By analyzing the thematic descriptors of a movie (genre, cast, director, and plot details), it identifies the closest thematic matches in the Netflix catalog and applies a small rating-based ranking boost to present the top five recommendations in a clean, cinematic interface.

## 🔗 Live Demo

[Open the UpNext app](<YOUR_STREAMLIT_APP_URL>)

<!-- ![UpNext demo](assets/upnext-demo.png) (Place a screenshot or GIF of the app in the assets folder once captured) -->

---

## 🎨 Features

- 🎯 **Theme-based Recommendations**: Recommends movies by matching underlying themes, genres, cast, and descriptions.
- ⭐ **Quality-Aware Ranking**: Incorporates a small rating-based nudge to break ties and ensure high-quality movies are ranked higher.
- 🖼️ **Live Poster & Backdrop Integration**: Dynamically fetches backdrop banners and movie posters from TMDB.
- 🥇 **Ranked Results View**: Display of top five recommendations styled with cinematic dark cards, movie badges, and similarity bars.
- 🎬 **Graceful Fallbacks**: Displays clean, styled default placeholders if movie posters or backdrop images are not available.
- 🔒 **Secure Keys & Exception Safety**: Gracefully handles missing TMDB API keys, request timeouts, and invalid data formats without crashing.

---

## 🧠 How the Recommendation Pipeline Works

UpNext uses a natural language processing pipeline to calculate movie-to-movie similarity:

```
Clean data
→ Combine genre, cast, director, and description into tags
→ Convert tags into TF-IDF vectors (using TfidfVectorizer)
→ Calculate cosine similarity
→ Select the closest movies
→ Apply a small rating boost for ranking
→ Display the top five results
```

### Ranking Formula
To produce the final recommended list, we retrieve the 20 most similar items via cosine similarity and re-rank them using a combined score:
$$\text{Score} = (\text{Cosine Similarity} \times 0.9) + (\text{Normalized Rating} \times 0.1)$$
* The **Similarity Score** displayed to users represents the raw cosine similarity percentage:
  $$\text{Similarity Score} = \max(0, \min(100, \text{round}(\text{Cosine Similarity} \times 100)))$$
* The rating boost uses a normalized rating (scaled to a $0.0 - 1.0$ range from TMDB's $0 - 10$ scale) to breaks ties and favor well-received movies.

---

## 🛠️ Technology Stack

| Purpose | Technology |
| :--- | :--- |
| **App / UI** | [Streamlit](https://streamlit.io/) |
| **Data Wrangling** | [Pandas](https://pandas.pydata.org/) |
| **Vectorization / Similarity** | [Scikit-learn](https://scikit-learn.org/) (TF-IDF & Cosine Similarity) |
| **Sparse Matrix Storage** | [SciPy](https://scipy.org/) |
| **Data Serialization** | [PyArrow](https://pyarrow.apache.org/) (Parquet format) & Pickle |
| **External Poster Service** | [TMDB API](https://www.themoviedb.org/) |

---

## 📁 Repository Structure

The project has a flat structure at the root of the repository:

```
NextWatch-movie-recommender/
├── app.py                      # Streamlit application
├── movies.pkl                  # Movie metadata database (Pickle fallback)
├── movies.parquet              # Movie metadata database (Safer Parquet format)
├── vectors.npz                 # Precalculated TF-IDF sparse matrix
├── requirements.txt            # Project dependencies list
├── README.md                   # Project documentation
├── .gitignore                  # Git file exclude list
└── .streamlit/
    └── secrets.toml.example    # Configuration placeholder for TMDB API key
```

---

## 🚀 Local Installation

### Prerequisites
* Python **3.9 to 3.11** is recommended for compatibility with all dependencies.

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Singhmehak07/NextWatch-movie-recommender.git
   cd NextWatch-movie-recommender
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the TMDB API Key:**
   Before running the app, create a `.streamlit/secrets.toml` file at the root containing your API key:
   ```toml
   TMDB_API_KEY = "your_real_tmdb_api_key"
   ```

4. **Launch the application:**
   ```bash
   streamlit run app.py
   ```
   Open `http://localhost:8501` in your browser.

---

## 🔑 TMDB API Configuration

### How to Create a TMDB API Key
1. Sign up or log into [The Movie Database (TMDB)](https://www.themoviedb.org/).
2. Navigate to your Account Settings page and choose **API** from the sidebar.
3. Click **Create** under the API Key section and select **Developer**.
4. Fill in the required details and accept the terms of use to generate your V3 API Key.

### Deployed Streamlit Cloud Configuration
When deploying to Streamlit Community Cloud:
1. Go to your app's dashboard.
2. Select **Settings** > **Secrets**.
3. Add your key exactly as follows:
   ```toml
   TMDB_API_KEY = "your_real_tmdb_api_key"
   ```

---

## 📊 Dataset Information

The app is built on a processed Netflix catalog covering titles available up to **2025**. During preprocessing, genre descriptors, cast lists, director names, and textual plot descriptions were merged into a single `tags` field to enable thematic comparisons. Ratings are derived from standard audience votes and normalized to a $0 - 10$ scale.

---

## ⚠️ Known Limitations

- **Metadata Dependence**: Recommendations are highly dependent on the quality and richness of descriptions and tags in the dataset.
- **No User Learning**: The system is completely content-based and does not learn from user viewing histories or personal profiles.
- **Title Ambiguity**: If duplicate movie titles exist in the dataset, the recommender currently matches the first occurrence.
- **Poster Inaccuracies**: TMDB API poster searches are matched by title and year. If metadata is missing or off, an incorrect poster might occasionally be selected.
- **Similarity Score Interpretation**: The similarity score indicates metadata alignment, not a guaranteed enjoyment rate.
- **Dataset Horizon**: The local catalog contains movies and TV shows released up to **2025**. Newer releases will not appear in results.
- **Minor Rating Impact**: Ratings are used strictly as a minor $10\%$ tie-breaker during final ranking.

---

## 🔒 Security Notes

### Exposed API Key Revocation
> [!WARNING]
> The TMDB API key that was previously hardcoded in the source code (`f31b26dcf87e179fce79c0790da2ebd2`) has been leaked publicly. **You must immediately log into your TMDB account and delete/revoke this key manually.**

### Purging Exposed Secrets from Git History
If you pushed a commit containing your real TMDB API key to GitHub, you should purge it from your history to prevent abuse:
1. Use `git-filter-repo` (recommended) or `BFG Repo-Cleaner` to remove the secret:
   ```bash
   git filter-repo --invert-paths --path-match-filters <pattern>
   ```
2. Alternatively, force-push the cleaned commits back to your remote:
   ```bash
   git push origin --force --all
   ```
   * **Warning**: Force pushing rewrites the git timeline. Other collaborators will need to re-clone the repository.

---

## 🙏 Credits

- Movie posters and background banners are provided by [The Movie Database (TMDB)](https://www.themoviedb.org/). This product uses the TMDB API but is not officially endorsed or certified by TMDB.

---

## 👤 Author

**Mehakpreet Singh** — part of an ongoing machine-learning learning journey.