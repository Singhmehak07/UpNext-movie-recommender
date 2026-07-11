import os
import pandas as pd

def migrate():
    print("Starting data migration...")
    
    # Paths
    pkl_src = os.path.join("UpNext", "movies.pkl")
    movies_csv = "netflix_movies_detailed_up_to_2025.csv"
    tv_csv = "netflix_tv_shows_detailed_up_to_2025.csv"
    
    pkl_dst = "movies.pkl"
    parquet_dst = "movies.parquet"
    
    # 1. Load movies.pkl
    if not os.path.exists(pkl_src):
        raise FileNotFoundError(f"Source file {pkl_src} not found.")
    df_pkl = pd.read_pickle(pkl_src)
    print(f"Loaded source pickle with {len(df_pkl)} rows.")
    print("Pickle columns:", df_pkl.columns.tolist())
    
    # 2. Load release years from detailed CSVs
    if not os.path.exists(movies_csv) or not os.path.exists(tv_csv):
        raise FileNotFoundError("Detailed CSV files are missing from the project root.")
        
    df_movies = pd.read_csv(movies_csv)[['show_id', 'release_year']]
    df_tv = pd.read_csv(tv_csv)[['show_id', 'release_year']]
    
    # Combine movie and TV show release years
    df_years = pd.concat([df_movies, df_tv]).drop_duplicates('show_id')
    print(f"Loaded {len(df_years)} release year entries from CSVs.")
    
    # 3. Merge release year into df_pkl
    df_merged = df_pkl.merge(df_years, on='show_id', how='left')
    
    # Verify merge count
    if len(df_merged) != len(df_pkl):
        raise ValueError(f"Merge error: Row count changed from {len(df_pkl)} to {len(df_merged)}")
        
    missing_years = df_merged['release_year'].isna().sum()
    print(f"Merged successfully. Missing years: {missing_years}")
    
    # Clean release_year: cast to nullable int or default
    df_merged['release_year'] = df_merged['release_year'].fillna(0).astype(int)
    
    # 4. Save to target paths
    df_merged.to_pickle(pkl_dst)
    print(f"Saved updated pickle to {pkl_dst}")
    
    df_merged.to_parquet(parquet_dst, index=False)
    print(f"Saved parquet to {parquet_dst}")
    
    # Verify parquet load
    df_test = pd.read_parquet(parquet_dst)
    print(f"Verified parquet load: {len(df_test)} rows.")
    print("New columns:", df_test.columns.tolist())
    print("Migration completed successfully!")

if __name__ == "__main__":
    migrate()
