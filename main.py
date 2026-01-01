import numpy as np  # Must be imported before surprise
import pandas as pd
from surprise import SVD, Dataset, Reader
from src.recommender import HybridRecommender

def find_data_file(filename):
    """Checks for file in 'data/' subfolder or current directory."""
    # Option 1: Look in data/ folder
    path_with_folder = os.path.join('data', filename)
    if os.path.exists(path_with_folder):
        return path_with_folder
    
    # Option 2: Look in current directory
    if os.path.exists(filename):
        return filename
    
    # Raise error if not found
    raise FileNotFoundError(f"Could not find {filename} in 'data/' or root. Current path: {os.getcwd()}")

def main():
    print("--- Initializing Sentiment-Aware Recommendation System ---")
    
    try:
        # 1. LOAD DATASETS
        # 'data' provides the mapping for 'id', 'brand', and 'reviews.username'
        data = pd.read_csv('data/sentiment_score.csv')
        brand_health_lookup = pd.read_csv('data/brand_health_lookup.csv')
        brand_trends_df = pd.read_csv('data/brand_trend_monthly.csv')
        
    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: Could not find necessary files. {e}")
        return

    # 2. RUNTIME MODEL TRAINING
    print("Training SVD model at runtime...")
    # Clean data specifically for Collaborative Filtering
    clean_cf_data = data.dropna(subset=['reviews.username', 'id', 'reviews.rating']).copy()
    
    # Initialize Surprise Reader and Dataset
    reader = Reader(rating_scale=(1, 5))
    cf_data = Dataset.load_from_df(clean_cf_data[['reviews.username', 'id', 'reviews.rating']], reader)
    trainset = cf_data.build_full_trainset()

    # Create and fit the SVD model
    svd_model = SVD(n_factors=50, random_state=42)
    svd_model.fit(trainset)
    print("Model training complete.")

    # 3. CAPTURE USER INPUT
    user_query = input("\nEnter the Username for recommendations (e.g., Dorothy W): ").strip()
    if not user_query:
        user_query = 'Dorothy W'
        print(f"No input provided, defaulting to: {user_query}")

    # 4. GENERATE HYBRID RECOMMENDATIONS
    print(f"Generating top 10 recommendations for '{user_query}'...")
    recommender = HybridRecommender(brand_health_lookup)
    results = recommender.get_hybrid_recommendations(
        user_id=user_query, 
        data=data, 
        cf_model=svd_model, 
        brand_health=brand_health_lookup, 
        brand_trends=brand_trends_df, 
        top_n=10
    )

    # 5. DISPLAY RESULTS
    print("\n--- FINAL RECOMMENDATION RESULTS ---")
    if results is not None and not results.empty:
        print(results)
    else:
        print("No recommendations found.")

if __name__ == "__main__":
    main()