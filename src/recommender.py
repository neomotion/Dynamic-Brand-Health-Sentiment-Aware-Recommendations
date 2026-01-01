import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split

class HybridRecommender:
    """
    A Hybrid Recommendation Engine combining SVD Collaborative Filtering 
    with Brand Health Scores for sentiment-aware product suggestions.
    """
    def __init__(self, brand_health_df):
        """
        Initializes the recommender with pre-calculated brand health metrics.
        
        Args:
            brand_health_df (pd.DataFrame): Dataframe containing 'brand' and 'health_score'.
        """
        self.model = SVD()
        self.brand_health_lookup = brand_health_df.set_index('brand')['health_score'].to_dict()
        self.reader = Reader(rating_scale=(1, 5))

    def train_model(self, df):
        """
        Trains the SVD model using the cleaned interactions.
        """
        # Prepare data for Surprise library
        data = Dataset.load_from_df(df[['reviews.username', 'name', 'reviews.rating']], self.reader)
        trainset, testset = train_test_split(data, test_size=0.2)
        
        # Train SVD
        self.model.fit(trainset)
        
        # Internal evaluation
        predictions = self.model.test(testset)
        rmse = accuracy.rmse(predictions, verbose=False)
        print(f"SVD Model Trained. Validation RMSE: {rmse:.4f}")
        return trainset

    def get_hybrid_recommendations(self, user_id, data, cf_model, brand_health, brand_trends, top_n=10):
        # list of all unique product IDs
        all_product_ids = data['id'].unique()

        # --- COLD START LOGIC (New User) ---
        # If user has never rated anything, recommend based on global health and trends
        if user_id not in data['reviews.username'].values:
            # Merge health and trends to find "Rising Stars" and Sort by health score and positive slope
            global_rank = brand_health.merge(brand_trends, on='brand')
            global_rank = global_rank.sort_values(by=['health_score', 'slope'], ascending=False)

            top_brands = global_rank['brand'].head(top_n).tolist()
            return data[data['brand'].isin(top_brands)][['name', 'brand']].drop_duplicates().head(top_n)

        # --- HYBRID LOGIC (Existing User) ---
        hybrid_predictions = []

        for product_id in all_product_ids:
            # Get CF Predicted Rating
            cf_pred = cf_model.predict(user_id, product_id).est

            # Lookup Brand Details
            product_info = data[data['id'] == product_id].iloc[0]
            brand_name = product_info['brand']

            # Get Health Score (Default to 50/100 if brand not found)
            health_row = brand_health[brand_health['brand'] == brand_name]
            health_score = health_row['health_score'].values[0] if not health_row.empty else 50

            # Get Trend Slope (Default to 0 if not found)
            trend_row = brand_trends[brand_trends['brand'] == brand_name]
            slope = trend_row['slope'].values[0] if not trend_row.empty else 0

            # CALCULATE HYBRID SCORE
            # Scaling everything to a 0-1 range
            final_score = (cf_pred / 5 * 0.8) + (health_score / 100 * 0.1) + (slope * 0.1)

            hybrid_predictions.append({
                'product_name': product_info['name'],
                'brand': brand_name,
                'hybrid_score': round(final_score, 4),
                'trend': trend_row['trend_status'].values[0] if not trend_row.empty else "Stable"
            })

        # Return Top N ranked by Hybrid Score
        hybrid_df = pd.DataFrame(hybrid_predictions).sort_values(by='hybrid_score', ascending=False)
        return hybrid_df.drop_duplicates(subset=['product_name']).head(top_n)


    def recommend_products(self, username, df, top_n=5):
        """
        Generates top-N recommendations for a specific user.
        """
        all_products = df[['name', 'brand']].drop_duplicates()
        user_viewed = df[df['reviews.username'] == username]['name'].unique()
        
        recommendations = []
        
        # Iterate through products user hasn't seen
        for _, row in all_products.iterrows():
            prod_name = row['name']
            brand_name = row['brand']
            
            if prod_name not in user_viewed:
                score = self.get_hybrid_score(username, prod_name, brand_name)
                recommendations.append({
                    'product': prod_name,
                    'brand': brand_name,
                    'hybrid_score': score
                })
        
        # Sort by hybrid score and return top N
        recom_df = pd.DataFrame(recommendations)
        if recom_df.empty:
            return pd.DataFrame(columns=['product', 'brand', 'hybrid_score'])
            
        return recom_df.sort_values(by='hybrid_score', ascending=False).head(top_n)

def prepare_recommendation_data(df):
    """
    Utility to filter and group data for the recommendation engine.
    """
    # Ensure usernames are strings and filter out anonymous if necessary
    df['reviews.username'] = df['reviews.username'].astype(str)
    
    # We focus on users with multiple reviews to improve SVD accuracy
    user_counts = df['reviews.username'].value_counts()
    active_users = user_counts[user_counts >= 1].index
    
    return df[df['reviews.username'].isin(active_users)]