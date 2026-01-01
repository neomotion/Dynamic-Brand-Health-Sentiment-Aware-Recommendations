import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LinearRegression
from transformers import BertTokenizerFast, BertForTokenClassification, BertForSequenceClassification

class SentimentReviewDataset(Dataset):
    """Dataset class for batch sentiment inference."""
    def __init__(self, reviews, tokenizer, max_len=128):
        self.reviews = reviews
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

"""
This class encapsulates the NER (Named Entity Recognition) logic. It uses fine-tuned BERT model to find brands 
within the text. By placing it in a class, you can initialize the heavy model once and reuse the extract_entities method 
across different data subsets.  
"""
class EntityExtractor:
    """Handles NER to extract brand and category mentions using BERT."""
    def __init__(self, model_path="./bert_ner_model", device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        self.model = BertForTokenClassification.from_pretrained(model_path).to(self.device)
        # Assuming id2label is standard for your NER model
        self.id2label = self.model.config.id2label 

    def extract_entities(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=2)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        entities = []
        for token, pred in zip(tokens, predictions[0]):
            label = self.id2label[pred.item()]
            if label != "O" and not token.startswith("##"):
                entities.append(token)
        return list(set(entities))

""" 
This handles the transition from raw text to a numerical "Sentiment Weight." It uses a weighted average of the BERT output
probabilities. Instead of just picking the highest probability (e.g., "4 stars"), it calculates a nuanced score
(e.g., "3.7 stars") which provides a smoother input for the health formula.
"""
class SentimentAnalyzer:
    """Handles high-volume sentiment scoring using a 5-star BERT model."""
    def __init__(self, model_name="nlptown/bert-base-multilingual-uncased-sentiment", device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name).to(self.device)

    def run_batch_sentiment(self, df, batch_size=32):
        self.model.eval()
        dataset = SentimentReviewDataset(df['cleaned_review'].values, self.tokenizer)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=2)

        weights = []
        # Weighted scale for 1-5 stars: [1*, 2*, 3*, 4*, 5*]
        scale = torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0]).to(self.device)

        with torch.no_grad():
            for batch in tqdm(loader, desc="Sentiment Analysis"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = F.softmax(outputs.logits, dim=1)
                
                batch_weights = torch.matmul(probs, scale)
                weights.extend(batch_weights.cpu().tolist())

        df['sentiment_weight'] = weights
        return df

""" 
Brand Health Score: It implements a custom formula. This combines the unstructured NLP signal (50%) with structured user
data (30% rating, 20% recommendation) to create a robust KPI (Key Performance Indicator).

Trend Prediction: It uses LinearRegression to look at how a brand's health score changes over time (monthly/weekly).
A positive slope indicates an "Upward" trend, providing predictive insights for the "Brand Health" dashboard.
"""
class BrandHealthEngine:
    """Calculates Health Scores and predicts market trends using regression."""
    
    @staticmethod
    def calculate_brand_health(df):
        """
        Formula: (Sentiment * 0.5) + (Rating/5 * 0.3) + (Recommendation Rate * 0.2)
        """
        temp_df = df.copy()
        temp_df['reviews.doRecommend'] = pd.to_numeric(
            temp_df['reviews.doRecommend'], errors='coerce'
        ).fillna(0).astype(float)

        brand_stats = temp_df.groupby('brand').agg({
            'sentiment_weight': 'mean',
            'reviews.rating': 'mean',
            'reviews.doRecommend': 'mean'
        }).reset_index()

        brand_stats['health_score'] = (
            (brand_stats['sentiment_weight'] * 0.5) +
            ((brand_stats['reviews.rating'] / 5) * 0.3) +
            (brand_stats['reviews.doRecommend'] * 0.2)
        ) * 100

        return brand_stats.round(2).sort_values(by='health_score', ascending=False)

    def predict_trends(self, df, time_range='M'):
        """Analyzes periodic health scores to determine upward/downward trends."""
        df['reviews.date'] = pd.to_datetime(df['reviews.date'], format='mixed', errors='coerce', utc=True)
        df = df.dropna(subset=['reviews.date'])
        df['time_bucket'] = df['reviews.date'].dt.to_period(time_range)

        # Get health history per time bucket
        periodic_history = df.groupby('time_bucket').apply(self.calculate_brand_health).reset_index()
        trend_results = []

        for brand in periodic_history['brand'].unique():
            brand_data = periodic_history[periodic_history['brand'] == brand].sort_values('time_bucket')

            if len(brand_data) >= 2:
                X = np.array(range(len(brand_data))).reshape(-1, 1)
                y = brand_data['health_score'].values
                
                model = LinearRegression().fit(X, y)
                slope = model.coef_[0]

                status = "Upward" if slope > 0.5 else ("Downward" if slope < -0.5 else "Stable")

                trend_results.append({
                    'brand': brand,
                    'slope': round(slope, 4),
                    'trend_status': status,
                    'recent_health': y[-1]
                })

        return pd.DataFrame(trend_results)