import re
import nltk
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizerFast


class TextPreprocessor:
    def __init__(self):
        """Initializes NLTK tools and stopwords."""
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        """Standardizes text by removing special characters, stopwords, and lemmatizing."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = text.split()
        cleaned = [self.lemmatizer.lemmatize(w) for w in tokens if w not in self.stop_words]
        return " ".join(cleaned)

    def label_sentiment(self, rating):
        """Maps numerical ratings to sentiment categories."""
        if rating >= 4:
            return 'Positive'
        elif rating == 3:
            return 'Neutral'
        else:
            return 'Negative'

    def generate_iob_tags(self, row):
        """Generates IOB tags for NER tasks based on brand and category matches."""
        tokens = str(row['cleaned_review']).split()
        brand_tokens = str(row['brand']).lower().split()
        cat_tokens = str(row['categories']).lower().replace(',', ' ').split()

        tags = ["O"] * len(tokens)
        for i in range(len(tokens)):
            # Brand Match
            if tokens[i:i+len(brand_tokens)] == brand_tokens:
                tags[i] = "B-BRAND"
                for j in range(1, len(brand_tokens)):
                    if i + j < len(tags):
                        tags[i+j] = "I-BRAND"
            # Category Match
            elif tokens[i] in cat_tokens:
                tags[i] = "B-CAT"
        return tags

    def process_dataframe(self, file_path):
        """Loads and executes the full cleaning pipeline on the dataframe."""
        df = pd.read_csv(file_path, engine='python')
        
        # Handling Missing Values
        df = df.dropna(subset=['reviews.text'])
        df['reviews.title'] = df['reviews.title'].fillna('No Title')
        df['reviews.username'] = df['reviews.username'].fillna('Anonymous')
        df['reviews.didPurchase'] = df['reviews.didPurchase'].fillna('Unknown')
        
        if not df['reviews.doRecommend'].empty:
            mode_val = df['reviews.doRecommend'].mode()[0]
            df['reviews.doRecommend'] = df['reviews.doRecommend'].fillna(mode_val)

        # Feature Engineering
        df['full_review'] = df['reviews.title'] + " " + df['reviews.text']
        df['cleaned_review'] = df['full_review'].apply(self.clean_text)
        df['sentiment'] = df['reviews.rating'].apply(self.label_sentiment)
        df['iob_tags'] = df.apply(self.generate_iob_tags, axis=1)

        # Drop unnecessary columns
        cols_to_drop = [
            'reviews.userCity', 'reviews.userProvince', 'ean', 'reviews.title', 
            'reviews.text', 'dateAdded', 'dateUpdated', 'keys', 'upc', 
            'manufacturerNumber', 'reviews.sourceURLs', 'full_review'
        ]
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        
        return df

# This prepares the data specifically for the BERT model
class ReviewDataset(Dataset):
    def __init__(self, texts, tags, tokenizer, max_len, label2id):
        self.texts = texts
        self.tags = tags
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = label2id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        tags = self.tags[item]

        encoding = self.tokenizer(
            text.split(),
            is_split_into_words=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Align labels with tokens
        labels = [self.label2id[t] for t in tags]
        labels = labels[:(self.max_len)] # Simple truncation
        labels += [self.label2id["O"]] * (self.max_len - len(labels))

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }