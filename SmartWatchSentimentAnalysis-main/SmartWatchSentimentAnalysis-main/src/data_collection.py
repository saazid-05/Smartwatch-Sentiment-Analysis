"""
Data collection module for smartwatch reviews
Uses Amazon reviews dataset or web scraping
"""
import pandas as pd
import requests
from typing import List, Dict
import json


def load_amazon_reviews(file_path: str = None) -> pd.DataFrame:
    """
    Load Amazon smartwatch reviews from file or dataset
    """
    if file_path:
        df = pd.read_csv(file_path)
    else:
        # Using sample data structure for demonstration
        # In production, use actual Amazon review dataset
        df = pd.DataFrame({
            'review_text': [],
            'rating': [],
            'product_name': [],
            'date': []
        })
    
    return df


def create_sentiment_labels(df: pd.DataFrame, rating_column: str = 'rating') -> pd.DataFrame:
    """
    Convert ratings to sentiment labels
    1-2 stars: Negative (0)
    3 stars: Neutral (1)
    4-5 stars: Positive (2)
    """
    df['sentiment'] = df[rating_column].apply(
        lambda x: 0 if x <= 2 else (1 if x == 3 else 2)
    )
    return df


def filter_smartwatch_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter reviews to only include smartwatch-related products
    """
    keywords = ['smartwatch', 'smart watch', 'fitness tracker', 'wearable', 
                'apple watch', 'galaxy watch', 'fitbit', 'garmin']
    
    mask = df['product_name'].str.lower().str.contains('|'.join(keywords), na=False)
    return df[mask]


def load_and_prepare_data(file_path: str = None) -> pd.DataFrame:
    """
    Main function to load and prepare dataset
    """
    df = load_amazon_reviews(file_path)
    
    if not df.empty:
        df = filter_smartwatch_reviews(df)
        df = create_sentiment_labels(df)
        df = df.dropna(subset=['review_text', 'sentiment'])
    
    return df


if __name__ == "__main__":
    # Example usage
    df = load_and_prepare_data("data/amazon_reviews.csv")
    print(f"Loaded {len(df)} reviews")
    print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
