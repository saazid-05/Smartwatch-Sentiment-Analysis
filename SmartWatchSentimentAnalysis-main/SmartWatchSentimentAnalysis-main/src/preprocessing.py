"""
Text preprocessing module
"""
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import pandas as pd

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


def clean_text(text: str, remove_stopwords: bool = False) -> str:
    """
    Clean and normalize text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text)
        text = ' '.join([word for word in tokens if word not in stop_words])
    
    return text


def preprocess_dataframe(df: pd.DataFrame, text_column: str = 'review_text') -> pd.DataFrame:
    """
    Preprocess entire dataframe
    """
    df = df.copy()
    df['cleaned_text'] = df[text_column].apply(lambda x: clean_text(str(x)))
    
    # Remove empty reviews
    df = df[df['cleaned_text'].str.len() > 10]
    
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1):
    """
    Split data into train, validation, and test sets
    """
    # First split: train+val and test
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=42, stratify=df['sentiment']
    )
    
    # Second split: train and val
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_ratio, random_state=42, stratify=train_val['sentiment']
    )
    
    return train, val, test


if __name__ == "__main__":
    # Example usage
    sample_text = "This smartwatch is AMAZING!!! Best purchase ever. https://example.com"
    cleaned = clean_text(sample_text)
    print(f"Original: {sample_text}")
    print(f"Cleaned: {cleaned}")
