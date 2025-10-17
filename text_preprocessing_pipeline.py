#!/usr/bin/env python3
"""
Text Preprocessing Pipeline for Sentiment Analysis

This module provides a comprehensive text preprocessing pipeline that transforms
raw text data into clean, standardized format suitable for machine learning models.

The pipeline performs the following operations:
1. Lowercase conversion
2. URL, mention, and hashtag symbol removal
3. Special character, number, and punctuation removal
4. Tokenization
5. Stopword removal
6. Lemmatization
7. Token rejoining
"""

import re
import string
import pandas as pd
from typing import List, Optional
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data (only needed once)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextPreprocessor:
    """
    A comprehensive text preprocessing class for sentiment analysis.
    """
    
    def __init__(self, custom_stopwords: Optional[List[str]] = None):
        """
        Initialize the text preprocessor.
        
        Args:
            custom_stopwords: Additional stopwords to remove beyond NLTK defaults
        """
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Add custom stopwords if provided
        if custom_stopwords:
            self.stop_words.update(custom_stopwords)
        
        # Common social media and sentiment-specific stopwords
        social_media_stopwords = {
            'rt', 'via', 'amp', 'quot', 'lt', 'gt', 'http', 'https', 'www',
            'com', 'org', 'net', 'edu', 'gov', 'co', 'uk', 'ca', 'au',
            'twitter', 'facebook', 'instagram', 'youtube', 'tiktok'
        }
        self.stop_words.update(social_media_stopwords)
    
    def remove_urls_mentions_hashtags(self, text: str) -> str:
        """
        Remove URLs, user mentions, and hashtag symbols while preserving hashtag words.
        
        Args:
            text: Input text string
            
        Returns:
            Text with URLs, mentions, and hashtag symbols removed
        """
        # Remove URLs (http, https, www)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove user mentions (@username)
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtag symbols but keep the word
        text = re.sub(r'#', '', text)
        
        return text
    
    def remove_special_characters_numbers(self, text: str) -> str:
        """
        Remove special characters, numbers, and punctuation, keeping only letters and spaces.
        
        Args:
            text: Input text string
            
        Returns:
            Text with only letters and spaces
        """
        # Keep only letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into individual words.
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens (words)
        """
        return word_tokenize(text.lower())
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from token list.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            List of tokens with stopwords removed
        """
        return [token for token in tokens if token not in self.stop_words and len(token) > 1]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens to their base form.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            List of lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess_text(self, text: str) -> str:
        """
        Complete text preprocessing pipeline.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned and preprocessed text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Step 1: Lowercase conversion
        text = text.lower()
        
        # Step 2: Remove URLs, mentions, and hashtag symbols
        text = self.remove_urls_mentions_hashtags(text)
        
        # Step 3: Remove special characters, numbers, and punctuation
        text = self.remove_special_characters_numbers(text)
        
        # Step 4: Tokenization
        tokens = self.tokenize_text(text)
        
        # Step 5: Stopword removal
        tokens = self.remove_stopwords(tokens)
        
        # Step 6: Lemmatization
        tokens = self.lemmatize_tokens(tokens)
        
        # Step 7: Join tokens back into string
        cleaned_text = ' '.join(tokens)
        
        return cleaned_text
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str = 'text', 
                           output_column: str = 'clean_text') -> pd.DataFrame:
        """
        Apply preprocessing to a pandas DataFrame.
        
        Args:
            df: Input DataFrame
            text_column: Name of the column containing text to preprocess
            output_column: Name of the column to store cleaned text
            
        Returns:
            DataFrame with additional cleaned text column
        """
        print(f"Preprocessing {len(df)} texts...")
        
        # Apply preprocessing to the text column
        df[output_column] = df[text_column].apply(self.preprocess_text)
        
        # Remove rows where cleaned text is empty
        initial_count = len(df)
        df = df[df[output_column].str.strip() != '']
        final_count = len(df)
        
        if initial_count != final_count:
            print(f"Removed {initial_count - final_count} rows with empty cleaned text")
        
        return df

def demonstrate_preprocessing():
    """
    Demonstrate the preprocessing pipeline with example texts.
    """
    print("=" * 80)
    print("TEXT PREPROCESSING PIPELINE DEMONSTRATION")
    print("=" * 80)
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Example texts showing various types of noise
    example_texts = [
        "So I spent a few hours making something for fun. . . If you don't know I am a HUGE @Borderlands fan and Maya is one of my favorite characters. So I decided to make myself a wallpaper for my PC. . Here is the original image versus the creation I made :) Enjoy! pic.twitter.com/mLsI5wf9Jg",
        "@VirginAmerica plus you've added commercials to the experience... tacky.",
        "RT @NatGeoChannel: Watch #BeforeTheFlood right here, as @LeoDiCaprio travels the world to tackle #climate change https://t.co/LkDehj3tNn",
        "TRENDING: New Yorkers encounter empty supermarket shelves (pictured, Wegmans in Brooklyn), sold-out online grocers (FoodKick, MaxDelivery) as #coronavirus-fearing shoppers stock up https://t.co/Gr76pcrLWh",
        "I love this movie! It's amazing!!! <unk> #awesome #bestmovieever"
    ]
    
    for i, text in enumerate(example_texts, 1):
        print(f"\nExample {i}:")
        print(f"Original: {text}")
        cleaned = preprocessor.preprocess_text(text)
        print(f"Cleaned:  {cleaned}")
        print("-" * 80)

def main():
    """
    Main function to demonstrate the preprocessing pipeline.
    """
    demonstrate_preprocessing()
    
    print("\n" + "=" * 80)
    print("PREPROCESSING PIPELINE READY FOR USE")
    print("=" * 80)
    print("To use this pipeline on your datasets:")
    print("1. Import: from text_preprocessing_pipeline import TextPreprocessor")
    print("2. Initialize: preprocessor = TextPreprocessor()")
    print("3. Process DataFrame: df = preprocessor.preprocess_dataframe(df)")
    print("4. Or process single text: cleaned = preprocessor.preprocess_text(text)")

if __name__ == "__main__":
    main()
