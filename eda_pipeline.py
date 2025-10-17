#!/usr/bin/env python3
"""
Comprehensive Exploratory Data Analysis (EDA) Pipeline for Sentiment Analysis Datasets

This pipeline performs detailed analysis of sentiment analysis datasets including:
- High-level statistics and distributions
- Text length analysis
- Word cloud generation
- N-gram analysis
- Comparative analysis across datasets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from pathlib import Path
import re
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SentimentEDA:
    """
    Comprehensive EDA class for sentiment analysis datasets.
    """
    
    def __init__(self, output_dir: str = "eda_results"):
        """
        Initialize the EDA pipeline.
        
        Args:
            output_dir: Directory to save all EDA results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for each dataset
        self.dataset_dirs = {
            'borderlands': self.output_dir / 'borderlands',
            'airline': self.output_dir / 'airline', 
            'climate': self.output_dir / 'climate',
            'corona': self.output_dir / 'corona',
            'combined': self.output_dir / 'combined'
        }
        
        for dir_path in self.dataset_dirs.values():
            dir_path.mkdir(exist_ok=True)
    
    def load_dataset(self, filepath: str) -> pd.DataFrame:
        """
        Load a dataset from CSV file.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            Loaded DataFrame
        """
        try:
            df = pd.read_csv(filepath)
            print(f"✅ Loaded {len(df):,} rows from {filepath}")
            return df
        except Exception as e:
            print(f"❌ Error loading {filepath}: {e}")
            return None
    
    def calculate_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate text length features.
        
        Args:
            df: DataFrame with 'clean_text' column
            
        Returns:
            DataFrame with additional word_count and char_count columns
        """
        df = df.copy()
        df['word_count'] = df['clean_text'].str.split().str.len()
        df['char_count'] = df['clean_text'].str.len()
        return df
    
    def generate_sentiment_distribution(self, df: pd.DataFrame, dataset_name: str):
        """
        Generate sentiment distribution visualization and statistics.
        
        Args:
            df: DataFrame with sentiment data
            dataset_name: Name of the dataset for file naming
        """
        output_dir = self.dataset_dirs[dataset_name]
        
        # Calculate distribution
        sentiment_counts = df['sentiment'].value_counts()
        sentiment_percentages = df['sentiment'].value_counts(normalize=True) * 100
        
        # Create bar chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(sentiment_counts.index, sentiment_counts.values, 
                      color=['#2E8B57', '#DC143C', '#4682B4'])  # Green, Red, Blue
        plt.title(f'Sentiment Distribution - {dataset_name.title()} Dataset', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Sentiment', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        
        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, sentiment_counts.values)):
            sentiment_label = sentiment_counts.index[i]
            percentage = sentiment_percentages[sentiment_label]
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + count*0.01,
                    f'{count:,}\n({percentage:.1f}%)',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'sentiment_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save statistics to text file
        with open(output_dir / 'summary_stats.txt', 'w') as f:
            f.write(f"SENTIMENT ANALYSIS - {dataset_name.upper()} DATASET\n")
            f.write("=" * 50 + "\n\n")
            f.write("SENTIMENT DISTRIBUTION:\n")
            f.write("-" * 25 + "\n")
            for sentiment in sentiment_counts.index:
                count = sentiment_counts[sentiment]
                percentage = sentiment_percentages[sentiment]
                f.write(f"{sentiment.upper()}: {count:,} ({percentage:.1f}%)\n")
            f.write(f"\nTOTAL SAMPLES: {len(df):,}\n")
        
        print(f"✅ Generated sentiment distribution for {dataset_name}")
    
    def generate_text_length_analysis(self, df: pd.DataFrame, dataset_name: str):
        """
        Generate text length analysis visualizations.
        
        Args:
            df: DataFrame with text features
            dataset_name: Name of the dataset
        """
        output_dir = self.dataset_dirs[dataset_name]
        
        # Word count histogram
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(df['word_count'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title(f'Word Count Distribution - {dataset_name.title()}', fontweight='bold')
        plt.xlabel('Number of Words')
        plt.ylabel('Frequency')
        plt.axvline(df['word_count'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["word_count"].mean():.1f}')
        plt.legend()
        
        # Character count histogram
        plt.subplot(1, 2, 2)
        plt.hist(df['char_count'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.title(f'Character Count Distribution - {dataset_name.title()}', fontweight='bold')
        plt.xlabel('Number of Characters')
        plt.ylabel('Frequency')
        plt.axvline(df['char_count'].mean(), color='red', linestyle='--',
                   label=f'Mean: {df["char_count"].mean():.1f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'word_count_histogram.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Word count by sentiment box plot
        plt.figure(figsize=(10, 6))
        df.boxplot(column='word_count', by='sentiment', ax=plt.gca())
        plt.title(f'Word Count by Sentiment - {dataset_name.title()}', fontweight='bold')
        plt.suptitle('')  # Remove default title
        plt.xlabel('Sentiment')
        plt.ylabel('Word Count')
        plt.tight_layout()
        plt.savefig(output_dir / 'word_count_by_sentiment.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate and save average word counts by sentiment
        avg_word_counts = df.groupby('sentiment')['word_count'].mean()
        
        with open(output_dir / 'summary_stats.txt', 'a') as f:
            f.write(f"\nTEXT LENGTH ANALYSIS:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Overall Average Word Count: {df['word_count'].mean():.1f}\n")
            f.write(f"Overall Average Character Count: {df['char_count'].mean():.1f}\n\n")
            f.write("Average Word Count by Sentiment:\n")
            for sentiment, avg_count in avg_word_counts.items():
                f.write(f"{sentiment.upper()}: {avg_count:.1f} words\n")
        
        print(f"✅ Generated text length analysis for {dataset_name}")
    
    def generate_word_clouds(self, df: pd.DataFrame, dataset_name: str):
        """
        Generate word clouds for each sentiment class.
        
        Args:
            df: DataFrame with sentiment and clean_text columns
            dataset_name: Name of the dataset
        """
        output_dir = self.dataset_dirs[dataset_name]
        
        sentiments = ['positive', 'negative', 'neutral']
        colors = ['Greens', 'Reds', 'Blues']
        
        for sentiment, color in zip(sentiments, colors):
            # Filter data for this sentiment
            sentiment_data = df[df['sentiment'] == sentiment]['clean_text']
            
            if len(sentiment_data) == 0:
                print(f"⚠️  No data found for {sentiment} sentiment in {dataset_name}")
                continue
            
            # Combine all text for this sentiment
            text = ' '.join(sentiment_data.astype(str))
            
            # Generate word cloud
            wordcloud = WordCloud(
                width=800, height=400,
                background_color='white',
                colormap=color,
                max_words=100,
                relative_scaling=0.5,
                random_state=42
            ).generate(text)
            
            # Create plot
            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Word Cloud - {sentiment.title()} Sentiment\n{dataset_name.title()} Dataset', 
                     fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_dir / f'wordcloud_{sentiment}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✅ Generated word clouds for {dataset_name}")
    
    def generate_ngram_analysis(self, df: pd.DataFrame, dataset_name: str, n: int = 2):
        """
        Generate n-gram analysis for each sentiment.
        
        Args:
            df: DataFrame with sentiment and clean_text columns
            dataset_name: Name of the dataset
            n: N-gram size (2 for bigrams, 3 for trigrams)
        """
        output_dir = self.dataset_dirs[dataset_name]
        
        sentiments = ['positive', 'negative', 'neutral']
        ngram_type = 'bigram' if n == 2 else 'trigram'
        
        for sentiment in sentiments:
            # Filter data for this sentiment
            sentiment_data = df[df['sentiment'] == sentiment]['clean_text']
            
            if len(sentiment_data) == 0:
                continue
            
            # Generate n-grams
            all_ngrams = []
            for text in sentiment_data:
                words = str(text).split()
                if len(words) >= n:
                    for i in range(len(words) - n + 1):
                        ngram = ' '.join(words[i:i+n])
                        all_ngrams.append(ngram)
            
            # Count n-grams
            ngram_counts = Counter(all_ngrams)
            top_ngrams = ngram_counts.most_common(20)
            
            if not top_ngrams:
                continue
            
            # Create bar chart
            ngrams, counts = zip(*top_ngrams)
            
            plt.figure(figsize=(12, 8))
            bars = plt.barh(range(len(ngrams)), counts, color='steelblue', alpha=0.7)
            plt.yticks(range(len(ngrams)), ngrams)
            plt.xlabel('Frequency')
            plt.title(f'Top 20 {ngram_type.title()}s - {sentiment.title()} Sentiment\n{dataset_name.title()} Dataset',
                     fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            
            # Add value labels
            for i, (bar, count) in enumerate(zip(bars, counts)):
                plt.text(bar.get_width() + count*0.01, bar.get_y() + bar.get_height()/2,
                        str(count), va='center', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'top_{ngram_type}s_{sentiment}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✅ Generated {ngram_type} analysis for {dataset_name}")
    
    def analyze_dataset(self, filepath: str, dataset_name: str):
        """
        Perform complete EDA analysis on a single dataset.
        
        Args:
            filepath: Path to the dataset CSV file
            dataset_name: Name of the dataset
        """
        print(f"\n{'='*60}")
        print(f"ANALYZING {dataset_name.upper()} DATASET")
        print(f"{'='*60}")
        
        # Load dataset
        df = self.load_dataset(filepath)
        if df is None:
            return
        
        # Calculate text features
        df = self.calculate_text_features(df)
        
        # Generate all analyses
        self.generate_sentiment_distribution(df, dataset_name)
        self.generate_text_length_analysis(df, dataset_name)
        self.generate_word_clouds(df, dataset_name)
        self.generate_ngram_analysis(df, dataset_name, n=2)  # Bigrams
        self.generate_ngram_analysis(df, dataset_name, n=3)  # Trigrams
        
        print(f"✅ Completed analysis for {dataset_name}")
        return df
    
    def perform_vocabulary_overlap_analysis(self, airline_df: pd.DataFrame, corona_df: pd.DataFrame):
        """
        Perform vocabulary overlap analysis between airline and corona datasets.
        
        Args:
            airline_df: Airline dataset DataFrame
            corona_df: Corona dataset DataFrame
        """
        output_dir = self.dataset_dirs['combined']
        
        # Get top 100 positive words from each dataset
        airline_positive = airline_df[airline_df['sentiment'] == 'positive']['clean_text']
        corona_positive = corona_df[corona_df['sentiment'] == 'positive']['clean_text']
        
        # Extract words
        airline_words = []
        for text in airline_positive:
            airline_words.extend(str(text).split())
        
        corona_words = []
        for text in corona_positive:
            corona_words.extend(str(text).split())
        
        # Count words
        airline_word_counts = Counter(airline_words)
        corona_word_counts = Counter(corona_words)
        
        # Get top 100 words
        airline_top100 = set([word for word, _ in airline_word_counts.most_common(100)])
        corona_top100 = set([word for word, _ in corona_word_counts.most_common(100)])
        
        # Calculate overlap
        overlap = airline_top100.intersection(corona_top100)
        overlap_count = len(overlap)
        overlap_percentage = (overlap_count / 100) * 100
        
        # Save results
        with open(output_dir / 'vocabulary_comparison.txt', 'w') as f:
            f.write("VOCABULARY OVERLAP ANALYSIS\n")
            f.write("=" * 40 + "\n\n")
            f.write("Comparing top 100 positive words between:\n")
            f.write("- Airline Dataset\n")
            f.write("- Corona Dataset\n\n")
            f.write(f"OVERLAP RESULTS:\n")
            f.write(f"- Overlapping words: {overlap_count} out of 100\n")
            f.write(f"- Overlap percentage: {overlap_percentage:.1f}%\n\n")
            f.write("OVERLAPPING WORDS:\n")
            f.write("-" * 20 + "\n")
            for word in sorted(overlap):
                f.write(f"- {word}\n")
        
        print(f"✅ Vocabulary overlap analysis completed: {overlap_count}/100 words overlap ({overlap_percentage:.1f}%)")
    
    def run_complete_eda(self):
        """
        Run complete EDA analysis on all datasets.
        """
        print("🚀 STARTING COMPREHENSIVE EDA ANALYSIS")
        print("=" * 80)
        
        # Define dataset files
        datasets = {
            'borderlands': 'cleaned_datasets/processed_dataset_1_borderlands.csv',
            'airline': 'cleaned_datasets/processed_dataset_2_airline.csv',
            'climate': 'cleaned_datasets/processed_dataset_3_climate.csv',
            'corona': 'cleaned_datasets/processed_dataset_4_corona.csv',
            'combined': 'cleaned_datasets/combined_sentiment_dataset.csv'
        }
        
        # Analyze each dataset
        analyzed_datasets = {}
        for dataset_name, filepath in datasets.items():
            df = self.analyze_dataset(filepath, dataset_name)
            if df is not None:
                analyzed_datasets[dataset_name] = df
        
        # Perform comparative analysis
        if 'airline' in analyzed_datasets and 'corona' in analyzed_datasets:
            self.perform_vocabulary_overlap_analysis(
                analyzed_datasets['airline'], 
                analyzed_datasets['corona']
            )
        
        print(f"\n🎉 EDA ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"📁 All results saved to: {self.output_dir}")
        print("=" * 80)

def main():
    """
    Main function to run the complete EDA pipeline.
    """
    # Initialize EDA pipeline
    eda = SentimentEDA()
    
    # Run complete analysis
    eda.run_complete_eda()

if __name__ == "__main__":
    main()
