#!/usr/bin/env python3
"""
Baseline Modeling Pipeline for Sentiment Analysis

This pipeline establishes performance benchmarks using Logistic Regression with 
Bag-of-Words features across all five cleaned datasets.

Features:
- Stratified train/test splits
- CountVectorizer with Bag-of-Words
- Logistic Regression baseline model
- Comprehensive evaluation metrics
- Confusion matrix visualization
- Performance comparison across datasets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import warnings
from pathlib import Path
import time
from typing import Dict, Tuple, List
import re

warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class BaselineModelingPipeline:
    """
    Comprehensive baseline modeling pipeline for sentiment analysis.
    """
    
    def __init__(self, output_dir: str = "modeling_results/baseline"):
        """
        Initialize the baseline modeling pipeline.
        
        Args:
            output_dir: Directory to save all modeling results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
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
        
        # Store results for comparison
        self.results = {}
    
    def load_dataset(self, filepath: str) -> pd.DataFrame:
        """
        Load a cleaned dataset from CSV file.
        
        Args:
            filepath: Path to the cleaned CSV file
            
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
    
    def prepare_features_and_target(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Prepare features (X) and target (y) from the dataset.
        
        Args:
            df: DataFrame with 'clean_text' and 'sentiment' columns
            
        Returns:
            Tuple of (features, target)
        """
        X = df['clean_text']
        y = df['sentiment']
        
        print(f"📊 Features shape: {X.shape}")
        print(f"📊 Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def split_data(self, X: pd.Series, y: pd.Series, test_size: float = 0.2, 
                   random_state: int = 42) -> Tuple:
        """
        Split data into training and testing sets with stratification.
        
        Args:
            X: Features
            y: Target
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y
        )
        
        print(f"📈 Training set: {len(X_train):,} samples")
        print(f"📈 Test set: {len(X_test):,} samples")
        print(f"📈 Training distribution: {y_train.value_counts().to_dict()}")
        print(f"📈 Test distribution: {y_test.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def create_bag_of_words_features(self, X_train: pd.Series, X_test: pd.Series, 
                                   max_features: int = 5000) -> Tuple:
        """
        Create Bag-of-Words features using CountVectorizer.
        
        Args:
            X_train: Training text data
            X_test: Test text data
            max_features: Maximum number of features to keep
            
        Returns:
            Tuple of (X_train_vectorized, X_test_vectorized, vectorizer)
        """
        print(f"🔤 Creating Bag-of-Words features (max_features={max_features})")
        
        # Initialize CountVectorizer
        vectorizer = CountVectorizer(
            max_features=max_features,
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 1)  # Unigrams only for baseline
        )
        
        # Fit and transform training data
        X_train_vectorized = vectorizer.fit_transform(X_train)
        
        # Transform test data (don't fit!)
        X_test_vectorized = vectorizer.transform(X_test)
        
        print(f"🔤 Vocabulary size: {len(vectorizer.vocabulary_):,}")
        print(f"🔤 Training features shape: {X_train_vectorized.shape}")
        print(f"🔤 Test features shape: {X_test_vectorized.shape}")
        
        return X_train_vectorized, X_test_vectorized, vectorizer
    
    def train_logistic_regression(self, X_train_vectorized, y_train: pd.Series) -> LogisticRegression:
        """
        Train a Logistic Regression model.
        
        Args:
            X_train_vectorized: Vectorized training features
            y_train: Training target labels
            
        Returns:
            Trained LogisticRegression model
        """
        print("🤖 Training Logistic Regression model...")
        
        # Initialize Logistic Regression
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'  # Handle class imbalance
        )
        
        # Train the model
        start_time = time.time()
        model.fit(X_train_vectorized, y_train)
        training_time = time.time() - start_time
        
        print(f"🤖 Model trained in {training_time:.2f} seconds")
        print(f"🤖 Model converged: {model.n_iter_[0]} iterations")
        
        return model
    
    def evaluate_model(self, model, X_test_vectorized, y_test: pd.Series, 
                      dataset_name: str) -> Dict:
        """
        Evaluate the trained model and generate performance metrics.
        
        Args:
            model: Trained model
            X_test_vectorized: Vectorized test features
            y_test: Test target labels
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary containing evaluation results
        """
        print(f"📊 Evaluating model on {dataset_name} dataset...")
        
        # Make predictions
        y_pred = model.predict(X_test_vectorized)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        # Generate classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Store results
        results = {
            'accuracy': accuracy,
            'weighted_precision': precision,
            'weighted_recall': recall,
            'weighted_f1': f1,
            'classification_report': class_report,
            'y_test': y_test,
            'y_pred': y_pred,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        print(f"📊 Accuracy: {accuracy:.4f}")
        print(f"📊 Weighted F1-Score: {f1:.4f}")
        
        return results
    
    def save_classification_report(self, results: Dict, dataset_name: str):
        """
        Save classification report to text file.
        
        Args:
            results: Evaluation results dictionary
            dataset_name: Name of the dataset
        """
        output_dir = self.dataset_dirs[dataset_name]
        
        # Convert classification report to string
        class_report_str = classification_report(
            results['y_test'], 
            results['y_pred']
        )
        
        # Create detailed report
        report_content = f"""
BASELINE MODEL PERFORMANCE - {dataset_name.upper()} DATASET
{'='*60}

MODEL: Logistic Regression with Bag-of-Words
FEATURES: CountVectorizer (max_features=5000)
TRAIN/TEST SPLIT: 80/20 (stratified)

OVERALL PERFORMANCE:
- Accuracy: {results['accuracy']:.4f}
- Weighted Precision: {results['weighted_precision']:.4f}
- Weighted Recall: {results['weighted_recall']:.4f}
- Weighted F1-Score: {results['weighted_f1']:.4f}

DETAILED CLASSIFICATION REPORT:
{class_report_str}

CONFUSION MATRIX:
{results['confusion_matrix']}

CLASS DISTRIBUTION IN TEST SET:
{results['y_test'].value_counts().to_dict()}
"""
        
        # Save to file
        with open(output_dir / 'classification_report.txt', 'w') as f:
            f.write(report_content)
        
        print(f"💾 Classification report saved to {output_dir / 'classification_report.txt'}")
    
    def create_confusion_matrix_plot(self, results: Dict, dataset_name: str):
        """
        Create and save confusion matrix visualization.
        
        Args:
            results: Evaluation results dictionary
            dataset_name: Name of the dataset
        """
        output_dir = self.dataset_dirs[dataset_name]
        
        # Get unique labels
        labels = sorted(results['y_test'].unique())
        
        # Create confusion matrix plot
        plt.figure(figsize=(10, 8))
        cm = results['confusion_matrix']
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': 'Count'})
        
        plt.title(f'Confusion Matrix - {dataset_name.title()} Dataset\nLogistic Regression Baseline', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Sentiment', fontsize=12)
        plt.ylabel('True Sentiment', fontsize=12)
        
        # Add accuracy text
        plt.figtext(0.5, 0.02, f'Accuracy: {results["accuracy"]:.4f} | Weighted F1: {results["weighted_f1"]:.4f}', 
                   ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Confusion matrix saved to {output_dir / 'confusion_matrix.png'}")
    
    def model_single_dataset(self, filepath: str, dataset_name: str) -> Dict:
        """
        Complete modeling pipeline for a single dataset.
        
        Args:
            filepath: Path to the cleaned dataset
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary containing modeling results
        """
        print(f"\n{'='*80}")
        print(f"MODELING {dataset_name.upper()} DATASET")
        print(f"{'='*80}")
        
        # Load dataset
        df = self.load_dataset(filepath)
        if df is None:
            return None
        
        # Prepare features and target
        X, y = self.prepare_features_and_target(df)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Create Bag-of-Words features
        X_train_vectorized, X_test_vectorized, vectorizer = self.create_bag_of_words_features(
            X_train, X_test
        )
        
        # Train model
        model = self.train_logistic_regression(X_train_vectorized, y_train)
        
        # Evaluate model
        results = self.evaluate_model(model, X_test_vectorized, y_test, dataset_name)
        
        # Save results
        self.save_classification_report(results, dataset_name)
        self.create_confusion_matrix_plot(results, dataset_name)
        
        # Store results for comparison
        self.results[dataset_name] = {
            'accuracy': results['accuracy'],
            'weighted_f1': results['weighted_f1'],
            'weighted_precision': results['weighted_precision'],
            'weighted_recall': results['weighted_recall'],
            'confusion_matrix': results['confusion_matrix'],
            'class_distribution': y.value_counts().to_dict(),
            'test_size': len(y_test)
        }
        
        print(f"✅ Completed modeling for {dataset_name}")
        return results
    
    def create_baseline_summary_table(self):
        """
        Create a summary table of baseline model performance across all datasets.
        """
        print(f"\n{'='*80}")
        print("CREATING BASELINE SUMMARY TABLE")
        print(f"{'='*80}")
        
        # Create summary DataFrame
        summary_data = []
        for dataset_name, results in self.results.items():
            summary_data.append({
                'Dataset': dataset_name.title(),
                'Test Samples': results['test_size'],
                'Accuracy': results['accuracy'],
                'Weighted F1-Score': results['weighted_f1'],
                'Weighted Precision': results['weighted_precision'],
                'Weighted Recall': results['weighted_recall']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Weighted F1-Score', ascending=False)
        
        # Save summary table
        summary_path = self.output_dir / 'baseline_summary_table.csv'
        summary_df.to_csv(summary_path, index=False)
        
        # Create formatted summary
        summary_content = f"""
BASELINE MODEL PERFORMANCE SUMMARY
{'='*60}

Model: Logistic Regression with Bag-of-Words (CountVectorizer)
Features: 5,000 most frequent unigrams
Train/Test Split: 80/20 (stratified)
Class Weight: Balanced

PERFORMANCE RANKING (by Weighted F1-Score):
{summary_df.to_string(index=False, float_format='%.4f')}

KEY INSIGHTS:
- Best performing dataset: {summary_df.iloc[0]['Dataset']} (F1: {summary_df.iloc[0]['Weighted F1-Score']:.4f})
- Worst performing dataset: {summary_df.iloc[-1]['Dataset']} (F1: {summary_df.iloc[-1]['Weighted F1-Score']:.4f})
- Average F1-Score across all datasets: {summary_df['Weighted F1-Score'].mean():.4f}
- Performance range: {summary_df['Weighted F1-Score'].max() - summary_df['Weighted F1-Score'].min():.4f}

This establishes our baseline performance for comparison with more advanced models.
"""
        
        # Save formatted summary
        with open(self.output_dir / 'baseline_summary.txt', 'w') as f:
            f.write(summary_content)
        
        print(f"📊 Summary table saved to {summary_path}")
        print(f"📊 Formatted summary saved to {self.output_dir / 'baseline_summary.txt'}")
        
        return summary_df
    
    def analyze_baseline_insights(self):
        """
        Analyze baseline model strengths and weaknesses.
        """
        print(f"\n{'='*80}")
        print("ANALYZING BASELINE INSIGHTS")
        print(f"{'='*80}")
        
        insights = []
        
        # Performance analysis
        f1_scores = [results['weighted_f1'] for results in self.results.values()]
        best_dataset = max(self.results.keys(), key=lambda x: self.results[x]['weighted_f1'])
        worst_dataset = min(self.results.keys(), key=lambda x: self.results[x]['weighted_f1'])
        
        insights.append(f"PERFORMANCE ANALYSIS:")
        insights.append(f"- Best performing dataset: {best_dataset.title()} (F1: {self.results[best_dataset]['weighted_f1']:.4f})")
        insights.append(f"- Worst performing dataset: {worst_dataset.title()} (F1: {self.results[worst_dataset]['weighted_f1']:.4f})")
        insights.append(f"- Performance variation: {max(f1_scores) - min(f1_scores):.4f}")
        insights.append("")
        
        # Class imbalance analysis
        insights.append("CLASS IMBALANCE IMPACT:")
        for dataset_name, results in self.results.items():
            class_dist = results['class_distribution']
            total = sum(class_dist.values())
            max_class_pct = max(class_dist.values()) / total * 100
            insights.append(f"- {dataset_name.title()}: {max_class_pct:.1f}% majority class")
        insights.append("")
        
        # Confusion matrix analysis
        insights.append("COMMON MISCLASSIFICATION PATTERNS:")
        for dataset_name, results in self.results.items():
            cm = results['confusion_matrix']
            labels = sorted(results['class_distribution'].keys())
            
            # Find most common misclassification
            max_off_diagonal = 0
            max_error = None
            for i in range(len(cm)):
                for j in range(len(cm)):
                    if i != j and cm[i][j] > max_off_diagonal:
                        max_off_diagonal = cm[i][j]
                        max_error = f"{labels[i]} → {labels[j]}"
            
            insights.append(f"- {dataset_name.title()}: Most common error is {max_error} ({max_off_diagonal} cases)")
        insights.append("")
        
        # Strengths and weaknesses
        insights.append("BASELINE MODEL STRENGTHS:")
        insights.append("- Simple and interpretable (Logistic Regression)")
        insights.append("- Fast training and prediction")
        insights.append("- Good performance on balanced datasets")
        insights.append("- Handles class imbalance with balanced class weights")
        insights.append("")
        
        insights.append("BASELINE MODEL WEAKNESSES:")
        insights.append("- Limited by Bag-of-Words representation (no word order)")
        insights.append("- Struggles with class imbalance in some datasets")
        insights.append("- No consideration of word relationships or context")
        insights.append("- Vocabulary limited to most frequent 5,000 words")
        insights.append("- May overfit to frequent words")
        insights.append("")
        
        insights.append("IMPROVEMENT OPPORTUNITIES:")
        insights.append("- Use TF-IDF instead of raw counts")
        insights.append("- Include n-grams (bigrams, trigrams)")
        insights.append("- Try more sophisticated models (SVM, Random Forest)")
        insights.append("- Use word embeddings (Word2Vec, GloVe)")
        insights.append("- Implement ensemble methods")
        
        # Save insights
        insights_content = "\n".join(insights)
        with open(self.output_dir / 'baseline_insights.txt', 'w') as f:
            f.write(insights_content)
        
        print("💡 Baseline insights saved to baseline_insights.txt")
        print("\n".join(insights[:10]))  # Print first 10 lines
        print("... (see baseline_insights.txt for complete analysis)")
    
    def run_complete_baseline_modeling(self):
        """
        Run complete baseline modeling pipeline on all datasets.
        """
        print("🚀 STARTING BASELINE MODELING PIPELINE")
        print("=" * 80)
        
        # Define dataset files
        datasets = {
            'borderlands': 'cleaned_datasets/processed_dataset_1_borderlands.csv',
            'airline': 'cleaned_datasets/processed_dataset_2_airline.csv',
            'climate': 'cleaned_datasets/processed_dataset_3_climate.csv',
            'corona': 'cleaned_datasets/processed_dataset_4_corona.csv',
            'combined': 'cleaned_datasets/combined_sentiment_dataset.csv'
        }
        
        # Model each dataset
        for dataset_name, filepath in datasets.items():
            self.model_single_dataset(filepath, dataset_name)
        
        # Create summary and analysis
        summary_df = self.create_baseline_summary_table()
        self.analyze_baseline_insights()
        
        print(f"\n🎉 BASELINE MODELING COMPLETED SUCCESSFULLY!")
        print(f"📁 All results saved to: {self.output_dir}")
        print("=" * 80)
        
        return summary_df

def main():
    """
    Main function to run the complete baseline modeling pipeline.
    """
    # Initialize baseline modeling pipeline
    pipeline = BaselineModelingPipeline()
    
    # Run complete modeling
    summary_df = pipeline.run_complete_baseline_modeling()
    
    # Display summary
    print("\n📊 BASELINE PERFORMANCE SUMMARY:")
    print(summary_df.to_string(index=False, float_format='%.4f'))

if __name__ == "__main__":
    main()
