#!/usr/bin/env python3
"""
Apply Text Preprocessing to All Sentiment Analysis Datasets

This script applies the comprehensive text preprocessing pipeline to all
processed datasets and saves the cleaned versions.
"""

import pandas as pd
import os
from pathlib import Path
from text_preprocessing_pipeline import TextPreprocessor
import time

def apply_preprocessing_to_datasets():
    """
    Apply text preprocessing to all processed datasets.
    """
    print("=" * 80)
    print("APPLYING TEXT PREPROCESSING TO ALL DATASETS")
    print("=" * 80)
    
    # Initialize preprocessor
    print("Initializing text preprocessor...")
    preprocessor = TextPreprocessor()
    
    # Define input and output directories
    input_dir = Path('processed_datasets')
    output_dir = Path('cleaned_datasets')
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # List of datasets to process
    datasets = [
        'processed_dataset_1_borderlands.csv',
        'processed_dataset_2_airline.csv',
        'processed_dataset_3_climate.csv',
        'processed_dataset_4_corona.csv',
        'combined_sentiment_dataset.csv'
    ]
    
    total_rows_processed = 0
    processing_times = {}
    
    for dataset_file in datasets:
        input_path = input_dir / dataset_file
        output_path = output_dir / dataset_file
        
        if not input_path.exists():
            print(f"⚠️  Skipping {dataset_file} - file not found")
            continue
        
        print(f"\n📁 Processing: {dataset_file}")
        print("-" * 60)
        
        # Load dataset
        start_time = time.time()
        try:
            df = pd.read_csv(input_path)
            print(f"✅ Loaded {len(df):,} rows")
        except Exception as e:
            print(f"❌ Error loading {dataset_file}: {e}")
            continue
        
        # Show sample of original text
        print(f"📝 Sample original text:")
        sample_text = df['text'].iloc[0] if 'text' in df.columns else df.iloc[0, 1]
        print(f"   {sample_text[:100]}{'...' if len(str(sample_text)) > 100 else ''}")
        
        # Apply preprocessing
        try:
            df_cleaned = preprocessor.preprocess_dataframe(df, text_column='text', output_column='clean_text')
            processing_time = time.time() - start_time
            processing_times[dataset_file] = processing_time
            
            print(f"✅ Preprocessing completed in {processing_time:.2f} seconds")
            print(f"📊 Final dataset: {len(df_cleaned):,} rows")
            
            # Show sample of cleaned text
            print(f"🧹 Sample cleaned text:")
            sample_cleaned = df_cleaned['clean_text'].iloc[0]
            print(f"   {sample_cleaned}")
            
            # Save cleaned dataset
            df_cleaned.to_csv(output_path, index=False)
            print(f"💾 Saved to: {output_path}")
            
            total_rows_processed += len(df_cleaned)
            
        except Exception as e:
            print(f"❌ Error processing {dataset_file}: {e}")
            continue
    
    # Summary
    print("\n" + "=" * 80)
    print("PREPROCESSING SUMMARY")
    print("=" * 80)
    print(f"📈 Total rows processed: {total_rows_processed:,}")
    print(f"📁 Output directory: {output_dir}")
    
    print(f"\n⏱️  Processing times:")
    for dataset, time_taken in processing_times.items():
        print(f"   {dataset}: {time_taken:.2f} seconds")
    
    print(f"\n✅ All cleaned datasets saved to: {output_dir}")
    print("=" * 80)

def compare_before_after():
    """
    Compare original vs cleaned text for a few examples.
    """
    print("\n" + "=" * 80)
    print("BEFORE vs AFTER COMPARISON")
    print("=" * 80)
    
    # Load a sample from each cleaned dataset
    cleaned_dir = Path('cleaned_datasets')
    
    datasets_to_compare = [
        'processed_dataset_1_borderlands.csv',
        'processed_dataset_2_airline.csv',
        'processed_dataset_4_corona.csv'
    ]
    
    for dataset_file in datasets_to_compare:
        file_path = cleaned_dir / dataset_file
        if not file_path.exists():
            continue
        
        print(f"\n📁 {dataset_file}")
        print("-" * 60)
        
        df = pd.read_csv(file_path)
        
        # Show 3 examples
        for i in range(min(3, len(df))):
            original = df['text'].iloc[i]
            cleaned = df['clean_text'].iloc[i]
            sentiment = df['sentiment'].iloc[i]
            
            print(f"\nExample {i+1} ({sentiment}):")
            print(f"Original: {original[:150]}{'...' if len(original) > 150 else ''}")
            print(f"Cleaned:  {cleaned}")

def validate_cleaned_datasets():
    """
    Validate the cleaned datasets for quality.
    """
    print("\n" + "=" * 80)
    print("CLEANED DATASETS VALIDATION")
    print("=" * 80)
    
    cleaned_dir = Path('cleaned_datasets')
    
    if not cleaned_dir.exists():
        print("❌ Cleaned datasets directory not found!")
        return
    
    datasets = [
        'processed_dataset_1_borderlands.csv',
        'processed_dataset_2_airline.csv',
        'processed_dataset_3_climate.csv',
        'processed_dataset_4_corona.csv',
        'combined_sentiment_dataset.csv'
    ]
    
    total_rows = 0
    
    for dataset_file in datasets:
        file_path = cleaned_dir / dataset_file
        if not file_path.exists():
            continue
        
        print(f"\n📁 {dataset_file}")
        print("-" * 40)
        
        try:
            df = pd.read_csv(file_path)
            
            # Basic validation
            print(f"✅ Rows: {len(df):,}")
            print(f"✅ Columns: {list(df.columns)}")
            
            # Check for empty cleaned text
            empty_cleaned = (df['clean_text'].str.strip() == '').sum()
            if empty_cleaned > 0:
                print(f"⚠️  Empty cleaned text: {empty_cleaned} rows")
            else:
                print(f"✅ No empty cleaned text")
            
            # Check sentiment distribution
            sentiment_dist = df['sentiment'].value_counts().to_dict()
            print(f"📊 Sentiment distribution: {sentiment_dist}")
            
            # Check average text length
            avg_length = df['clean_text'].str.len().mean()
            print(f"📏 Average cleaned text length: {avg_length:.1f} characters")
            
            total_rows += len(df)
            
        except Exception as e:
            print(f"❌ Error validating {dataset_file}: {e}")
    
    print(f"\n📈 Total validated rows: {total_rows:,}")

def main():
    """
    Main function to apply preprocessing and validate results.
    """
    # Apply preprocessing
    apply_preprocessing_to_datasets()
    
    # Compare before and after
    compare_before_after()
    
    # Validate results
    validate_cleaned_datasets()
    
    print("\n🎉 TEXT PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("Your datasets are now ready for machine learning model training.")

if __name__ == "__main__":
    main()
