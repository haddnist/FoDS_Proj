#!/usr/bin/env python3
"""
Validation script for EDA results to ensure all required files were generated.
"""

import os
from pathlib import Path

def validate_eda_results():
    """
    Validate that all required EDA files were generated.
    """
    print("🔍 VALIDATING EDA RESULTS")
    print("=" * 50)
    
    eda_dir = Path('eda_results')
    if not eda_dir.exists():
        print("❌ EDA results directory not found!")
        return False
    
    # Required files for each dataset
    required_files = [
        'summary_stats.txt',
        'sentiment_distribution.png',
        'word_count_histogram.png',
        'word_count_by_sentiment.png',
        'wordcloud_positive.png',
        'wordcloud_negative.png',
        'wordcloud_neutral.png',
        'top_bigrams_positive.png',
        'top_bigrams_negative.png',
        'top_bigrams_neutral.png',
        'top_trigrams_positive.png',
        'top_trigrams_negative.png',
        'top_trigrams_neutral.png'
    ]
    
    # Additional file for combined dataset
    combined_extra = ['vocabulary_comparison.txt']
    
    datasets = ['borderlands', 'airline', 'climate', 'corona', 'combined']
    
    all_valid = True
    total_files = 0
    
    for dataset in datasets:
        print(f"\n📁 Checking {dataset.upper()} dataset:")
        dataset_dir = eda_dir / dataset
        
        if not dataset_dir.exists():
            print(f"❌ Directory not found: {dataset_dir}")
            all_valid = False
            continue
        
        # Check required files
        files_to_check = required_files.copy()
        if dataset == 'combined':
            files_to_check.extend(combined_extra)
        
        for file in files_to_check:
            file_path = dataset_dir / file
            if file_path.exists():
                print(f"✅ {file}")
                total_files += 1
            else:
                print(f"❌ Missing: {file}")
                all_valid = False
    
    print(f"\n📊 VALIDATION SUMMARY:")
    print(f"Total files generated: {total_files}")
    print(f"All files present: {'✅ YES' if all_valid else '❌ NO'}")
    
    if all_valid:
        print(f"\n🎉 EDA VALIDATION SUCCESSFUL!")
        print(f"All {total_files} required files are present.")
    else:
        print(f"\n⚠️  EDA VALIDATION FAILED!")
        print("Some required files are missing.")
    
    return all_valid

if __name__ == "__main__":
    validate_eda_results()
