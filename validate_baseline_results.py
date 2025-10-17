#!/usr/bin/env python3
"""
Validation script for baseline modeling results to ensure all required files were generated.
"""

import os
from pathlib import Path

def validate_baseline_results():
    """
    Validate that all required baseline modeling files were generated.
    """
    print("🔍 VALIDATING BASELINE MODELING RESULTS")
    print("=" * 60)
    
    baseline_dir = Path('modeling_results/baseline')
    if not baseline_dir.exists():
        print("❌ Baseline modeling directory not found!")
        return False
    
    # Required files for each dataset
    required_files = [
        'classification_report.txt',
        'confusion_matrix.png'
    ]
    
    # Summary files
    summary_files = [
        'baseline_summary_table.csv',
        'baseline_summary.txt',
        'baseline_insights.txt'
    ]
    
    datasets = ['borderlands', 'airline', 'climate', 'corona', 'combined']
    
    all_valid = True
    total_files = 0
    
    # Check dataset-specific files
    for dataset in datasets:
        print(f"\n📁 Checking {dataset.upper()} dataset:")
        dataset_dir = baseline_dir / dataset
        
        if not dataset_dir.exists():
            print(f"❌ Directory not found: {dataset_dir}")
            all_valid = False
            continue
        
        for file in required_files:
            file_path = dataset_dir / file
            if file_path.exists():
                print(f"✅ {file}")
                total_files += 1
            else:
                print(f"❌ Missing: {file}")
                all_valid = False
    
    # Check summary files
    print(f"\n📊 Checking summary files:")
    for file in summary_files:
        file_path = baseline_dir / file
        if file_path.exists():
            print(f"✅ {file}")
            total_files += 1
        else:
            print(f"❌ Missing: {file}")
            all_valid = False
    
    print(f"\n📊 VALIDATION SUMMARY:")
    print(f"Total files generated: {total_files}")
    print(f"Expected files: {len(datasets) * len(required_files) + len(summary_files)}")
    print(f"All files present: {'✅ YES' if all_valid else '❌ NO'}")
    
    if all_valid:
        print(f"\n🎉 BASELINE MODELING VALIDATION SUCCESSFUL!")
        print(f"All {total_files} required files are present.")
        
        # Display performance summary
        try:
            import pandas as pd
            summary_path = baseline_dir / 'baseline_summary_table.csv'
            if summary_path.exists():
                df = pd.read_csv(summary_path)
                print(f"\n📈 BASELINE PERFORMANCE SUMMARY:")
                print(df.to_string(index=False, float_format='%.4f'))
        except ImportError:
            print("📈 Performance summary available in baseline_summary_table.csv")
    else:
        print(f"\n⚠️  BASELINE MODELING VALIDATION FAILED!")
        print("Some required files are missing.")
    
    return all_valid

if __name__ == "__main__":
    validate_baseline_results()
