#!/usr/bin/env python3
"""
Validation script for intermediate modeling results to ensure all required files were generated.
"""

import os
from pathlib import Path

def validate_intermediate_results():
    """
    Validate that all required intermediate modeling files were generated.
    """
    print("🔍 VALIDATING INTERMEDIATE MODELING RESULTS")
    print("=" * 60)
    
    intermediate_dir = Path('modeling_results/intermediate')
    if not intermediate_dir.exists():
        print("❌ Intermediate modeling directory not found!")
        return False
    
    # Required files for each model/dataset combination
    required_files = [
        'classification_report.txt',
        'confusion_matrix.png'
    ]
    
    # Summary files
    summary_files = [
        'comprehensive_comparison_table.csv',
        'comprehensive_comparison_summary.txt',
        'performance_improvements.csv',
        'performance_improvements_analysis.txt'
    ]
    
    models = ['naive_bayes', 'svm']
    datasets = ['borderlands', 'airline', 'climate', 'corona', 'combined']
    
    all_valid = True
    total_files = 0
    
    # Check model/dataset-specific files
    for model in models:
        print(f"\n📁 Checking {model.upper()} model:")
        model_dir = intermediate_dir / model
        
        if not model_dir.exists():
            print(f"❌ Model directory not found: {model_dir}")
            all_valid = False
            continue
        
        for dataset in datasets:
            dataset_dir = model_dir / dataset
            
            if not dataset_dir.exists():
                print(f"❌ Dataset directory not found: {dataset_dir}")
                all_valid = False
                continue
            
            for file in required_files:
                file_path = dataset_dir / file
                if file_path.exists():
                    print(f"✅ {dataset}/{file}")
                    total_files += 1
                else:
                    print(f"❌ Missing: {dataset}/{file}")
                    all_valid = False
    
    # Check summary files
    print(f"\n📊 Checking summary files:")
    for file in summary_files:
        file_path = intermediate_dir / file
        if file_path.exists():
            print(f"✅ {file}")
            total_files += 1
        else:
            print(f"❌ Missing: {file}")
            all_valid = False
    
    print(f"\n📊 VALIDATION SUMMARY:")
    print(f"Total files generated: {total_files}")
    print(f"Expected files: {len(models) * len(datasets) * len(required_files) + len(summary_files)}")
    print(f"All files present: {'✅ YES' if all_valid else '❌ NO'}")
    
    if all_valid:
        print(f"\n🎉 INTERMEDIATE MODELING VALIDATION SUCCESSFUL!")
        print(f"All {total_files} required files are present.")
        
        # Display performance summary
        try:
            import pandas as pd
            comparison_path = intermediate_dir / 'comprehensive_comparison_table.csv'
            if comparison_path.exists():
                df = pd.read_csv(comparison_path)
                print(f"\n📈 INTERMEDIATE MODELING PERFORMANCE SUMMARY:")
                print(df.to_string(index=False, float_format='%.4f'))
        except ImportError:
            print("📈 Performance summary available in comprehensive_comparison_table.csv")
    else:
        print(f"\n⚠️  INTERMEDIATE MODELING VALIDATION FAILED!")
        print("Some required files are missing.")
    
    return all_valid

if __name__ == "__main__":
    validate_intermediate_results()
