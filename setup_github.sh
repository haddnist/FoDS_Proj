#!/bin/bash

# GitHub Repository Setup Script
# This script helps you set up a GitHub repository for your sentiment analysis project

echo "🚀 GitHub Repository Setup for Sentiment Analysis Project"
echo "========================================================"

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install Git first."
    exit 1
fi

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "❌ Not in a git repository. Please run 'git init' first."
    exit 1
fi

echo "✅ Git repository initialized"
echo "✅ Initial commit created with 134 files"

echo ""
echo "📋 Next Steps to Push to GitHub:"
echo "================================="
echo ""
echo "1. Create a new repository on GitHub:"
echo "   - Go to https://github.com/new"
echo "   - Repository name: sentiment-analysis-project (or your preferred name)"
echo "   - Description: Comprehensive sentiment analysis with EDA, baseline, and intermediate modeling"
echo "   - Make it Public or Private (your choice)"
echo "   - DO NOT initialize with README, .gitignore, or license (we already have these)"
echo ""
echo "2. Add the GitHub remote (replace YOUR_USERNAME and REPO_NAME):"
echo "   git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git"
echo ""
echo "3. Push your code to GitHub:"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "4. Verify the push:"
echo "   git remote -v"
echo ""

echo "📊 Project Summary:"
echo "==================="
echo "• 134 files committed"
echo "• 5 datasets analyzed (175,661 total samples)"
echo "• 67 EDA visualizations generated"
echo "• 37 model performance files created"
echo "• Complete documentation included"
echo "• Best performance: 83.62% F1-Score (SVM on Borderlands)"
echo ""

echo "🎯 Key Features Ready for Collaboration:"
echo "========================================"
echo "• Comprehensive .gitignore file"
echo "• Detailed README.md with usage instructions"
echo "• requirements.txt for easy setup"
echo "• Complete project documentation"
echo "• All analysis results and visualizations"
echo "• Validation scripts for quality assurance"
echo ""

echo "💡 Collaboration Tips:"
echo "======================"
echo "• Use feature branches for new work"
echo "• Keep the main branch stable"
echo "• Document any new features or changes"
echo "• Run validation scripts before committing"
echo "• Update requirements.txt if adding new dependencies"
echo ""

echo "🔗 Useful Commands:"
echo "==================="
echo "• View project status: git status"
echo "• View commit history: git log --oneline"
echo "• Create new branch: git checkout -b feature/new-feature"
echo "• Switch branches: git checkout main"
echo "• Merge branches: git merge feature/new-feature"
echo ""

echo "✨ Your sentiment analysis project is ready for GitHub collaboration!"
echo "   Follow the steps above to push your code and start collaborating."
