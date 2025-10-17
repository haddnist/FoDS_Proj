# Sentiment Analysis Project

A comprehensive sentiment analysis project that processes and analyzes sentiment across multiple domains using advanced machine learning techniques.

## 🎯 Project Overview

This project implements a complete sentiment analysis pipeline from data preprocessing to advanced modeling, demonstrating the effectiveness of different feature engineering approaches and machine learning algorithms across diverse domains.

## 📊 Datasets

The project analyzes sentiment across five different domains:

1. **Borderlands** - Gaming sentiment (72,218 samples)
2. **Airline** - Customer service sentiment (14,612 samples)  
3. **Climate** - Environmental sentiment (43,914 samples)
4. **Corona** - COVID-19 sentiment (44,917 samples)
5. **Combined** - All datasets merged (175,661 samples)

## 🏗️ Project Structure

```
proj/
├── dataset/                          # Raw datasets
│   ├── 1.csv                        # Borderlands sentiment
│   ├── 2.csv                        # US Airline sentiment
│   ├── 3.csv                        # Climate change sentiment
│   └── 4/                           # Corona NLP sentiment
│       ├── Corona_NLP_train.csv
│       └── Corona_NLP_test.csv
├── processed_datasets/               # Standardized datasets
├── cleaned_datasets/                 # Preprocessed datasets
├── eda_results/                      # Exploratory Data Analysis results
├── modeling_results/                 # Model performance results
│   ├── baseline/                     # Logistic Regression baseline
│   └── intermediate/                 # Naive Bayes & SVM models
├── text_preprocessing_pipeline.py    # Text cleaning pipeline
├── eda_pipeline.py                   # EDA analysis pipeline
├── baseline_modeling_pipeline.py     # Baseline modeling
├── intermediate_modeling_pipeline.py # Advanced modeling
└── *.md                             # Documentation files
```

## 🚀 Key Features

### **Text Preprocessing Pipeline**
- Lowercase conversion
- URL, mention, and hashtag removal
- Special character and punctuation removal
- Tokenization and stopword removal
- Lemmatization
- Comprehensive noise reduction

### **Exploratory Data Analysis (EDA)**
- Sentiment distribution analysis
- Text length analysis
- Word cloud generation
- N-gram analysis (bigrams and trigrams)
- Vocabulary overlap analysis
- Cross-dataset comparative analysis

### **Modeling Approaches**

#### **Baseline Models**
- **Logistic Regression** with Bag-of-Words features
- 5,000 most frequent unigrams
- Balanced class weights for imbalanced datasets

#### **Intermediate Models**
- **Multinomial Naive Bayes** with TF-IDF + N-grams
- **Linear SVM** with TF-IDF + N-grams
- 15,000 features including bigrams
- Advanced TF-IDF weighting with sublinear scaling

## 📈 Performance Results

### **Baseline Performance (Logistic Regression)**
| Dataset | Accuracy | Weighted F1-Score |
|---------|----------|-------------------|
| Corona | 0.7675 | 0.7724 |
| Borderlands | 0.7391 | 0.7389 |
| Airline | 0.7219 | 0.7332 |
| Climate | 0.6822 | 0.6892 |
| Combined | 0.6629 | 0.6631 |

### **Intermediate Performance (SVM with TF-IDF + N-grams)**
| Dataset | Accuracy | Weighted F1-Score | Improvement |
|---------|----------|-------------------|-------------|
| Borderlands | 0.8362 | 0.8362 | **+13.17%** |
| Corona | 0.7752 | 0.7769 | +0.59% |
| Airline | 0.7598 | 0.7606 | +3.75% |
| Climate | 0.7134 | 0.7143 | +3.65% |
| Combined | 0.7008 | 0.7004 | +5.63% |

## 🔧 Installation & Setup

### **Prerequisites**
- Python 3.8+
- pip package manager

### **Install Required Packages**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk wordcloud
```

### **Download NLTK Data**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab')"
```

## 🏃‍♂️ Usage

### **1. Text Preprocessing**
```bash
python apply_text_preprocessing.py
```

### **2. Exploratory Data Analysis**
```bash
python eda_pipeline.py
```

### **3. Baseline Modeling**
```bash
python baseline_modeling_pipeline.py
```

### **4. Intermediate Modeling**
```bash
python intermediate_modeling_pipeline.py
```

### **5. Validation**
```bash
python validate_eda_results.py
python validate_baseline_results.py
python validate_intermediate_results.py
```

## 📊 Key Insights

### **Feature Engineering Impact**
- **TF-IDF + N-grams** consistently outperformed Bag-of-Words
- **3x vocabulary expansion** (5,000 → 15,000 features) with contextual phrases
- **Bigrams** captured sentiment nuances like "not good", "very bad"

### **Model Performance**
- **SVM** emerged as the best performer across all datasets
- **Average improvement**: +5.36% over baseline
- **Best improvement**: Borderlands dataset (+13.17% F1 improvement)

### **Domain-Specific Patterns**
- **Class imbalance** significantly impacts performance
- **Neutral sentiment** remains the most challenging to classify
- **Domain adaptation** techniques are essential for cross-domain performance

## 📁 Output Files

### **EDA Results**
- 67 visualization files (word clouds, histograms, n-gram charts)
- Sentiment distribution analysis
- Vocabulary overlap analysis

### **Modeling Results**
- 37 model performance files (classification reports, confusion matrices)
- Comprehensive performance comparison tables
- Detailed improvement analysis

  
## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Documentation

- [Text Preprocessing Summary](TEXT_PREPROCESSING_SUMMARY.md)
- [EDA Summary Report](EDA_SUMMARY_REPORT.md)
- [Baseline Modeling Report](BASELINE_MODELING_REPORT.md)
- [Intermediate Modeling Report](INTERMEDIATE_MODELING_REPORT.md)

## 🎯 Future Work

- [ ] Hyperparameter tuning for SVM
- [ ] Ensemble methods implementation
- [ ] Deep learning models (LSTM, BERT)
- [ ] Real-time sentiment analysis API
- [ ] Cross-domain transfer learning

---

**Total Analysis**: 175,661 samples across 5 domains  
**Best Performance**: 83.62% F1-Score (SVM on Borderlands dataset)  
**Key Achievement**: +13.17% improvement over baseline with advanced feature engineering
