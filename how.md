### **Step-by-Step Action Plan**

#### **Phase 1: Foundational Work & Data Preparation (Address Prof's Comments on Pre-processing & Data)**

This phase focuses on getting your data ready and making key decisions based on the feedback.

1.  **Dataset Consolidation and Decisions:**
    *   Download all four datasets mentioned in your proposal.
    *   **Decision on COVID-19 Dataset Classes:** For this intermediate report, simplify the problem.
        *   **Action:** Map the 5 classes to 3.
            *   `very negative` -> `negative`
            *   `negative` -> `negative`
            *   `neutral` -> `neutral`
            *   `positive` -> `positive`
            *   `very positive` -> `positive`
        *   Decide what to do with `mixed`. The easiest approach is to drop these rows for now.
        *   **In your report, you MUST justify this choice.** State: *"To establish a robust baseline and compare models on a consistent classification task, we consolidated the five sentiment labels in the COVID-19 dataset into three standard categories (Positive, Negative, Neutral). The 'mixed' sentiment tweets were excluded from this intermediate analysis to avoid ambiguity. A 5-class classification will be explored as part of our future work for the final report."*

2.  **Create a Standard Pre-processing Pipeline:** This directly answers "What all EDA & data pre-processing will be done?". Create a Python function that takes a tweet (text) and performs the following steps in order:
    *   Convert text to lowercase.
    *   Remove URLs (`http...`).
    *   Remove Twitter mentions (`@username`).
    *   Remove hashtags symbols (`#`) but keep the text (e.g., `#success` becomes `success`).
    *   Remove punctuation and special characters.
    *   Remove numbers.
    *   **Tokenization:** Split text into a list of words.
    *   **Remove Stopwords:** Use a standard library like NLTK or SpaCy.
    *   **Lemmatization:** Convert words to their root form (e.g., "running" -> "run"). This is generally better than stemming.
    *   Apply this pipeline to the 'tweet text' column of your datasets.

3.  **Train-Test Split Strategy (Address Prof's Comment):**
    *   For each dataset, split it into training (80%) and testing (20%) sets.
    *   **Crucially:** Use `stratify` in your split function (`train_test_split` from scikit-learn). This ensures that the proportion of positive, negative, and neutral tweets is the same in both your training and testing sets. This is vital for imbalanced datasets.
    *   **In your report, mention:** *"We used an 80/20 train-test split. To handle the observed class imbalance in the datasets, the split was stratified to maintain the same distribution of sentiment classes in both the training and testing sets, ensuring our evaluation is representative."*

#### **Phase 2: Exploratory Data Analysis (EDA) (Address Prof's Comment on Insights)**

This is where you generate the plots and insights for your report. Use the *training data* for this.

1.  **Class Distribution:** For each dataset, create a bar chart showing the count of 'positive', 'negative', and 'neutral' tweets.
    *   **Insight:** This will visually demonstrate the class imbalance, justifying your use of the weighted F1-score.
2.  **Tweet Length Analysis:** Create histograms showing the distribution of the number of words per tweet for each sentiment class.
    *   **Insight:** Do negative tweets tend to be longer? Are positive tweets short and punchy?
3.  **Word Clouds:** Generate a word cloud for each sentiment class (positive, negative, neutral) for one or two of your key datasets (e.g., Airline and COVID-19).
    *   **Insight:** This provides a powerful visual of the most frequent words associated with each sentiment.
4.  **N-gram Analysis:** Identify the top 10 most common bigrams (two-word phrases) and trigrams (three-word phrases) for each sentiment class.
    *   **Insight:** This goes beyond single words. You might find phrases like "customer service" in negative tweets or "thank you" in positive ones.

#### **Phase 3: Modeling (Tasks Completed: Baseline & Intermediate)**

Now, build and evaluate your models on the pre-processed, split data.

1.  **Feature Extraction:**
    *   **Bag of Words (BoW):** Use `CountVectorizer` for your baseline Logistic Regression model.
    *   **TF-IDF:** Use `TfidfVectorizer` for all your intermediate models (Naive Bayes, SVM, RF, GBDT).

2.  **Model Training:**
    *   **Baseline:** Train a `LogisticRegression` model using the BoW features.
    *   **Intermediate:** Train the following models using the TF-IDF features:
        *   `MultinomialNB` (Naive Bayes)
        *   `LinearSVC` (Support Vector Machine, often faster and better for text)
        *   `RandomForestClassifier` (addresses prof's comment)
        *   `GradientBoostingClassifier` (or `XGBClassifier` if you want to use that library)

3.  **Model Evaluation (Address Prof's Comments on Metrics & Golden Dataset):**
    *   For **each model**, predict sentiments on the **held-out test set**.
    *   Calculate the following metrics:
        *   Accuracy
        *   Precision (for each class)
        *   Recall (for each class)
        *   **Weighted F1-Score** (as requested by the professor). The `classification_report` function in scikit-learn gives you all of these in one go.
    *   Generate a **Confusion Matrix** for each model. This is excellent for visualizing what kinds of errors the model is making (e.g., confusing neutral with positive).

