# Sentiment Analysis of Online Reviews

Small NLP project comparing **Logistic Regression** and **Multinomial Naive Bayes** on movie review sentiment classification (`Positive` vs `Negative`).

## Project Goal
The goal of this project is to answer a simple question:
> *Which classical ML algorithm works better for sentiment classification of online reviews — Logistic Regression or Naive Bayes — and where do both models make mistakes?*

## Dataset & Preprocessing
- **Dataset:** IMDb Movie Reviews (random sample of 5,000 reviews: 4,000 for training, 1,000 for testing).
- **Preprocessing:** Removed HTML `<br />` tags and punctuation, lowercased the text, and vectorized reviews using `TfidfVectorizer` (`max_features=5000`, English stop words removed).

## Results

Evaluation on the test set (1,000 reviews):

| Model | Accuracy | Precision | Recall | F1-score |
| :--- | :---: | :---: | :---: | :---: |
| Logistic Regression | **84.8%** | 0.8321 | **0.8750** | **0.8530** |
| Naive Bayes | 83.5% | **0.8480** | 0.8194 | 0.8335 |



### Observations
1. **Logistic Regression vs Naive Bayes:**
   Logistic Regression performed slightly better overall (84.8% vs 83.5% accuracy, and 0.853 vs 0.834 F1-score), mainly because of higher recall on positive reviews (0.875 vs 0.819). Naive Bayes assumes every word is independent, while Logistic Regression adjusts weights for all words together, which helps in longer reviews with mixed vocabulary.

2. **Most Important Features (from Logistic Regression weights):**
   - **Positive words:** `great`, `best`, `excellent`, `amazing`, `fun`, `love`, `favorite`, `life`
   - **Negative words:** `bad`, `worst`, `awful`, `waste`, `acting`, `stupid`, `poor`, `boring`

3. **Where the models fail (Error Analysis):**
   When testing the trained models on a few custom sentences, both models handled straightforward reviews well, but failed on negation:

   | Review | Actual | Logistic Regression | Naive Bayes |
   | :--- | :---: | :---: | :---: |
   | *"Amazing product! I absolutely love it."* | Positive | Positive | Positive |
   | *"Terrible quality. Waste of money."* | Negative | Negative | Negative |
   | *"I expected something better from this director, the plot was disappointing."* | Negative | Negative | Negative |
   | *"Not bad at all, I actually enjoyed the story despite a slow start."* | Positive | **Negative** | **Negative** |

   In the last sentence, standard TF-IDF drops `"not"` as a stop word and treats words separately. Because `"bad"` and `"slow"` have strong negative weights, both models predict `Negative` even though the review is positive.

## Tools Used
- Python (`pandas`, `numpy`, `scikit-learn`, `matplotlib`)
- Google Colab
