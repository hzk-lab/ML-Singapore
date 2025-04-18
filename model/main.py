# ===================== Imports =====================
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from nltk.stem import PorterStemmer

# ===================== Functions =====================

def stem_text(text):
    """Applies stemming to a given sentence using Porter Stemmer."""
    stemmer = PorterStemmer()
    return " ".join([stemmer.stem(word) for word in text.split()])

def train_model(model, X_train, y_train):
    """Trains the pipeline model on training data."""
    model.fit(X_train, y_train)

def predict(model, X_test):
    """Generates predictions from the trained model."""
    return model.predict(X_test)

def generate_result(test, y_pred, filename):
    """Generates a CSV file with predictions."""
    test['Verdict'] = pd.Series(y_pred)
    test.drop(columns=['Text'], inplace=True)
    test.to_csv(filename, index=False)

# ===================== Main Script =====================

def main():
    # Load the dataset
    df = pd.read_csv('/home/huangzekai/桌面/ML Singpaore/ML-Singapore/raw_data/mental_health.csv')

    # Check column names (Optional Debugging)
    print("Available columns:", df.columns)

    # Features and labels
    X = df['text']
    y = df['label']

    # Train-validation split (80% training, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define model pipeline: TF-IDF vectorizer + Logistic Regression
    model = make_pipeline(
        TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            preprocessor=stem_text
        ),
        LogisticRegression(
            solver='saga',
            class_weight='balanced',
            max_iter=1000
        )
    )

    # Step 1: Cross-validation on training data
    print("Performing 5-fold cross-validation on training set...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
    print("CV F1-macro scores:", cv_scores)
    print("Average CV F1-macro score: {:.4f}".format(np.mean(cv_scores)))

    # Step 2: Train on full training set
    print("Training on full training set...")
    train_model(model, X_train, y_train)

    # Step 3: Evaluate on held-out validation set
    print("Evaluating on held-out validation set...")
    y_val_pred = predict(model, X_val)
    val_score = f1_score(y_val, y_val_pred, average='macro')
    print("Validation F1-macro score on held-out set: {:.4f}".format(val_score))

    # Optional: Save predictions to CSV
    val_df = X_val.to_frame()
    val_df['Text'] = X_val
    generate_result(val_df, y_val_pred, '/home/huangzekai/桌面/ML Singpaore/ML-Singapore/output/main_predictions.csv')

# Run script
if __name__ == "__main__":
    main()