import pandas as pd
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_model(project_root: Path):
    """Trains a TF-IDF and Logistic Regression model."""
    
    # Load and merge data
    labels_path = project_root / "data" / "labels" / "weak.csv"
    parsed_data_path = project_root / "data" / "parsed.json"
    
    df_labels = pd.read_csv(labels_path)
    df_parsed = pd.read_json(parsed_data_path)
    
    df = pd.merge(df_labels, df_parsed, on='id').dropna(subset=['body_text'])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Prepare data
    X = df['body_text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train and evaluate model
    model = LogisticRegression(class_weight='balanced', random_state=42)
    model.fit(X_train_tfidf, y_train)
    print(classification_report(y_test, model.predict(X_test_tfidf)))
    
    # Save model and vectorizer
    model_dir = project_root / "src" / "models"
    model_dir.mkdir(exist_ok=True)
    joblib.dump(model, model_dir / "tfidf_model.joblib")
    joblib.dump(vectorizer, model_dir / "tfidf_vectorizer.joblib")
    
    print("Model and vectorizer saved successfully.")

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    train_model(project_root)
