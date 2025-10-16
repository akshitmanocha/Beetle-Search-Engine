import pandas as pd
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data if not present
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


def preprocess_text(text: str) -> str:
    """Basic text preprocessing for TF-IDF."""
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Lemmatize and remove stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

def train_and_predict(project_root: Path):
    """Trains a TF-IDF model and saves predictions."""
    
    # Load and merge data
    labels_path = project_root / "data" / "labels" / "weak.csv"
    parsed_data_path = project_root / "data" / "parsed.json"
    
    df_labels = pd.read_csv(labels_path)
    df_parsed = pd.read_json(parsed_data_path)
    
    # Merge while only taking the body_text from parsed_data to avoid column conflicts
    df = pd.merge(df_labels, df_parsed[['id', 'body_text']], on='id').dropna(subset=['body_text'])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Preprocessing text data...")
    df['processed_text'] = df['body_text'].apply(preprocess_text)
    
    # Prepare data for training
    X = df['processed_text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Vectorize text
    print("Training TF-IDF model...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train and evaluate model
    model = LogisticRegression(class_weight='balanced', random_state=42)
    model.fit(X_train_tfidf, y_train)
    print("Model evaluation on test set:")
    print(classification_report(y_test, model.predict(X_test_tfidf)))
    
    # Predict on the entire dataset
    print("Generating predictions for the entire dataset...")
    X_full_tfidf = vectorizer.transform(df['processed_text'])
    df['label'] = model.predict(X_full_tfidf)

    # Save the refined labels
    output_path = project_root / "data" / "labels" / "strong.csv"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Select columns to save
    output_df = df[['id', 'url', 'title', 'word_count', 'authors', 'publish_date', 'score', 'label', 'reasoning']]
    output_df.to_csv(output_path, index=False)
    
    print(f"'strong.csv' saved successfully to {output_path}")

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    train_and_predict(project_root)
