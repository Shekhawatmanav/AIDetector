import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
def load_data():
    print("Loading dataset...")
    df = pd.read_csv('../data/ai_vs_human.csv')  # Ensure the correct path to your CSV
    print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
    print("Column names:", df.columns)  # Print the column names to check
    return df

# Preprocess the data
def preprocess_data(df):
    print("Preprocessing data...")

    # Check for missing values in the 'generated' column
    missing_values = df['generated'].isnull().sum()
    if missing_values > 0:
        print(f"Warning: Found {missing_values} missing values in the 'generated' column. Dropping these rows.")
        df = df.dropna(subset=['generated'])  # Drop rows where 'generated' is NaN

    # Use the 'generated' column directly as the label
    df['label'] = df['generated'].astype(int)  # Ensure it's in integer format
    
    X = df['text']  # This column contains the text data
    y = df['label']  # The newly created label column
    print("Data preprocessing completed.")
    return X, y

# Train the model
def train_model():
    print("Training process started...")
    
    # Load and preprocess data
    df = load_data()
    X, y = preprocess_data(df)
    
    # Split data into training and test sets
    print("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split: {len(X_train)} training samples, {len(X_test)} test samples.")
    
    # Convert text to TF-IDF features
    print("Vectorizing text data...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)  # Fit and transform the training data
    X_test_tfidf = vectorizer.transform(X_test)  # Transform the test data
    print("Text data vectorized.")
    
    # Train a Logistic Regression model with class weighting
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, class_weight='balanced')  # Use class weights for imbalance
    model.fit(X_train_tfidf, y_train)  # Fit the model to the training data
    print("Model training completed.")
    
    # Make predictions on the test set
    print("Making predictions on the test set...")
    y_pred = model.predict(X_test_tfidf)
    
    # Evaluate the model
    print("Evaluating the model...")
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy * 100:.2f}%')
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Classification Report
    print(classification_report(y_test, y_pred))
    
    # Save the trained model and vectorizer for later use in prediction
    print("Saving the model and vectorizer...")
    if not os.path.exists('backend'):
        os.makedirs('backend')

    with open('backend/model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    with open('backend/vectorizer.pkl', 'wb') as vec_file:
        pickle.dump(vectorizer, vec_file)
    print("Model and vectorizer saved successfully in the 'backend/' directory.")

# Load the saved model and vectorizer for use in predictions
def load_model():
    print("Loading the saved model and vectorizer...")
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)
    print("Model and vectorizer loaded successfully.")
    return model, vectorizer

# Run the train_model function to train and save the model
if __name__ == '__main__':
    try:
        train_model()
    except Exception as e:
        print(f"An error occurred during training: {e}")
