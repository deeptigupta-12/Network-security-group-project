import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import re

def extract_features(url):
    """Extracts features from a URL string."""
    # 1. URL Length
    url_length = len(url)
    
    # 2. HTTPS Presence
    has_https = 1 if url.lower().startswith('https') else 0
    
    # 3. Special Characters Count
    # Define a set of special characters often found in URLs
    special_chars = set('@-?=%_.~!*()\'"')
    special_chars_count = sum(1 for char in url if char in special_chars)
    
    return [url_length, has_https, special_chars_count]

def main():
    print("Loading dataset...")
    # Load the Kaggle dataset
    csv_path = r"C:\Users\deept\Downloads\archive\PhiUSIIL_Phishing_URL_Dataset.csv"
    
    try:
        # We only need the URL and label columns. 
        # label: 1 is legitimate, 0 is phishing in this dataset usually, but let's check.
        # Actually, let's load all columns first to be safe, then filter.
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    print(f"Dataset loaded. Total rows: {len(df)}")
    
    # Take a random sample to keep it fast and "simple" as requested, 
    # and because 235k rows might take a bit to process sequentially.
    # Let's use 10,000 samples.
    sample_size = min(10000, len(df))
    df = df.sample(n=sample_size, random_state=42)
    print(f"Using a random sample of {sample_size} rows for faster processing.")

    # Apply our custom feature extraction on the raw URL
    print("Extracting features (URL length, HTTPS presence, Special characters)...")
    
    # Extract features into a new DataFrame
    features_list = df['URL'].apply(extract_features).tolist()
    X = pd.DataFrame(features_list, columns=['URL_Length', 'Has_HTTPS', 'Special_Chars_Count'])
    
    # Target variable (label)
    # The Kaggle dataset uses 'label' where 1=legitimate, 0=phishing. 
    y = df['label']

    # Train/Test Split
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    # Predict and Evaluate
    print("\n--- Results ---")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\n(Note: In this dataset, 1 = Legitimate, 0 = Phishing)")

    # Sample Predictions
    print("\n--- Sample Predictions ---")
    sample_urls = [
        "https://www.google.com",
        "http://secure-login-update-paypal-support.com/login",
        "https://github.com",
        "http://www.bankofamerica.com.update.security.verify.id-8473.xyz"
    ]
    
    sample_features = [extract_features(url) for url in sample_urls]
    sample_X = pd.DataFrame(sample_features, columns=['URL_Length', 'Has_HTTPS', 'Special_Chars_Count'])
    
    predictions = model.predict(sample_X)
    
    for url, pred in zip(sample_urls, predictions):
        label = "Safe (1)" if pred == 1 else "Phishing (0)"
        print(f"URL: {url}")
        print(f"Prediction: {label}\n")

    # Interactive User Input
    print("\n--- Interactive Check ---")
    print("Type 'exit' or 'quit' to stop.")
    while True:
        user_url = input("Enter a URL to check: ").strip()
        if user_url.lower() in ['exit', 'quit']:
            print("Exiting...")
            break
        if not user_url:
            continue
            
        user_features = extract_features(user_url)
        user_X = pd.DataFrame([user_features], columns=['URL_Length', 'Has_HTTPS', 'Special_Chars_Count'])
        
        user_pred = model.predict(user_X)[0]
        user_label = "Safe" if user_pred == 1 else "Phishing"
        
        print(f"Prediction: {user_label}\n")

if __name__ == "__main__":
    main()
