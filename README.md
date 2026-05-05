Phishing URL Detection using Machine Learning

Overview:
This project is a simple Machine Learning-based phishing URL detector. It uses basic URL features and a Random Forest Classifier to classify whether a URL is legitimate (safe) or phishing (malicious).
The model is trained on a dataset of URLs and then tested to evaluate its performance. It also allows users to input URLs interactively for real-time prediction.

Features:
-Extracts important features from URLs:
    URL Length
    HTTPS Presence
    Special Characters Count
-Uses Random Forest Algorithm for classification
-Displays:
  Accuracy Score
  Confusion Matrix
-Provides:
  Sample predictions
  Interactive user input for live checking

  
  How It Works
-Load dataset containing URLs and labels
-Extract features from each URL
-Split dataset into training and testing sets
-Train Random Forest model
-Evaluate model performance
-Predict new URLs

Project Structure
  project/
    │── main.py        # Main script
    │── README.md      # Documentation

    
Dataset
The dataset used is: PhiUSIIL Phishing URL Dataset (Kaggle)
Contains:
-URL
-Label (1 = Legitimate, 0 = Phishing)
