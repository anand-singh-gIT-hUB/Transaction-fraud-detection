Credit Card Fraud Detection using Random Forest & SMOTE

#Project Overview:
Credit card fraud detection is a critical machine learning problem due to extreme class imbalance and the high cost of misclassification.
This project implements a Random Forest–based fraud detection system using SMOTE oversampling, hyperparameter tuning, and robust evaluation metrics to accurately identify fraudulent transactions.

The pipeline is designed to be research-aligned, industry-ready, and deployment-friendly, with the trained model exported as a .pkl file for real-world integration.

#Objectives
  -Detect fraudulent credit card transactions with high reliability
  -Address severe class imbalance using SMOTE
  -Optimize Random Forest performance using hyperparameter tuning
  -Evaluate model using industry-standard metrics
  -Export the trained model for production deployment

#Methodology
  @Data Preprocessing
   -Removed rows with missing target labels
   -Performed stratified train-test split to preserve class distribution
   -Class Imbalance Handling
   -Applied SMOTE (Synthetic Minority Oversampling Technique) on training data only
   -Prevented data leakage and ensured fair evaluation
   -Model Selection
   -Used Random Forest Classifier (ensemble-based, non-linear, robust to noise)

@Hyperparameter Tuning
  -Employed RandomizedSearchCV
  -Optimized using ROC-AUC score as the primary metric

@Model Evaluation
  -Accuracy
  -Precision
  -Recall
  -F1-score
  -ROC-AUC
  -Confusion Matrix

@Feature Importance Analysis
  -Model Persistence

@Saved trained model as a .pkl file for reuse and deployment

@Evaluation Metrics Used
  Metric	                    Purpose
  Accuracy	        Overall classification performance
  Precision	        Reduces false positives
  Recall	          Captures fraudulent transactions
  F1-Score	        Balance between precision and recall
  ROC-AUC	          Robust metric for imbalanced datasets
  Confusion Matrix	Error distribution analysis
  
@Tech Stack
  -Programming Language: Python
  -Libraries: pandas, numpy, scikit-learn, imbalanced-learn, pickle

@Environment: Google Colab / Jupyter Notebook

📂 Project Structure
├── creditcard.csv
├── random_forest_fraud_model.pkl
├── credit_card_fraud_detection.ipynb
└── README.md

@How to Run the Project
  -Clone the repository:  git clone https://github.com/your-username/credit-card-fraud-detection.git
  -Install dependencies: pip install -r requirements.txt
  -Open the notebook in Google Colab or Jupyter Notebook
  -Run cells sequentially to:
    -Train the model
    -Evaluate performance
    -Generate .pkl file

@Model Deployment
  -The trained model is saved as: random_forest_fraud_model.pkl
  -It can be directly loaded into: 
      -Flask / Django backend, REST APIs
      -Real-time fraud detection systems
          Example:
            import pickle
            with open("random_forest_fraud_model.pkl", "rb") as file:
            model = pickle.load(file)

@Key Highlights
  -SMOTE applied only on training data
  -ROC-AUC–based model selection
  -No data leakage
  -Production-ready pipeline
  -Research and MNC-aligned methodology

@Use Cases
  -Banking fraud prevention systems
  -FinTech transaction monitoring
  -Risk assessment platforms
  -Real-time payment security

@Future Enhancements
  -Comparison with Depp Learning models
  -Cost-sensitive learning
  -Real-time inference API
  -Model monitoring & drift detection

👤 Author

Anand Raj
B.Tech Computer Science & Engineering
Machine Learning | Data Science | Fraud Detection
