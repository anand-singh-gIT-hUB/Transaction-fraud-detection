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
  -Accuracy :  0.9996498354226486
  -Precision : 0.9411764705882353
  -Recall : 0.9142857142857143
  -F1-score : 0.927536231884058
  -ROC-AUC : 0.9545001404099971
  -Confusion Matrix :
      [[14242     2]
      [    3    32]]

@Feature Importance Analysis
  -Model Persistence : 
    14	V14	0.307607
    3	V3	0.118279
    10	V10	0.103907
    17	V17	0.099536
    12	V12	0.076487
    4	V4	0.064551
    16	V16	0.055059
    2	V2	0.033049
    9	V9	0.021208
    7	V7	0.01960

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
├── requirements.txt
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
