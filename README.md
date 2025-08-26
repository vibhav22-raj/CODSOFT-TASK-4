# CODSOFT-TASK-4

# CodSoft Machine Learning Internship  

This repository contains the solutions to three machine learning projects completed during the *CodSoft Internship*. Each task applies ML techniques to solve real-world problems such as fraud detection, churn prediction, and spam classification.  

---

## 📌 Task 4 – Spam Message Classifier  

Classify SMS messages as *spam* or *ham (legitimate)* using Natural Language Processing.  

### Dataset  
- SMS dataset labeled as spam/ham (spam.csv).  
- Preprocessing includes:  
  - Lowercasing text  
  - Removing punctuation & stopwords  
  - Tokenization  
  - TF-IDF vectorization (up to 5000 features, uni- & bi-grams)  

### Model  
- Logistic Regression (with balanced class weights).  

### Results  

#### Classification Report  
| Class | Precision | Recall | F1-score | Support |  
|-------|-----------|--------|----------|---------|  
| Ham   | 0.99      | 0.99   | 0.99     | 903     |  
| Spam  | 0.91      | 0.90   | 0.90     | 131     |  

*Accuracy:* 0.98 (1034 samples)  

#### Confusion Matrix
[[891 12]
[ 13 118]
