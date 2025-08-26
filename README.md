# CODSOFT-TASK-4

# CodSoft Machine Learning Internship  

This repository contains the solutions to three machine learning projects completed during the *CodSoft Internship*. Each task applies ML techniques to solve real-world problems such as fraud detection, churn prediction, and spam classification.  

---

# SMS Spam Detection System

A machine learning project to classify SMS messages as spam or ham (legitimate) using natural language processing and multiple classification algorithms.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Technical Details](#technical-details)
- [File Structure](#file-structure)
- [API Usage](#api-usage)

## 🎯 Overview

This project implements an SMS spam detection system that:
- Preprocesses SMS text data with advanced NLP techniques
- Trains multiple classification models (Naive Bayes, Logistic Regression, Linear SVC)
- Compares model performance using accuracy and classification reports
- Provides a prediction function for new SMS messages
- Saves trained models for deployment

## 📊 Dataset

The project uses the `spam.csv` dataset with the following characteristics:
- **Source**: SMS Spam Collection Dataset
- **Target Variable**: `label` (ham = 0, spam = 1)
- **Features**: SMS message text content
- **Encoding**: Latin-1 to handle special characters
- **Columns Used**: `v1` (label) and `v2` (message content)

### Dataset Structure:
- **Ham (Legitimate)**: Normal SMS messages from friends, family, businesses
- **Spam**: Promotional messages, scams, unwanted advertisements
- **Preprocessing**: Text cleaning, lemmatization, stopword removal

## 🛠️ Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
nltk>=3.6.0
joblib>=1.1.0
```

## 📥 Installation

1. Clone this repository:
```bash
git clone <your-repository-url>
cd sms-spam-detection
```

2. Install required packages:
```bash
pip install pandas numpy scikit-learn nltk joblib
```

3. Ensure you have the dataset `spam.csv` in the same directory as the script.

4. Run the script (it will automatically download required NLTK data):
```bash
python Code.py
```

## 🚀 Usage

### Training the Models:
```bash
python Code.py
```

### Using the Saved Model:
```python
import joblib

# Load the trained model and vectorizer
model = joblib.load("sms_spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Define the prediction function
def predict_sms(text):
    # Apply the same text cleaning
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)
    return 'Spam' if prediction[0] == 1 else 'Ham'

# Make predictions
result = predict_sms("Congratulations! You've won $1000!")
print(result)  # Output: Spam
```

## 📈 Model Performance

The script evaluates three different models:

### 1. Multinomial Naive Bayes
- **Best for**: Text classification with discrete features
- **Advantages**: Fast training, works well with small datasets
- **Use case**: Baseline model for text classification

### 2. Logistic Regression
- **Best for**: Linear relationships in high-dimensional data
- **Advantages**: Interpretable, probabilistic output
- **Parameters**: `max_iter=1000` for convergence

### 3. Linear Support Vector Classifier (SVC)
- **Best for**: High-dimensional text data
- **Advantages**: Effective for text classification, memory efficient
- **Final Model**: Selected as the primary model

### Evaluation Metrics:
- **Accuracy**: Overall prediction accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

## 🔧 Technical Details

### Text Preprocessing Pipeline:
1. **Lowercase Conversion**: Standardizes text case
2. **Punctuation Removal**: Removes special characters using regex
3. **Tokenization**: Splits text into individual words
4. **Stopword Removal**: Eliminates common English words
5. **Lemmatization**: Reduces words to their root form

```python
def clean_text(text):
    text = text.lower()  
    text = re.sub(r'[^\w\s]', '', text)  
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)
```

### Feature Extraction:
- **TF-IDF Vectorization**: Converts text to numerical features
- **Max Features**: Limited to 3000 most important features
- **Sparse Matrix**: Efficient storage for high-dimensional data

### Data Split:
- **Training Set**: 80% of the data
- **Test Set**: 20% of the data
- **Random State**: 42 for reproducible results

## 📁 File Structure

```
project/
│
├── Code.py                    # Main script
├── spam.csv                   # Dataset (required)
├── sms_spam_model.pkl        # Saved trained model (generated)
├── tfidf_vectorizer.pkl      # Saved TF-IDF vectorizer (generated)
└── README.md                 # This file
```

## 🔮 API Usage

### Prediction Function:
```python
def predict_sms(text):
    """
    Predicts whether an SMS message is spam or ham.
    
    Args:
        text (str): SMS message text
        
    Returns:
        str: 'Spam' or 'Ham'
    """
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)
    return 'Spam' if prediction[0] == 1 else 'Ham'
```

### Example Predictions:
```python
# Spam examples
print(predict_sms("Congratulations! You've won a free iPhone. Call now!"))
# Output: Spam

print(predict_sms("URGENT! Your account will be closed. Click here now!"))
# Output: Spam

# Ham examples  
print(predict_sms("Hi, let's catch up over coffee tomorrow?"))
# Output: Ham

print(predict_sms("Meeting at 3 PM in conference room"))
# Output: Ham
```

## 💡 Key Features

- **Advanced NLP**: Comprehensive text preprocessing pipeline
- **Multiple Models**: Comparison of three different algorithms
- **Production Ready**: Saves both model and vectorizer for deployment
- **Easy Integration**: Simple prediction function for new messages
- **Robust Preprocessing**: Handles various text formats and special characters
- **Memory Efficient**: Uses sparse matrices for feature storage

## 🎯 Expected Output

The script will display:
```
--- Naive Bayes Results ---
Accuracy: 0.xxxx
Classification Report:
              precision    recall  f1-score   support
...

--- Logistic Regression Results ---
Accuracy: 0.xxxx
Classification Report:
...

--- Linear SVC Results ---
Accuracy: 0.xxxx
Classification Report:
...

Test Predictions:
1: Spam
2: Ham
```

## 🔧 Customization

### Adjust TF-IDF Parameters:
```python
vectorizer = TfidfVectorizer(
    max_features=5000,      # Increase vocabulary size
    ngram_range=(1, 2),     # Include bigrams
    min_df=2,               # Minimum document frequency
    max_df=0.95             # Maximum document frequency
)
```

### Modify Text Cleaning:
```python
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)     # Remove URLs
    text = re.sub(r'\d+', '', text)         # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)     # Remove punctuation
    # ... rest of the function
```

## 🚨 Important Notes

- **NLTK Downloads**: Script automatically downloads required NLTK data
- **Encoding**: Uses Latin-1 encoding for the CSV file
- **Model Selection**: Linear SVC is used as the final model
- **Memory Usage**: TF-IDF creates sparse matrices to save memory

## 🔮 Future Enhancements

- Add deep learning models (LSTM, BERT)
- Implement cross-validation
- Add more sophisticated text preprocessing
- Create web API with Flask/FastAPI
- Add confidence scores to predictions
- Implement online learning for model updates
- Add support for multiple languages

## 📄 License

This project is open source and available under the MIT License.
