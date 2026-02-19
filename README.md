# Facilitating Stock Recommendations through Sentiment Analysis

---

## Overview

Can public sentiment predict whether a stock should be bought, held, or sold?

This project builds a hybrid model that combines **17 years of historical MRF stock data** with **sentiment analysis of Indian news headlines** to generate Buy / Hold / Sell recommendations. Five ML models were benchmarked — Random Forest came out on top at **85.02% accuracy**.

---

## How It Works
```
Step 1: Numerical Analysis
└── Pull MRF.NS historical prices via YFinance (2001–2020)
└── Train LSTM on closing prices → predict stock movement

Step 2: Textual Analysis  
└── Load India News Headlines dataset
└── Score each headline using TextBlob (polarity + subjectivity)
└── Visualize sentiment distribution across cities & years

Step 3: Hybrid Model
└── Merge stock data + news headlines by date
└── Score sentiment using VADER (compound, positive, negative, neutral)
└── One-hot encode sentiment labels
└── Train 4 classifiers → compare accuracy
└── Output: BUY / HOLD / SELL
```

---

## Results

| Model | Accuracy |
|---|---|
| **Random Forest** | **85.02%** |
| Logistic Regression | 84.21% |
| Gradient Boosting | 83.40% |
| Decision Tree | 76.52% |

**MRF Sentiment Breakdown:**

| Sentiment | Score |
|---|---|
| Positive | 9.30% |
| Neutral | 81.20% |
| Negative | 9.50% |

> Dominant neutral sentiment → model outputs **HOLD**

---

## Features Used for Classification
```python
['Open', 'High', 'Low', 'Volume',   # Numerical stock data
 'Compound', 'Negative',             # VADER sentiment scores
 'Neutral', 'Positive']              # VADER sentiment scores
```

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-1565c0?style=flat-square&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1565c0?style=flat-square&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-1565c0?style=flat-square&logo=tensorflow&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-1565c0?style=flat-square&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-1565c0?style=flat-square&logo=jupyter&logoColor=white)

| Library | Purpose |
|---|---|
| `yfinance` | Historical stock price data |
| `NLTK` | Tokenization |
| `TextBlob` | Polarity & subjectivity scoring |
| `VADER` | Social media-tuned sentiment scoring |
| `Keras / LSTM` | Sequential stock price prediction |
| `scikit-learn` | Random Forest, Logistic Regression, Decision Tree, Gradient Boosting |

---

## Dataset

- **Numerical:** MRF.NS stock prices (YFinance, 2001–2020)
- **Textual:** India News Headlines CSV (`india-news-headlines.csv`)

---

## Paper

Published at **ICDICI 2024** — International Conference on Data Intelligence and Cognitive Informatics.

*Shlok Bhura, Tanish Bhilare, Dr. Kavita Kelkar — K.J. Somaiya College of Engineering, Mumbai*

[![IEEE](https://img.shields.io/badge/IEEE-View%20Paper-1565c0?style=flat-square&logo=ieee&logoColor=white)](https://ieeexplore.ieee.org/document/10810912)
