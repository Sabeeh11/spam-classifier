# SMS Spam Classifier — Naïve Bayes

A spam detection web app built with Naïve Bayes (Bayesian Learning).

## About
- Dataset: SMS Spam Collection (UCI / Kaggle)
- Model: Multinomial Naïve Bayes with Laplace Smoothing
- Features: Bag of Words (CountVectorizer)

## How it works
The model applies Bayes' Theorem to classify messages as spam or ham.
For each class it computes:

**P(class | message) ∝ P(class) × ∏ P(word | class)**

The class with the higher posterior probability wins (MAP decision).

## Built with
- Python
- scikit-learn
- Streamlit

## Course
ICS 502 — Machine Learning
