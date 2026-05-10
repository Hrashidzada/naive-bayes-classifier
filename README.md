# Naive Bayes Classifier (C++)

A multi-class text classification system built from scratch in C++ with no external machine learning libraries. The model predicts the topic of university forum posts based on their word content using a Naive Bayes probabilistic approach.

The goal of this project was to deeply understand how classical machine learning models work at a low level by implementing every component manually.

---

## Key Features

- **Fully from-scratch ML implementation**  
  Implements a Naive Bayes classifier without using any ML frameworks.

- **Multi-class text classification**  
  Predicts topics of forum posts based on word-level probabilities.

- **Custom training pipeline**  
  Learns class priors and word likelihoods directly from labeled CSV data.

- **Robust probability handling**  
  Uses log-probabilities to prevent numerical underflow on large vocabularies.

- **Smoothing for unseen words**  
  Handles unseen or rare words using fallback probability strategies.

---

## How It Works (High Level)

- The model computes prior probabilities for each class based on frequency in the training data.
- It learns word likelihoods for each class by tracking how often words appear in labeled posts.
- During prediction, it scores each class using summed log-probabilities of observed words.
- The class with the highest score is selected as the prediction.

---

## Tech Stack

- C++
- Standard Template Library (STL)
- Custom CSV parsing utility
- Makefile build system

---

## Data

- Trained on labeled university forum posts (CSV format)
- Each entry contains a topic label and post content
- Vocabulary built dynamically from training data

---

## Performance

The classifier achieves strong accuracy on unseen forum posts, correctly predicting post topics based on learned word distributions across classes.

---

## My Role

I built the entire system from scratch, including:

- Implementing the full Naive Bayes algorithm in C++
- Designing the training and prediction pipeline
- Building custom text tokenization and vocabulary handling
- Implementing log-probability scoring for numerical stability
- Structuring data handling using STL containers
- Writing evaluation and testing workflow

---

## What I Learned

- How Naive Bayes classification works at a mathematical and implementation level
- Why log-probabilities are necessary in real-world probabilistic models
- How to design clean, efficient C++ systems using STL
- How to build a complete machine learning pipeline without external libraries
