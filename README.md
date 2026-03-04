# naive-bayes-classifier
A multi-class Naive Bayes text classifier built from scratch in C++ with no external ML libraries. Trained on university Piazza forum posts, it learns to predict which topic a new post belongs to based on its word content.
How It Works
The classifier implements the Naive Bayes algorithm from the ground up:

Training: Reads a CSV of labeled posts (each row has a tag and content column). For each post, it tokenizes the content into unique words and tracks three things:

Prior probabilities: How often each topic label appears in the training set (e.g., if 15% of posts are labeled "exam", the prior for "exam" is 0.15)
Word likelihoods: How often each word appears in posts of a given label (e.g., how often the word "pointer" shows up in posts labeled "C++")
Vocabulary: The full set of unique words seen across all training posts


Prediction: Given a new post, the classifier scores every possible label by summing the log-prior and the log-likelihoods of each word in the post under that label. The label with the highest total score wins.
Log-probabilities: All probabilities are stored and computed as logarithms. This avoids floating-point underflow that would happen if you multiplied thousands of small raw probabilities together, which matters when the vocabulary exceeds 10,000 unique words.
Smoothing for unseen words: If a word was never seen with a particular label during training, the classifier falls back to the word's overall frequency across all labels. If the word was never seen at all, it falls back to 1 / total_posts. This prevents any single unseen word from zeroing out the entire prediction.

Project Structure
naive-bayes-classifier/
├── main.cpp              # Full classifier implementation
├── csvstream.hpp         # Single-header CSV parsing library
├── train.csv             # Training data (labeled forum posts)
├── test.csv              # Test data (for evaluation)
├── Makefile              # Build configuration
└── README.md
Building and Running
Compile
bashg++ -std=c++17 -O2 -o classifier.exe main.cpp
Or if you have the Makefile:
bashmake
Run in test mode (train + predict + evaluate)
bash./classifier.exe train.csv test.csv
This trains on train.csv, predicts labels for every post in test.csv, and prints the accuracy at the end:
trained on 2400 examples

test data:
  correct = exam, predicted = exam, log-probability score = -1234.56
  content = when is the midterm and what topics does it cover
  ...

performance: 312 / 360 posts predicted correctly
Run in train-only mode (inspect the model)
bash./classifier.exe train.csv
This prints the full training summary: number of posts, vocabulary size, log-prior for each label, and every (label, word) pair with its count and log-likelihood. Useful for debugging and understanding what the model learned.
Implementation Details

Data ingestion: Uses the csvstream.hpp single-header library to parse CSV files. Each row is read into a map<string, string> keyed by column name.
Tokenization: Posts are split on whitespace into unique words using std::set, so duplicate words in a single post are only counted once.
Data structures: Priors and likelihoods are stored in nested std::map containers (map<string, int> for label counts, map<string, map<string, int>> for per-label word counts).
Prediction: For each candidate label, the classifier sums log(P(label)) and log(P(word | label)) for every word in the input. The label with the highest sum is the prediction.
Error handling: Catches csvstream_exception for missing or malformed input files and prints a clean error message.

What I Learned

How Naive Bayes works under the hood, not just calling sklearn.fit() but actually computing priors, likelihoods, and handling edge cases like unseen words
Why log-probabilities matter in practice (raw probability multiplication breaks down fast with large vocabularies)
Writing clean, testable C++ with STL containers and proper memory management
